"""MCP boundary for Ophelia and Composio actions."""

from __future__ import annotations

import asyncio
import datetime as dt
import os
from typing import Any, Awaitable, Callable

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from composio_calendar import ComposioCalendarClient, ComposioCalendarError
from ophelia_client import OpheliaAPIClient, OpheliaAPIError, redact

load_dotenv()
mcp = FastMCP(
    "ophelia",
    instructions=(
        "Ophelia booking and Composio Google tools. Obtain explicit user "
        "approval before every consequential action."
    ),
)


def _ophelia_client() -> OpheliaAPIClient:
    return OpheliaAPIClient.from_env()


def _composio_client(user_id: str | None = None) -> ComposioCalendarClient:
    resolved = user_id or os.getenv("COMPOSIO_USER_ID") or os.getenv(
        "OPHELIA_USER_ID", "demo_user"
    )
    return ComposioCalendarClient.from_env(user_id=resolved)


async def _ophelia_call(
    operation: Callable[[], Awaitable[dict[str, Any]]],
    transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    try:
        result = await operation()
        return {"ok": True, "data": transform(result) if transform else _bounded_payload(result)}
    except OpheliaAPIError as exc:
        return {
            "ok": False,
            "error": exc.to_state_error(),
            "provider_response": redact(exc.response),
        }
    except RuntimeError as exc:
        return {
            "ok": False,
            "error": {"category": "configuration", "message": str(exc)},
        }


async def _composio_call(
    operation: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    try:
        result = await asyncio.to_thread(operation)
        return {"ok": True, "data": _drop_raw_payloads(result)}
    except (ComposioCalendarError, RuntimeError, ValueError) as exc:
        return {
            "ok": False,
            "error": {"category": "composio", "message": str(exc)},
        }


def _drop_raw_payloads(value: Any) -> Any:
    """Keep actionable fields while excluding verbose third-party responses."""
    if isinstance(value, dict):
        return {
            key: _drop_raw_payloads(item)
            for key, item in value.items()
            if key != "raw"
        }
    if isinstance(value, list):
        return [_drop_raw_payloads(item) for item in value]
    return value


def _bounded_payload(value: Any) -> Any:
    """Bound tool observations so provider responses cannot exhaust model TPM."""
    if isinstance(value, dict):
        return {
            key: _bounded_payload(item)
            for key, item in value.items()
            if key not in {"raw", "image", "photo_url"}
        }
    if isinstance(value, list):
        return [_bounded_payload(item) for item in value[:10]]
    if isinstance(value, str) and len(value) > 1000:
        return value[:1000] + "…"
    return value


def _compact_venue_search(response: dict[str, Any]) -> dict[str, Any]:
    venues = response.get("venues")
    if not isinstance(venues, list):
        return _bounded_payload(response)

    compact = []
    for venue in venues[:5]:
        if not isinstance(venue, dict):
            continue
        compact.append(
            {
                key: _bounded_payload(venue[key])
                for key in (
                    "id",
                    "name",
                    "cuisine",
                    "price",
                    "rating",
                    "address",
                    "city",
                    "state",
                    "provider",
                    "bookable",
                    "available_times",
                    "slots",
                )
                if key in venue
            }
        )
    return {
        "venues": compact,
        "returned_count": len(compact),
        "total_count": response.get("count", len(venues)),
        "truncated": len(venues) > len(compact),
    }


def _compact_availability(response: dict[str, Any]) -> dict[str, Any]:
    slots = response.get("slots")
    if not isinstance(slots, list):
        return _bounded_payload(response)
    return {
        "venue_id": response.get("venue_id"),
        "slots": [
            {
                key: slot[key]
                for key in (
                    "availability_id",
                    "datetime",
                    "party_size",
                    "status",
                    "price_required",
                    "deposit_required",
                )
                if isinstance(slot, dict) and key in slot
            }
            for slot in slots[:7]
            if isinstance(slot, dict)
        ],
        "truncated": len(slots) > 7,
    }


def _consent_error(action: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "category": "consent_required",
            "message": f"Explicit user confirmation is required before {action}.",
        },
    }


def _validation_error(message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {"category": "validation", "message": message},
    }


def _is_iso_datetime(value: str) -> bool:
    try:
        dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    return "T" in value


@mcp.tool()
async def search_venues(
    vertical: str,
    term: str,
    location: str,
    booking_datetime: str,
    party_size: int | None = None,
) -> dict[str, Any]:
    """Search Ophelia venues. Call only after all required arguments are known.

    booking_datetime must be an absolute ISO-8601 date and time, not words such
    as "tomorrow". For dining, party_size is required. Supported verticals are
    "dining" and "fitness".
    """
    vertical = vertical.strip().lower()
    if vertical not in {"dining", "fitness"}:
        return _validation_error("vertical must be 'dining' or 'fitness'.")
    if not term.strip() or not location.strip():
        return _validation_error("term and location must be non-empty.")
    if not _is_iso_datetime(booking_datetime):
        return _validation_error(
            "booking_datetime must be an absolute ISO-8601 date and time."
        )
    if vertical == "dining" and (party_size is None or party_size < 1):
        return _validation_error("party_size is required for dining searches.")

    payload: dict[str, Any] = {
        "vertical": vertical,
        "term": term,
        "location": location,
        "datetime": booking_datetime,
    }
    if party_size is not None:
        payload["party_size"] = party_size
    return await _ophelia_call(
        lambda: _ophelia_client().search_venues(payload),
        _compact_venue_search,
    )


@mcp.tool()
async def search_availability(
    venue_id: str,
    booking_datetime: str,
    party_size: int,
    window_minutes: int = 60,
) -> dict[str, Any]:
    """Check a selected venue. booking_datetime must be absolute ISO-8601."""
    if not venue_id.strip():
        return _validation_error("venue_id is required.")
    if not _is_iso_datetime(booking_datetime):
        return _validation_error(
            "booking_datetime must be an absolute ISO-8601 date and time."
        )
    if party_size < 1:
        return _validation_error("party_size must be at least 1.")
    if window_minutes < 1:
        return _validation_error("window_minutes must be at least 1.")
    payload = {
        "venue_id": venue_id,
        "party_size": party_size,
        "datetime": booking_datetime,
        "window_minutes": window_minutes,
    }
    return await _ophelia_call(
        lambda: _ophelia_client().search_availability(payload),
        _compact_availability,
    )


@mcp.tool()
async def create_booking(
    vertical: str,
    venue_id: str,
    booking_datetime: str,
    customer_name: str,
    customer_email: str,
    customer_phone: str,
    card_number: str,
    card_exp_month: str,
    card_exp_year: str,
    card_cvv: str,
    card_name: str,
    card_postal: str,
    idempotency_key: str,
    user_confirmed: bool,
    party_size: int | None = None,
    billing_address_line1: str = "",
    billing_city: str = "",
    billing_state: str = "",
    billing_country: str = "US",
    password: str | None = None,
) -> dict[str, Any]:
    """Create one Ophelia booking from explicit typed fields.

    Call only after showing the venue, date/time, party size, and customer
    details and receiving explicit confirmation. booking_datetime must be
    absolute ISO-8601. Reuse idempotency_key when retrying the same intention.
    Dining requires party_size. Both verticals require the card fields. Fitness
    also requires the full billing address and password.
    """
    if not user_confirmed:
        return _consent_error("booking")
    vertical = vertical.strip().lower()
    if vertical not in {"dining", "fitness"}:
        return _validation_error("vertical must be 'dining' or 'fitness'.")
    if not venue_id.strip():
        return _validation_error("venue_id is required.")
    if not _is_iso_datetime(booking_datetime):
        return _validation_error(
            "booking_datetime must be an absolute ISO-8601 date and time."
        )
    if not customer_name.strip():
        return _validation_error("customer_name is required.")
    if "@" not in customer_email or "." not in customer_email.rsplit("@", 1)[-1]:
        return _validation_error("customer_email must be a valid email address.")
    if not customer_phone.strip():
        return _validation_error("customer_phone is required.")
    if vertical == "dining" and (party_size is None or party_size < 1):
        return _validation_error("party_size is required for dining bookings.")
    if not idempotency_key.strip():
        return _validation_error("A stable idempotency_key is required.")
    payment_fields = {
        "card_number": card_number,
        "exp_month": card_exp_month,
        "exp_year": card_exp_year,
        "cvv": card_cvv,
        "name_on_card": card_name,
        "postal_code": card_postal,
    }
    if any(not str(value).strip() for value in payment_fields.values()):
        return _validation_error(
            "Card number, expiration month/year, CVV, name, and postal code "
            "are required immediately before booking."
        )
    if vertical == "fitness":
        billing_fields = {
            "address_line1": billing_address_line1,
            "city": billing_city,
            "state": billing_state,
            "country": billing_country or "US",
        }
        if any(not str(value).strip() for value in billing_fields.values()):
            return _validation_error(
                "Fitness bookings require the full billing address."
            )
        if not password:
            return _validation_error("Fitness bookings require the account password.")
        payment_fields.update(billing_fields)

    payload: dict[str, Any] = {
        "vertical": vertical,
        "venue_id": venue_id,
        "datetime": booking_datetime,
        "customer": {
            "name": customer_name,
            "email": customer_email,
            "phone_number": customer_phone,
        },
        "metadata": {"payment": payment_fields},
    }
    if vertical == "dining":
        payload["party_size"] = party_size
    if password:
        payload["metadata"]["password"] = password

    return await _ophelia_call(
        lambda: _ophelia_client().create_booking(payload, idempotency_key)
    )


@mcp.tool()
async def get_booking(booking_id: str) -> dict[str, Any]:
    """Get the authoritative status and next action for a booking."""
    return await _ophelia_call(lambda: _ophelia_client().get_booking(booking_id))


@mcp.tool()
async def continue_booking(
    booking_id: str,
    user_confirmed: bool,
    otp_code: str | None = None,
    payment: dict[str, Any] | None = None,
    password: str | None = None,
) -> dict[str, Any]:
    """Continue a booking with the specific value requested by next_action.

    Provide only the requested OTP, payment object, or password after obtaining
    it from the user and receiving explicit approval.
    """
    if not user_confirmed:
        return _consent_error("booking continuation")
    supplied = sum(value is not None for value in (otp_code, payment, password))
    if supplied != 1:
        return _validation_error(
            "Provide exactly one of otp_code, payment, or password."
        )
    payload: dict[str, Any] = {}
    if otp_code is not None:
        payload["otp_code"] = otp_code
    elif payment is not None:
        payload["payment"] = payment
    elif password is not None:
        payload["password"] = password
    return await _ophelia_call(
        lambda: _ophelia_client().continue_booking(booking_id, payload)
    )


@mcp.tool()
async def cancel_booking(
    booking_id: str,
    user_confirmed: bool,
) -> dict[str, Any]:
    """Cancel a booking only after explicit user confirmation."""
    if not user_confirmed:
        return _consent_error("cancellation")
    return await _ophelia_call(lambda: _ophelia_client().cancel_booking(booking_id))


@mcp.tool()
async def resolve_contact_email(
    guest_name: str,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Resolve a guest email using the user's Google Contacts."""
    return await _composio_call(
        lambda: _composio_client(user_id).resolve_contact_email(guest_name)
    )


@mcp.tool()
async def check_calendar_availability(
    start_iso: str,
    duration_minutes: int,
    current_user_name: str,
    guest_name: str = "",
    guest_email: str = "",
    user_id: str | None = None,
) -> dict[str, Any]:
    """Check the requested interval in the user's and optional guest calendar."""
    return await _composio_call(
        lambda: _composio_client(user_id).check_availability(
            start_iso=start_iso,
            duration_minutes=duration_minutes,
            current_user_name=current_user_name,
            guest_name=guest_name,
            guest_email=guest_email,
        )
    )


@mcp.tool()
async def create_calendar_event(
    event: dict[str, Any],
    user_confirmed: bool,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Create a Google Calendar event after explicit confirmation. Avoid a
    duplicate when the booking provider already created an event."""
    if not user_confirmed:
        return _consent_error("calendar event creation")
    return await _composio_call(
        lambda: _composio_client(user_id).create_event(event=event)
    )


@mcp.tool()
async def send_guest_invite_email(
    event: dict[str, Any],
    user_confirmed: bool,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Send a Gmail guest invitation after showing its recipient and booking
    details and obtaining explicit user confirmation."""
    if not user_confirmed:
        return _consent_error("guest email")
    return await _composio_call(
        lambda: _composio_client(user_id).send_guest_invite_email(event=event)
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
