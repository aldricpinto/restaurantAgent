import asyncio
import datetime as dt
import json
import os
import re
import sys
import uuid
from typing import Any, Literal, NotRequired

import dateparser
import redis
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from ophelia_client import OpheliaAPIClient, OpheliaAPIError, redact
from utils.logger import agent_logger as logger


load_dotenv()


TERMINAL_STATUSES = {"confirmed", "failed", "cancelled", "expired", "rate_limited", "error", "verification_required"}
MAX_POLL_ATTEMPTS = 5
POST_CONTINUE_SETTLE_SECONDS = 180
POST_CONTINUE_SETTLE_INTERVAL_SECONDS = 10
WAITABLE_PROVIDER_ERROR_CODES = {"not_confirmed_after_otp"}
DESIRED_LOCATIONS = [
    {
        "key": "central_park",
        "label": "Central Park",
        "search_location": "Central Park, New York, NY",
        "aliases": [
            "central park",
            "nyc near central park",
            "new york near central park",
            "near central park",
            "near central park in nyc",
            "near central park in new york",
        ],
    },
    {
        "key": "soho",
        "label": "SoHo",
        "search_location": "SoHo, New York, NY",
        "aliases": ["soho", "soho nyc", "soho new york", "soho manhattan"],
    },
    {
        "key": "times_square",
        "label": "Times Square",
        "search_location": "Times Square, New York, NY",
        "aliases": [
            "times square",
            "nyc near times square",
            "new york near times square",
            "near times square",
        ],
    },
    {
        "key": "west_village",
        "label": "West Village",
        "search_location": "West Village, New York, NY",
        "aliases": ["west village", "west village nyc", "west village new york"],
    },
]
LOCATION_CHOICES = [
    f"{idx}. {location['label']} — {location['search_location']}"
    for idx, location in enumerate(DESIRED_LOCATIONS, start=1)
]


class BookingState(TypedDict, total=False):
    user_request: str
    vertical: str
    term: str
    raw_location: str
    desired_location_key: str
    location: str
    datetime_phrase: str
    booking_datetime: str
    party_size: int
    preferences: dict[str, Any]
    missing_fields: list[str]
    venues: list[dict[str, Any]]
    selected_venue_id: str
    selected_venue: dict[str, Any]
    availability: dict[str, Any]
    consented: bool
    consented_at: str
    customer_name: str
    idempotency_key: str
    booking_id: str
    booking_status: str
    booking_response: dict[str, Any]
    next_action: dict[str, Any]
    final_summary: str
    booking_succeeded: bool
    cancelled: bool
    cancelled_on: str
    error: dict[str, Any]
    poll_attempts: int


redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
r = redis.from_url(redis_url)


def can_book(venue_id: str, max_attempts: int = 3, time_window: int = 600) -> bool:
    if venue_id == "unknown":
        return True
    key = f"rate_limit:{venue_id}"
    try:
        cc = r.get(key)
        if cc is None:
            r.set(key, 1, ex=time_window)
            return True
        cc_int = int(cc)
        if cc_int >= max_attempts:
            return False
        r.incr(key)
        return True
    except (redis.exceptions.ConnectionError, ValueError, TypeError) as e:
        logger.warning("Redis rate limiter unavailable for key %s: %s. Defaulting to True.", key, e)
        return True


def get_cool_off_period(venue_id: str) -> int:
    if venue_id == "unknown":
        return 0
    scheds = [60, 300, 1800, 3600]
    key = f"failures:{venue_id}"
    try:
        failures = r.get(key)
        if not failures:
            return 0
        failures_count = int(failures)
        if failures_count == 0:
            return 0
        return scheds[min(failures_count - 1, len(scheds) - 1)]
    except (redis.exceptions.ConnectionError, ValueError, TypeError) as e:
        logger.warning("Redis cool-off unavailable for key %s: %s. Defaulting to 0.", key, e)
        return 0


def increment_failures(venue_id: str, expire_time: int = 3600) -> None:
    if venue_id == "unknown":
        return
    key = f"failures:{venue_id}"
    try:
        r.incr(key)
        r.expire(key, expire_time)
    except redis.exceptions.ConnectionError as e:
        logger.warning("Redis increment_failures unavailable for key %s: %s", key, e)


def clear_failures(venue_id: str) -> None:
    if venue_id == "unknown":
        return
    key = f"failures:{venue_id}"
    try:
        r.delete(key)
    except redis.exceptions.ConnectionError as e:
        logger.warning("Redis clear_failures unavailable for key %s: %s", key, e)


def now_iso() -> str:
    return dt.datetime.now().replace(microsecond=0).isoformat()


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_datetime(datetime_phrase: str, user_request: str) -> str | None:
    phrase = (datetime_phrase or "").strip()
    if not phrase:
        return None

    base = dt.datetime.now()
    settings = {
        "PREFER_DATES_FROM": "future",
        "RELATIVE_BASE": base,
        "RETURN_AS_TIMEZONE_AWARE": False,
    }
    parsed = dateparser.parse(phrase, settings=settings)
    if parsed is None and phrase.lower() in user_request.lower():
        parsed = dateparser.parse(user_request, settings=settings)
    if parsed is None:
        return None
    return parsed.replace(microsecond=0).isoformat()


def format_api_datetime(datetime_value: str) -> str:
    """
    Ophelia availability expects an ISO timestamp with a Z suffix.
    The MVP stores local wall time as YYYY-MM-DDTHH:MM:SS, so format it
    consistently for the REST payload without changing the selected time.
    """
    value = (datetime_value or "").strip()
    if not value:
        return value
    if value.endswith("Z"):
        return value
    if re.search(r"[+-]\d{2}:\d{2}$", value):
        return value
    return f"{value}Z"


def parse_desired_location(location_phrase: str, user_request: str = "") -> tuple[str, str]:
    """
    This function will help match natural-language location text against the user's predefined
    desired locations. Returns (search_location, desired_location_key).
    """
    source = f"{location_phrase or ''} {user_request or ''}".lower()
    source = re.sub(r"\s+", " ", source).strip()
    if not source:
        return "", ""

    alias_candidates: list[tuple[int, dict[str, Any]]] = []
    for location in DESIRED_LOCATIONS:
        aliases = [location["label"], location["search_location"], *location["aliases"]]
        for alias in aliases:
            alias_lower = alias.lower()
            if alias_lower and alias_lower in source:
                alias_candidates.append((len(alias_lower), location))

    if not alias_candidates:
        return "", ""

    _, selected = max(alias_candidates, key=lambda item: item[0])
    return selected["search_location"], selected["key"]


def location_from_answer(answer: Any, fallback_location: str = "", fallback_key: str = "") -> tuple[str, str]:
    raw = str(answer or "").strip()
    if not raw:
        return fallback_location, fallback_key

    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(DESIRED_LOCATIONS):
            selected = DESIRED_LOCATIONS[idx]
            return selected["search_location"], selected["key"]

    matched_location, matched_key = parse_desired_location(raw)
    if matched_location:
        return matched_location, matched_key

    return raw, ""


def required_missing(state: BookingState) -> list[str]:
    missing: list[str] = []
    for field in ("term", "location", "booking_datetime", "party_size"):
        if not state.get(field):
            missing.append(field)
    return missing


def venue_display(venue: dict[str, Any]) -> str:
    address = ", ".join(
        str(venue.get(key, "")).strip()
        for key in ("address", "city", "state")
        if venue.get(key)
    )
    provider = venue.get("provider") or venue.get("source") or "unknown provider"
    return f"{venue.get('name', 'Unknown venue')} ({venue.get('id')}) — {address} [{provider}]"


def selected_provider(state: BookingState) -> str:
    return str(
        state.get("selected_venue", {}).get("provider")
        or state.get("selected_venue", {}).get("source")
        or ""
    ).strip().lower()


def requested_time_label(booking_datetime: str) -> str:
    try:
        parsed = dt.datetime.fromisoformat((booking_datetime or "").replace("Z", ""))
    except ValueError:
        return ""
    return parsed.strftime("%I:%M %p").lstrip("0")


def safe_error_state(error: OpheliaAPIError) -> dict[str, Any]:
    return {
        "booking_status": "error",
        "error": error.to_state_error(),
        "booking_response": {},
    }


def booking_error_code(response: dict[str, Any]) -> str | None:
    for key in ("error_code", "code"):
        value = response.get(key)
        if isinstance(value, str):
            return value

    error = response.get("error")
    if isinstance(error, dict):
        for key in ("error_code", "code", "type"):
            value = error.get(key)
            if isinstance(value, str):
                return value
    if isinstance(error, str):
        return error

    metadata = response.get("metadata")
    if isinstance(metadata, dict):
        for key in ("error_code", "code"):
            value = metadata.get(key)
            if isinstance(value, str):
                return value

    return None


def has_confirmation_evidence(response: dict[str, Any]) -> bool:
    if not isinstance(response, dict):
        return False

    provider_booking_id = response.get("provider_booking_id")
    if isinstance(provider_booking_id, str) and provider_booking_id.strip():
        return True

    confirmation = response.get("confirmation")
    if isinstance(confirmation, dict):
        for key in ("confirmation_code", "confirmation_number", "number", "code"):
            value = confirmation.get(key)
            if isinstance(value, str) and value.strip():
                return True

    metadata = response.get("metadata")
    if isinstance(metadata, dict):
        for key in ("confirmation_code", "confirmation_number", "provider_booking_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return True

    return False


def public_status_from_booking_response(response: dict[str, Any]) -> str:
    status = response.get("status", "unknown")
    if status == "confirmed" and not has_confirmation_evidence(response):
        return "verification_required"
    return status


async def main() -> None:
    api = OpheliaAPIClient.from_env()
    model = ChatGroq(model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))

    async def extract_intent_node(state: BookingState) -> dict[str, Any]:
        logger.info("extract_intent_node: parsing user request")
        user_request = state.get("user_request", "")
        prompt = f"""
You extract booking intent for the Ophelia API.

Return ONLY valid JSON with these keys:
- vertical: "dining", "fitness", or "entertainment"
- term: restaurant/cuisine/activity/event search term
- location: city/neighborhood/state/country text
- datetime_phrase: exact natural language date/time phrase from the user, e.g. "today at 7pm"
- party_size: integer number of people/tickets when present, otherwise null
- preferences: object of soft preferences
- missing_fields: array of missing required fields for a dining booking

Rules:
- MVP supports dining only, but still classify the vertical.
- Do not invent missing values.
- For "Italian place in SoHo" term is "Italian" and location is "SoHo".
- Keep datetime_phrase as natural language; do not normalize it.

User request: {user_request}
""".strip()

        last_error: Exception | None = None
        parsed: dict[str, Any] = {}
        for attempt in range(2):
            try:
                resp = await model.ainvoke([("user", prompt)])
                parsed = parse_json_object(resp.content)
                break
            except Exception as exc:
                last_error = exc
                logger.warning("extract_intent_node: parse attempt %d failed: %s", attempt + 1, exc)
        else:
            logger.warning("extract_intent_node: falling back to clarification after parse failure: %s", last_error)
            return {"missing_fields": ["term", "location", "datetime", "party_size"]}

        vertical = parsed.get("vertical") or "dining"
        datetime_phrase = parsed.get("datetime_phrase") or ""
        booking_datetime = normalize_datetime(datetime_phrase, user_request)
        raw_location = parsed.get("location") or ""
        location, desired_location_key = parse_desired_location(raw_location, user_request)
        party_size = parsed.get("party_size")
        try:
            party_size = int(party_size) if party_size is not None else None
        except (TypeError, ValueError):
            party_size = None

        update: BookingState = {
            "vertical": vertical,
            "term": parsed.get("term") or "",
            "raw_location": raw_location,
            "desired_location_key": desired_location_key,
            "location": location,
            "datetime_phrase": datetime_phrase,
            "preferences": parsed.get("preferences") or {},
            "venues": [],
            "selected_venue_id": "",
            "selected_venue": {},
            "availability": {},
            "consented": False,
            "customer_name": "",
            "idempotency_key": "",
            "booking_id": "",
            "booking_status": "",
            "booking_response": {},
            "next_action": {},
            "error": {},
            "poll_attempts": 0,
        }
        if booking_datetime:
            update["booking_datetime"] = booking_datetime
        if party_size:
            update["party_size"] = party_size

        update["missing_fields"] = required_missing(update)
        logger.info("extract_intent_node: extracted intent=%s", redact(update))
        return update

    def clarify_node(state: BookingState) -> dict[str, Any]:
        missing = required_missing(state)
        logger.info("clarify_node: missing fields=%s", missing)
        answers = interrupt(
            {
                "message": "I need a few details before searching.",
                "fields_needed": missing,
                "known": {
                    "term": state.get("term"),
                    "raw_location": state.get("raw_location"),
                    "location": state.get("location"),
                    "datetime_phrase": state.get("datetime_phrase"),
                    "booking_datetime": state.get("booking_datetime"),
                    "party_size": state.get("party_size"),
                },
                "location_choices": LOCATION_CHOICES,
            }
        )

        user_request = state.get("user_request", "")
        term = answers.get("term") or state.get("term", "")
        raw_location = answers.get("location") or answers.get("raw_location") or state.get("raw_location", "")
        location, desired_location_key = location_from_answer(
            raw_location,
            fallback_location=state.get("location", ""),
            fallback_key=state.get("desired_location_key", ""),
        )
        datetime_phrase = answers.get("datetime") or answers.get("datetime_phrase") or state.get("datetime_phrase", "")
        booking_datetime = normalize_datetime(datetime_phrase, f"{user_request} {datetime_phrase}")
        party_size = answers.get("party_size") or state.get("party_size")
        try:
            party_size = int(party_size) if party_size else None
        except (TypeError, ValueError):
            party_size = None

        update: BookingState = {
            "term": term,
            "raw_location": raw_location,
            "desired_location_key": desired_location_key,
            "location": location,
            "datetime_phrase": datetime_phrase,
            "missing_fields": [],
        }
        if booking_datetime:
            update["booking_datetime"] = booking_datetime
        if party_size:
            update["party_size"] = party_size
        update["missing_fields"] = required_missing(update | {k: v for k, v in state.items() if k not in update})
        return update

    def route_after_intent(state: BookingState) -> Literal["clarify", "search"]:
        return "clarify" if required_missing(state) else "search"

    def route_after_search(state: BookingState) -> Literal["select_venue", "summary"]:
        return "summary" if state.get("booking_status") in TERMINAL_STATUSES else "select_venue"

    def route_after_availability(state: BookingState) -> Literal["preflight_and_consent", "summary"]:
        return "summary" if state.get("booking_status") in TERMINAL_STATUSES else "preflight_and_consent"

    async def search_node(state: BookingState) -> dict[str, Any]:
        payload = {
            "vertical": "dining",
            "term": state["term"],
            "location": state["location"],
            "datetime": state["booking_datetime"],
            "party_size": int(state["party_size"]),
        }
        logger.info("search_node: payload=%s", redact(payload))
        try:
            response = await api.search_venues(payload)
        except OpheliaAPIError as exc:
            logger.warning("search_node: Ophelia API error=%s", exc.to_state_error())
            return safe_error_state(exc)

        venues = response.get("venues", [])
        if not venues:
            return {
                "venues": [],
                "booking_status": "failed",
                "error": {"category": "user_input", "message": "No venues found for that request."},
            }
        logger.info("search_node: received %d venues", len(venues))
        return {"venues": venues[:10], "booking_status": ""}

    def select_venue_node(state: BookingState) -> dict[str, Any]:
        venues = state.get("venues", [])
        if not venues:
            return {}

        choices = [f"{idx}. {venue_display(venue)}" for idx, venue in enumerate(venues[:5], start=1)]
        selection = interrupt(
            {
                "message": "Pick a venue from the search results.",
                "choices": choices,
                "fields_needed": ["selection"],
            }
        )

        raw_choice = str(selection.get("selection", "1")).strip()
        try:
            choice_idx = int(raw_choice) - 1
        except ValueError:
            choice_idx = 0
        choice_idx = max(0, min(choice_idx, len(venues[:5]) - 1))
        selected = venues[choice_idx]
        return {
            "selected_venue_id": selected.get("id", ""),
            "selected_venue": selected,
        }

    async def availability_node(state: BookingState) -> dict[str, Any]:
        venue_id = state.get("selected_venue_id")
        if not venue_id:
            return {
                "booking_status": "failed",
                "error": {"category": "validation", "message": "No selected venue ID found."},
            }

        if selected_provider(state) == "opentable":
            selected_venue = state.get("selected_venue", {})
            inline_times = selected_venue.get("available_times") or []
            requested_time = requested_time_label(state.get("booking_datetime", ""))
            logger.info(
                "availability_node: skipping standalone availability for OpenTable; inline_times=%s requested_time=%s",
                inline_times,
                requested_time,
            )
            if not inline_times:
                return {
                    "booking_status": "failed",
                    "error": {
                        "category": "provider_terminal",
                        "message": "OpenTable did not return inline availability for the selected venue.",
                    },
                    "availability": {
                        "source": "venues/search.inline",
                        "provider": "opentable",
                        "available_times": [],
                    },
                }

            return {
                "availability": {
                    "source": "venues/search.inline",
                    "provider": "opentable",
                    "available_times": inline_times,
                    "requested_time": requested_time,
                    "requested_time_available": requested_time in inline_times if requested_time else None,
                }
            }

        payload = {
            "venue_id": venue_id,
            "party_size": int(state["party_size"]),
            "datetime": format_api_datetime(state["booking_datetime"]),
            "window_minutes": 60,
        }
        logger.info("availability_node: payload=%s", redact(payload))
        try:
            response = await api.search_availability(payload)
        except OpheliaAPIError as exc:
            if exc.status_code == 404:
                logger.info("availability_node: availability not supported/found, continuing to consent")
                return {"availability": {"skipped": True, "reason": exc.error_code or "not_available"}}
            logger.warning("availability_node: Ophelia API error=%s", exc.to_state_error())
            return safe_error_state(exc)
        return {"availability": response}

    def preflight_and_consent_node(state: BookingState) -> dict[str, Any]:
        venue = state.get("selected_venue", {})
        consent = interrupt(
            {
                "message": "Please confirm before I create a real booking.",
                "booking_details": {
                    "venue": venue_display(venue),
                    "datetime": state.get("booking_datetime"),
                    "party_size": state.get("party_size"),
                },
                "fields_needed": ["approved"],
            }
        )
        approved = str(consent.get("approved", "")).strip().lower() in {"y", "yes", "true", "approve", "approved"}
        if not approved:
            return {
                "consented": False,
                "booking_status": "cancelled",
                "booking_response": {"status": "cancelled", "reason": "User declined booking consent."},
            }
        return {"consented": True, "consented_at": now_iso()}

    def route_after_consent(state: BookingState) -> Literal["create", "summary"]:
        return "create" if state.get("consented") else "summary"

    async def create_booking_node(state: BookingState) -> dict[str, Any]:
        venue_id = state.get("selected_venue_id") or "unknown"
        if not can_book(venue_id):
            logger.warning("create_booking_node: rate limit exceeded for venue %s", venue_id)
            return {
                "booking_status": "rate_limited",
                "error": {
                    "category": "rate_limit",
                    "message": f"Booking request blocked due to rate limits for venue {venue_id}.",
                },
            }

        wait_time = get_cool_off_period(venue_id)
        if wait_time > 0:
            logger.info("create_booking_node: cool-off active for venue %s; sleeping %d seconds", venue_id, wait_time)
            print(f"\n[Cooldown] Waiting {wait_time} seconds before trying this provider again...")
            await asyncio.sleep(wait_time)

        details = interrupt(
            {
                "message": "Please provide the customer details for this booking request.",
                "fields_needed": ["name", "email", "phone", "card_number", "card_exp_month", "card_exp_year", "card_cvv", "card_name", "card_postal"],
            }
        )

        idempotency_key = state.get("idempotency_key") or str(uuid.uuid4())
        payload: dict[str, Any] = {
            "vertical": "dining",
            "venue_id": venue_id,
            "datetime": state["booking_datetime"],
            "party_size": int(state["party_size"]),
            "customer": {
                "name": details["name"],
                "email": details["email"],
                "phone_number": details["phone"],
            },
            "metadata": {},
        }
        if details.get("card_number"):
            payload["metadata"]["payment"] = {
                "card_number": details.get("card_number"),
                "exp_month": details.get("card_exp_month"),
                "exp_year": details.get("card_exp_year"),
                "cvv": details.get("card_cvv"),
                "name_on_card": details.get("card_name"),
                "postal_code": details.get("card_postal"),
            }

        logger.info("create_booking_node: creating booking payload=%s idempotency_key=%s", redact(payload), idempotency_key)
        try:
            response = await api.create_booking(payload, idempotency_key)
        except OpheliaAPIError as exc:
            logger.warning("create_booking_node: Ophelia API error=%s", exc.to_state_error())
            return safe_error_state(exc) | {"idempotency_key": idempotency_key}

        status = public_status_from_booking_response(response)
        if status == "failed":
            increment_failures(venue_id)
        elif status in {"confirmed", "requires_action", "processing", "verification_required"}:
            clear_failures(venue_id)

        return {
            "idempotency_key": idempotency_key,
            "customer_name": details["name"],
            "booking_id": response.get("id", ""),
            "booking_status": status,
            "booking_response": response,
            "next_action": response.get("next_action") or {},
            "poll_attempts": 0,
        }

    def route_after_status(state: BookingState) -> Literal["poll", "action", "summary"]:
        status = state.get("booking_status", "")
        if status == "processing":
            return "poll"
        if status == "requires_action":
            return "action"
        return "summary"

    async def poll_booking_node(state: BookingState) -> dict[str, Any]:
        booking_id = state.get("booking_id")
        attempts = int(state.get("poll_attempts", 0))
        if not booking_id:
            return {"booking_status": "error", "error": {"category": "validation", "message": "Missing booking ID for polling."}}
        if attempts >= MAX_POLL_ATTEMPTS:
            return {
                "booking_status": "error",
                "poll_attempts": attempts,
                "booking_response": state.get("booking_response", {}),
                "error": {"category": "network", "message": "Booking is still processing after the polling limit."},
            }

        await asyncio.sleep(min(2**attempts, 10))
        try:
            response = await api.get_booking(booking_id)
        except OpheliaAPIError as exc:
            logger.warning("poll_booking_node: Ophelia API error=%s", exc.to_state_error())
            return safe_error_state(exc)
        return {
            "booking_status": public_status_from_booking_response(response),
            "booking_response": response,
            "next_action": response.get("next_action") or {},
            "poll_attempts": attempts + 1,
        }

    async def reconcile_booking_after_timeout(booking_id: str, source: str) -> dict[str, Any]:
        logger.info("%s: reconciling booking_id=%s after timeout/network error", source, booking_id)
        try:
            response = await api.get_booking(booking_id)
        except OpheliaAPIError as exc:
            logger.warning("%s: reconciliation failed=%s", source, exc.to_state_error())
            return safe_error_state(exc) | {"booking_id": booking_id}

        return {
            "booking_id": booking_id,
            "booking_status": public_status_from_booking_response(response),
            "booking_response": response,
            "next_action": response.get("next_action") or {},
            "poll_attempts": 0,
        }

    async def wait_for_provider_settle_after_continue(
        booking_id: str,
        initial_response: dict[str, Any],
    ) -> dict[str, Any]:
        error_code = booking_error_code(initial_response)
        if initial_response.get("status") in TERMINAL_STATUSES or error_code not in WAITABLE_PROVIDER_ERROR_CODES:
            return {
                "booking_status": public_status_from_booking_response(initial_response),
                "booking_response": initial_response,
                "next_action": initial_response.get("next_action") or {},
            }

        logger.info(
            "continue_booking_node: provider returned waitable failure error_code=%s; polling booking_id=%s for up to %ss",
            error_code,
            booking_id,
            POST_CONTINUE_SETTLE_SECONDS,
        )
        print(
            "\nProvider has not reached the confirmation page yet. "
            f"I'll wait up to {POST_CONTINUE_SETTLE_SECONDS} seconds and keep checking the booking status..."
        )

        deadline = dt.datetime.now() + dt.timedelta(seconds=POST_CONTINUE_SETTLE_SECONDS)
        latest_response = initial_response
        while dt.datetime.now() < deadline:
            await asyncio.sleep(POST_CONTINUE_SETTLE_INTERVAL_SECONDS)
            try:
                latest_response = await api.get_booking(booking_id)
            except OpheliaAPIError as exc:
                logger.warning("continue_booking_node: settle poll failed=%s", exc.to_state_error())
                continue

            latest_status = public_status_from_booking_response(latest_response)
            latest_error_code = booking_error_code(latest_response)
            logger.info(
                "continue_booking_node: settle poll status=%s error_code=%s",
                latest_status,
                latest_error_code,
            )

            if latest_status in TERMINAL_STATUSES or latest_error_code not in WAITABLE_PROVIDER_ERROR_CODES:
                return {
                    "booking_id": booking_id,
                    "booking_status": latest_status,
                    "booking_response": latest_response,
                    "next_action": latest_response.get("next_action") or {},
                    "poll_attempts": 0,
                }

        logger.info(
            "continue_booking_node: settle window expired for booking_id=%s; accepting latest failed status",
            booking_id,
        )
        return {
            "booking_id": booking_id,
            "booking_status": public_status_from_booking_response(latest_response),
            "booking_response": latest_response,
            "next_action": latest_response.get("next_action") or {},
            "poll_attempts": 0,
        }

    async def continue_booking_node(state: BookingState) -> dict[str, Any]:
        booking_id = state.get("booking_id")
        next_action = state.get("next_action") or {}
        action_type = next_action.get("type", "otp")
        if not booking_id:
            return {"booking_status": "error", "error": {"category": "validation", "message": "Missing booking ID for continuation."}}

        if action_type == "payment":
            prompt = {
                "message": f"Booking {booking_id} requires payment details.",
                "booking_id": booking_id,
                "next_action": next_action,
                "fields_needed": ["card_number", "card_exp_month", "card_exp_year", "card_cvv", "card_name", "card_postal"],
            }
            payment = interrupt(prompt)
            payload = {
                "payment": {
                    "card_number": payment.get("card_number"),
                    "exp_month": payment.get("card_exp_month"),
                    "exp_year": payment.get("card_exp_year"),
                    "cvv": payment.get("card_cvv"),
                    "name_on_card": payment.get("card_name"),
                    "postal_code": payment.get("card_postal"),
                }
            }
        else:
            otp = interrupt(
                {
                    "message": f"Booking {booking_id} requires an OTP.",
                    "booking_id": booking_id,
                    "next_action": next_action,
                    "fields_needed": ["otp_code"],
                }
            )
            payload = {"otp_code": otp.get("otp_code") if isinstance(otp, dict) else str(otp)}

        logger.info("continue_booking_node: continuing booking_id=%s action_type=%s", booking_id, action_type)
        try:
            response = await api.continue_booking(booking_id, payload)
        except OpheliaAPIError as exc:
            logger.warning("continue_booking_node: Ophelia API error=%s", exc.to_state_error())
            if exc.category == "network":
                return await reconcile_booking_after_timeout(booking_id, "continue_booking_node")
            return safe_error_state(exc)

        return await wait_for_provider_settle_after_continue(booking_id, response)

    async def summary_node(state: BookingState) -> dict[str, Any]:
        status = state.get("booking_status", "unknown")
        succeeded = status == "confirmed"
        venue_name = state.get("selected_venue", {}).get("name") or "Unknown venue"
        booking_details = (
            "Booking Details:\n"
            f"- Restaurant Search Result: {venue_name}\n"
            f"- Guest Name: {state.get('customer_name') or 'Not provided'}\n"
            f"- Reserved Date & Time: {state.get('booking_datetime')}\n"
            f"- Party Size: {state.get('party_size')}\n"
            f"- Booking ID: {state.get('booking_id')}\n"
            f"- Final status: {status}\n"
        )
        grounded = {
            "status": status,
            "booking_id": state.get("booking_id"),
            "venue": venue_name,
            "datetime": state.get("booking_datetime"),
            "party_size": state.get("party_size"),
            "error": state.get("error"),
            "booking_response": redact(state.get("booking_response", {})),
        }
        prompt = f"""
Write a short user-facing booking result.
Use only the grounded JSON. Do not imply success unless status is confirmed.
If status is verification_required, explicitly say the booking could not be verified as confirmed yet.
If failed/error/rate_limited/expired, be honest and include the sanitized error code if present.
Please ensure you replace any bracketed placeholders like [Insert Date], [Insert Time],
[Guest], [Insert Party Size], or [Guest Name] with the actual details provided below.
Do not invent a guest name; use the guest name from Booking Details only if present.

{booking_details}

Grounded JSON:
{json.dumps(grounded, indent=2)}
""".strip()
        resp = await model.ainvoke([("user", prompt)])
        print("\n" + "=" * 60)
        print(resp.content)
        print("=" * 60 + "\n")
        return {"final_summary": resp.content, "booking_succeeded": succeeded}

    def waiting_node(state: BookingState) -> dict[str, Any]:
        user_input = interrupt(
            {
                "message": "Is there anything else I can help you with? (e.g. 'cancel my booking' or type 'exit')",
                "fields_needed": ["message"],
            }
        )
        return {"user_request": user_input["message"]}

    def route_after_user_input(state: BookingState) -> Literal["cancel", "new_booking", "end"]:
        query = state.get("user_request", "").lower().strip()
        if "cancel" in query:
            return "cancel"
        if "exit" in query or "quit" in query:
            return "end"
        return "new_booking"

    async def cancel_node(state: BookingState) -> dict[str, Any]:
        booking_id = state.get("booking_id")
        if not booking_id:
            return {"booking_status": "none", "final_summary": "No active booking to cancel."}
        try:
            response = await api.cancel_booking(booking_id)
        except OpheliaAPIError as exc:
            logger.warning("cancel_node: Ophelia API error=%s", exc.to_state_error())
            return safe_error_state(exc)
        return {
            "booking_id": booking_id,
            "booking_status": response.get("status", "unknown"),
            "booking_response": response,
            "cancelled_on": now_iso(),
            "cancelled": response.get("status") == "cancelled",
        }

    workflow = StateGraph(BookingState)
    workflow.add_node("extract_intent", extract_intent_node)
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("search", search_node)
    workflow.add_node("select_venue", select_venue_node)
    workflow.add_node("availability", availability_node)
    workflow.add_node("preflight_and_consent", preflight_and_consent_node)
    workflow.add_node("create", create_booking_node)
    workflow.add_node("poll", poll_booking_node)
    workflow.add_node("action", continue_booking_node)
    workflow.add_node("summary", summary_node)
    workflow.add_node("wait_for_user", waiting_node)
    workflow.add_node("cancel", cancel_node)

    workflow.add_edge(START, "extract_intent")
    workflow.add_conditional_edges("extract_intent", route_after_intent, {"clarify": "clarify", "search": "search"})
    workflow.add_conditional_edges("clarify", route_after_intent, {"clarify": "clarify", "search": "search"})
    workflow.add_conditional_edges("search", route_after_search, {"select_venue": "select_venue", "summary": "summary"})
    workflow.add_edge("select_venue", "availability")
    workflow.add_conditional_edges(
        "availability",
        route_after_availability,
        {"preflight_and_consent": "preflight_and_consent", "summary": "summary"},
    )
    workflow.add_conditional_edges("preflight_and_consent", route_after_consent, {"create": "create", "summary": "summary"})
    workflow.add_conditional_edges("create", route_after_status, {"poll": "poll", "action": "action", "summary": "summary"})
    workflow.add_conditional_edges("poll", route_after_status, {"poll": "poll", "action": "action", "summary": "summary"})
    workflow.add_conditional_edges("action", route_after_status, {"poll": "poll", "action": "action", "summary": "summary"})
    workflow.add_edge("summary", "wait_for_user")
    workflow.add_edge("cancel", "summary")
    workflow.add_conditional_edges(
        "wait_for_user",
        route_after_user_input,
        {"cancel": "cancel", "new_booking": "extract_intent", "end": END},
    )

    app = workflow.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    try:
        png_graph = app.get_graph().draw_mermaid_png()
        with open("workflow.png", "wb") as f:
            f.write(png_graph)
    except Exception as exc:
        logger.error("Failed to draw graph: %s", exc)

    print("\n" + "=" * 60)
    print("Ophelia Booking Agent")
    print("=" * 60)

    user_request = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    if not user_request:
        user_request = input("What's the word?\n> ").strip()

    result = await app.ainvoke({"user_request": user_request}, config=config)

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        logger.info("Agent paused on interrupt: %s", payload.get("message"))
        resume_value = prompt_for_interrupt(payload)
        if isinstance(resume_value, dict) and resume_value.get("message", "").lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        result = await app.ainvoke(Command(resume=resume_value), config=config)

    logger.info("Agent run finished")


def prompt_for_interrupt(payload: dict[str, Any]) -> Any:
    print(f"\n{payload.get('message', 'Input needed')}")

    if payload.get("choices"):
        for choice in payload["choices"]:
            print(choice)

    if payload.get("location_choices"):
        print("\nLocation choices:")
        for choice in payload["location_choices"]:
            print(choice)

    if payload.get("booking_details"):
        print("\nBooking details:")
        for key, value in payload["booking_details"].items():
            print(f"- {key}: {value}")

    fields = payload.get("fields_needed")
    if not fields:
        return input("> ").strip()

    if fields == ["message"]:
        return {"message": input("> ").strip()}

    answers: dict[str, Any] = {}
    for field in fields:
        label = field.replace("_", " ").title()
        if field == "approved":
            answers[field] = input("Approve booking? Type yes/no: ").strip()
        elif field == "selection":
            answers[field] = input("Selection number: ").strip()
        elif field == "location":
            answers[field] = input("Location number or custom location: ").strip()
        elif field == "card_cvv":
            answers[field] = input(f"{label}: ").strip()
        elif field == "otp_code":
            answers[field] = input("OTP code: ").strip()
        else:
            answers[field] = input(f"{label}: ").strip()
    return answers


if __name__ == "__main__":
    asyncio.run(main())
