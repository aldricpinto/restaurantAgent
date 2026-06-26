import os
import logging
from dataclasses import dataclass
from typing import Any

import httpx


logger = logging.getLogger("ophelia.agent")

SENSITIVE_KEYS = {
    "authorization",
    "api_key",
    "card_number",
    "cvv",
    "otp",
    "otp_code",
    "password",
    "phone_number",
    "email",
    "name",
    "customer_name",
    "customer",
    "payment",
}


class OpheliaAPIError(Exception):
    def __init__(
        self,
        *,
        status_code: int | None,
        category: str,
        message: str,
        error_code: str | None = None,
        response: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.category = category
        self.message = message
        self.error_code = error_code
        self.response = response or {}

    def to_state_error(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "status_code": self.status_code,
            "error_code": self.error_code,
            "message": self.message,
        }


@dataclass
class OpheliaAPIClient:
    base_url: str
    api_key: str
    timeout: float = 45.0

    @classmethod
    def from_env(cls) -> "OpheliaAPIClient":
        base_url = os.getenv("OPHELIA_BASE_URL", "https://api.opheliaos.com/v1").rstrip("/")
        api_key = os.getenv("OPHELIA_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPHELIA_API_KEY is required")
        return cls(base_url=base_url, api_key=api_key)

    def _headers(self, idempotency_key: str | None = None) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    async def search_venues(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/venues/search", json=payload, timeout=150)

    async def search_availability(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", "/availability/search", json=payload, timeout=60)

    async def create_booking(self, payload: dict[str, Any], idempotency_key: str) -> dict[str, Any]:
        return await self._request(
            "POST",
            "/bookings",
            json=payload,
            idempotency_key=idempotency_key,
            timeout=180,
        )

    async def get_booking(self, booking_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/bookings/{booking_id}", timeout=150)

    async def continue_booking(self, booking_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", f"/bookings/{booking_id}/continue", json=payload, timeout=180)

    async def cancel_booking(self, booking_id: str) -> dict[str, Any]:
        return await self._request("POST", f"/bookings/{booking_id}/cancel", timeout=180)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = self._headers(idempotency_key)
        safe_headers = redact(headers)
        timeout_value = timeout or self.timeout
        logger.debug(
            "ophelia_api request: method=%s path=%s url=%s timeout=%s headers=%s body=%s",
            method,
            path,
            url,
            timeout_value,
            safe_headers,
            redact(json or {}),
        )
        try:
            async with httpx.AsyncClient(timeout=timeout_value) as client:
                response = await client.request(
                    method,
                    url,
                    headers=headers,
                    json=json,
                )
        except httpx.TimeoutException as exc:
            logger.debug(
                "ophelia_api timeout: method=%s path=%s url=%s timeout=%s body=%s error=%r",
                method,
                path,
                url,
                timeout_value,
                redact(json or {}),
                exc,
            )
            raise OpheliaAPIError(
                status_code=None,
                category="network",
                message="The Ophelia API request timed out.",
                error_code="timeout",
            ) from exc
        except httpx.HTTPError as exc:
            logger.debug(
                "ophelia_api http_error: method=%s path=%s url=%s timeout=%s body=%s error=%r",
                method,
                path,
                url,
                timeout_value,
                redact(json or {}),
                exc,
            )
            raise OpheliaAPIError(
                status_code=None,
                category="network",
                message="Could not reach the Ophelia API.",
                error_code="connection_error",
            ) from exc

        try:
            body = response.json() if response.content else {}
        except ValueError as exc:
            logger.debug(
                "ophelia_api invalid_json_response: method=%s path=%s status=%s text=%s",
                method,
                path,
                response.status_code,
                response.text[:2000],
            )
            raise OpheliaAPIError(
                status_code=response.status_code,
                category="network",
                message="Ophelia returned a non-JSON response.",
                error_code="invalid_json",
            ) from exc

        logger.debug(
            "ophelia_api response: method=%s path=%s status=%s body=%s",
            method,
            path,
            response.status_code,
            redact(body),
        )

        if response.status_code >= 400:
            error = self._error_from_response(response.status_code, body)
            logger.debug(
                "ophelia_api error_response: method=%s path=%s status=%s category=%s error_code=%s body=%s",
                method,
                path,
                response.status_code,
                error.category,
                error.error_code,
                redact(body),
            )
            raise error

        return body

    def _error_from_response(self, status_code: int, body: dict[str, Any]) -> OpheliaAPIError:
        error_code = _extract_error_code(body)
        if status_code == 400:
            category = "validation"
            message = "Ophelia could not validate the request. Please check the booking details."
        elif status_code in (401, 403):
            category = "auth"
            message = "The Ophelia API credentials are not authorized for this request."
        elif status_code == 402:
            category = "billing"
            message = "The Ophelia account cannot complete this request because of billing or allowance limits."
        elif status_code == 429:
            category = "rate_limit"
            message = "Ophelia or the provider is rate limiting this request. Please try again shortly."
        elif status_code >= 500:
            category = "network"
            message = "Ophelia returned a temporary server error."
        else:
            category = "provider"
            message = "Ophelia could not complete the provider request."

        return OpheliaAPIError(
            status_code=status_code,
            category=category,
            message=message,
            error_code=error_code,
            response=body,
        )


def _extract_error_code(body: dict[str, Any]) -> str | None:
    for key in ("error_code", "code"):
        value = body.get(key)
        if isinstance(value, str):
            return value

    error = body.get("error")
    if isinstance(error, dict):
        for key in ("error_code", "code", "type"):
            value = error.get(key)
            if isinstance(value, str):
                return value
    if isinstance(error, str):
        return error

    return None


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if key.lower() in SENSITIVE_KEYS:
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = redact(item)
        return redacted
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value
