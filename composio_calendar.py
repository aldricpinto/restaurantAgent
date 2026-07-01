import datetime as dt
import html
import json
import os
import re
from urllib.parse import quote_plus
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_groq import ChatGroq


class ComposioCalendarError(Exception):
    pass


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    return parsed if isinstance(parsed, dict) else {"result": parsed}


def _last_message_content(result: Any) -> str:
    if isinstance(result, dict):
        messages = result.get("messages") or []
        if messages:
            last = messages[-1]
            return str(getattr(last, "content", last.get("content") if isinstance(last, dict) else last))
        return str(result)
    return str(result)


def _iter_values(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_values(child)


def _first_email_from_contacts_payload(payload: Any) -> tuple[str, str]:
    for item in _iter_values(payload):
        email_addresses = item.get("emailAddresses")
        if not isinstance(email_addresses, list):
            continue
        emails = [str(email.get("value") or "").strip() for email in email_addresses if isinstance(email, dict)]
        email = next((candidate for candidate in emails if candidate), "")
        if not email:
            continue
        names = item.get("names") if isinstance(item.get("names"), list) else []
        display_name = ""
        if names and isinstance(names[0], dict):
            display_name = str(names[0].get("displayName") or names[0].get("unstructuredName") or "").strip()
        return email, display_name
    return "", ""


def _best_error_message(payload: Any) -> str:
    fallback = ""
    for item in _iter_values(payload):
        message = item.get("message")
        if message:
            return str(message)
        error = item.get("error")
        if error and not fallback:
            fallback = str(error)
    return fallback or str(payload)


def _contains_interval(container: dict[str, Any], start: dt.datetime, end: dt.datetime) -> bool:
    free_start = dt.datetime.fromisoformat(str(container.get("start", "")).replace("Z", "+00:00"))
    free_end = dt.datetime.fromisoformat(str(container.get("end", "")).replace("Z", "+00:00"))
    probe_start = start
    probe_end = end
    if free_start.tzinfo and probe_start.tzinfo is None:
        probe_start = probe_start.replace(tzinfo=free_start.tzinfo)
        probe_end = probe_end.replace(tzinfo=free_start.tzinfo)
    return free_start <= probe_start and probe_end <= free_end


def _parse_iso(value: str) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _display_datetime(value: str) -> str:
    parsed = _parse_iso(value)
    if not parsed:
        return value
    return parsed.strftime("%A, %B %-d, %Y at %-I:%M %p")


def _calendar_datetime(value: str) -> str:
    parsed = _parse_iso(value)
    if not parsed:
        return ""
    return parsed.strftime("%Y%m%dT%H%M%S")


def _google_calendar_link(event: dict[str, Any]) -> str:
    start = _calendar_datetime(str(event.get("start") or ""))
    end = _calendar_datetime(str(event.get("end") or ""))
    if not start or not end:
        return ""
    params = {
        "action": "TEMPLATE",
        "text": str(event.get("title") or "Dinner"),
        "dates": f"{start}/{end}",
        "location": str(event.get("location") or ""),
        "details": str(event.get("description") or ""),
    }
    query = "&".join(f"{key}={quote_plus(value)}" for key, value in params.items())
    return f"https://calendar.google.com/calendar/render?{query}"


@dataclass
class ComposioCalendarClient:
    user_id: str
    model_name: str = "openai/gpt-oss-120b"
    toolkit: str = "googlecalendar"
    contacts_toolkit: str = "googlecontacts"
    gmail_toolkit: str = "gmail"

    @classmethod
    def from_env(cls, user_id: str | None = None) -> "ComposioCalendarClient":
        api_key = os.getenv("COMPOSIO_API_KEY", "").strip()
        if not api_key:
            raise ComposioCalendarError("COMPOSIO_API_KEY is required for calendar integration.")
        return cls(
            user_id=user_id or os.getenv("COMPOSIO_USER_ID") or os.getenv("OPHELIA_USER_ID", "demo_user"),
            model_name=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            toolkit=os.getenv("COMPOSIO_CALENDAR_TOOLKIT", "googlecalendar"),
            contacts_toolkit=os.getenv("COMPOSIO_CONTACTS_TOOLKIT", "googlecontacts"),
            gmail_toolkit=os.getenv("COMPOSIO_GMAIL_TOOLKIT", "gmail"),
        )

    def _auth_config_id(self, toolkit: str) -> str:
        normalized = re.sub(r"[^A-Z0-9]+", "_", toolkit.upper()).strip("_")
        candidates = [
            f"COMPOSIO_{normalized}_AUTH_CONFIG_ID",
        ]
        if toolkit == self.toolkit:
            candidates.append("COMPOSIO_CALENDAR_AUTH_CONFIG_ID")
        if toolkit == self.contacts_toolkit:
            candidates.append("COMPOSIO_CONTACTS_AUTH_CONFIG_ID")
        if toolkit == self.gmail_toolkit:
            candidates.append("COMPOSIO_GMAIL_AUTH_CONFIG_ID")
        for key in candidates:
            value = os.getenv(key, "").strip()
            if value:
                return value
        return ""

    def _session(self, toolkits: list[str] | None = None) -> Any:
        os.environ.setdefault("COMPOSIO_CACHE_DIR", "/private/tmp/ophelia-composio-cache")
        try:
            from composio import Composio
            from composio_langgraph import LanggraphProvider
        except ImportError as exc:
            raise ComposioCalendarError(
                "Install composio and composio_langgraph to use calendar integration."
            ) from exc

        composio = Composio(provider=LanggraphProvider())
        requested_toolkits = toolkits or [self.toolkit]
        auth_configs = {toolkit: auth_id for toolkit in requested_toolkits if (auth_id := self._auth_config_id(toolkit))}
        if self.contacts_toolkit in requested_toolkits and self.contacts_toolkit not in auth_configs:
            raise ComposioCalendarError(
                "Google Contacts requires a Composio auth config. Set COMPOSIO_CONTACTS_AUTH_CONFIG_ID "
                "from your Composio dashboard, or use an env email fallback for demos."
            )
        try:
            return composio.create(
                user_id=self.user_id,
                toolkits=requested_toolkits,
                auth_configs=auth_configs or None,
                sandbox={"enable": False},
            )
        except TypeError:
            # Older/newer SDKs may expose a narrower create() signature.
            return composio.create(user_id=self.user_id)

    def _agent(self, toolkits: list[str] | None = None) -> Any:
        session = self._session(toolkits=toolkits)
        tools = session.tools()
        llm = ChatGroq(model_name=self.model_name)
        return create_agent(tools=tools, model=llm)

    def _execute_tool(self, toolkit: str, tool_slug: str, arguments: dict[str, Any], *, step: str) -> dict[str, Any]:
        tools = {tool.name: tool for tool in self._session(toolkits=[toolkit]).tools()}
        execute = tools.get("COMPOSIO_MULTI_EXECUTE_TOOL")
        if execute is None:
            raise ComposioCalendarError(f"Composio router did not expose COMPOSIO_MULTI_EXECUTE_TOOL for {toolkit}.")
        result = execute.invoke(
            {
                "tools": [{"tool_slug": tool_slug, "arguments": arguments}],
                "thought": f"Execute {tool_slug} for the Ophelia booking flow.",
                "current_step": step,
                "current_step_metric": "1/1 tools",
            }
        )
        if not isinstance(result, dict):
            raise ComposioCalendarError(f"Unexpected Composio response: {result}")
        if result.get("successful") is False:
            raise ComposioCalendarError(f"{tool_slug} failed: {_best_error_message(result)}")
        return result

    def _authorize_toolkit(self, toolkit: str, label: str) -> None:
        session = self._session(toolkits=[toolkit])
        try:
            request = session.authorize(toolkit)
        except AttributeError as exc:
            raise ComposioCalendarError("This Composio SDK version does not expose session.authorize().") from exc

        url = (
            getattr(request, "redirect_url", None)
            or getattr(request, "url", None)
            or getattr(request, "auth_url", None)
            or str(request)
        )
        print(f"Open this URL to connect {label}:")
        print(url)
        wait = getattr(request, "wait_for_connection", None)
        if callable(wait):
            print(f"Waiting for {label} connection to complete...")
            try:
                wait()
            except Exception as exc:
                name = exc.__class__.__name__
                if "Timeout" in name:
                    print(f"{label} connection is still pending. Finish the browser flow, then rerun --connect-calendar or start the agent.")
                    return
                raise ComposioCalendarError(f"{label} connection failed: {exc}") from exc
            print(f"{label} connected.")

    def connect_google_calendar(self) -> None:
        self._authorize_toolkit(self.toolkit, "Google Calendar")
        try:
            self._authorize_toolkit(self.contacts_toolkit, "Google Contacts")
        except ComposioCalendarError as exc:
            print(f"Google Contacts connection skipped or failed: {exc}")
        try:
            self._authorize_toolkit(self.gmail_toolkit, "Gmail")
        except ComposioCalendarError as exc:
            print(f"Gmail connection skipped or failed: {exc}")

    def resolve_contact_email(self, guest_name: str) -> dict[str, Any]:
        if not guest_name.strip():
            return {"found": False, "name": "", "email": "", "source": "none", "reason": "No guest name provided."}

        # Google People search can return stale/missing results unless the search cache is warmed first.
        try:
            self._execute_tool(
                self.contacts_toolkit,
                "GOOGLECONTACTS_SEARCH_CONTACTS",
                {"query": "", "page_size": 0, "read_mask": "names,emailAddresses"},
                step="WARM_CONTACTS_SEARCH",
            )
        except ComposioCalendarError:
            # The real search below should surface any actionable error.
            pass

        result = self._execute_tool(
            self.contacts_toolkit,
            "GOOGLECONTACTS_SEARCH_CONTACTS",
            {"query": guest_name, "page_size": 10, "read_mask": "names,emailAddresses"},
            step="SEARCH_CONTACTS",
        )

        email, display_name = _first_email_from_contacts_payload(result)
        if not email:
            return {
                "found": False,
                "name": guest_name,
                "email": "",
                "source": "google_contacts",
                "reason": "No matching Google Contact with an email address was found.",
            }
        return {
            "found": True,
            "name": display_name or guest_name,
            "email": email,
            "source": "google_contacts",
            "reason": "Resolved through Google Contacts.",
        }

    def check_availability(
        self,
        *,
        start_iso: str,
        duration_minutes: int,
        current_user_name: str,
        guest_name: str = "",
        guest_email: str = "",
    ) -> dict[str, Any]:
        start = dt.datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        end = start + dt.timedelta(minutes=duration_minutes)
        window_start = start - dt.timedelta(hours=2)
        window_end = start + dt.timedelta(hours=3)
        calendar_ids = ["primary"]
        calendar_labels = {"primary": "your calendar"}
        if guest_email:
            calendar_ids.append(guest_email)
            calendar_labels[guest_email] = f"{guest_name}'s calendar" if guest_name else "the guest calendar"

        result = self._execute_tool(
            self.toolkit,
            "GOOGLECALENDAR_FIND_FREE_SLOTS",
            {
                "items": calendar_ids,
                "time_min": window_start.isoformat(),
                "time_max": window_end.isoformat(),
                "timezone": os.getenv("OPHELIA_TIMEZONE", "America/New_York"),
            },
            step="CHECK_CALENDAR",
        )
        calendars = {}
        for item in _iter_values(result):
            if isinstance(item.get("calendars"), dict):
                calendars = item["calendars"]
                break

        unavailable = []
        proposed = ""
        checked = []
        for calendar_id in calendar_ids:
            data = calendars.get(calendar_id) or calendars.get("primary" if calendar_id == "primary" else calendar_id) or {}
            label = calendar_labels.get(calendar_id, calendar_id)
            checked.append(label)
            free_slots = data.get("free") if isinstance(data, dict) else []
            is_free = any(_contains_interval(slot, start, end) for slot in free_slots if isinstance(slot, dict))
            if not is_free:
                unavailable.append(label)
            if not proposed and isinstance(free_slots, list):
                for slot in free_slots:
                    if not isinstance(slot, dict):
                        continue
                    slot_start = dt.datetime.fromisoformat(str(slot.get("start", "")).replace("Z", "+00:00"))
                    slot_end = dt.datetime.fromisoformat(str(slot.get("end", "")).replace("Z", "+00:00"))
                    if slot_end - slot_start >= dt.timedelta(minutes=duration_minutes):
                        proposed = slot_start.isoformat()
                        break

        if unavailable and not proposed:
            proposed = (start + dt.timedelta(minutes=45)).isoformat()

        return {
            "available": not unavailable,
            "reason": "Requested time is free." if not unavailable else f"Requested time conflicts or is not visible for {', '.join(unavailable)}.",
            "checked_calendars": checked,
            "proposed_datetime": "" if not unavailable else proposed,
            "raw": result,
        }

    def send_guest_invite_email(self, *, event: dict[str, Any]) -> dict[str, Any]:
        attendees = []
        for attendee in event.get("attendees", []):
            if isinstance(attendee, str) and attendee.strip():
                attendees.append(attendee.strip())
            elif isinstance(attendee, dict) and attendee.get("email"):
                attendees.append(str(attendee["email"]).strip())
        recipient = attendees[0] if attendees else str(event.get("guest_email") or "").strip()
        if not recipient:
            raise ComposioCalendarError("No guest email is available for the guest invitation email.")

        organizer_name = str(event.get("organizer_name") or "Aldric")
        guest_name = str(event.get("guest_name") or "there")
        internal_title = str(event.get("title") or "Dinner invitation")
        guest_calendar_title = str(event.get("guest_calendar_title") or f"Dinner with {organizer_name}")
        venue_name = str(event.get("venue_name") or event.get("location") or "the restaurant")
        location = str(event.get("location") or "")
        when = _display_datetime(str(event.get("start") or ""))
        party_size = event.get("party_size") or ""
        confirmation_code = str(event.get("confirmation_code") or "").strip()
        booking_id = str(event.get("booking_id") or "").strip()
        provider = str(event.get("provider") or "").strip()
        calendar_event = {**event, "title": guest_calendar_title}
        add_to_calendar_url = _google_calendar_link(calendar_event)

        subject = f"Dinner invite from {organizer_name}: {venue_name}"
        rows = [
            ("When", when),
            ("Where", location),
            ("Party", f"{party_size} guests" if party_size else ""),
            ("Confirmation", confirmation_code),
            ("Booking ID", booking_id),
            ("Provider", provider),
        ]
        detail_rows = "".join(
            f"<tr><td style='padding:6px 16px 6px 0;color:#666'>{html.escape(label)}</td>"
            f"<td style='padding:6px 0;color:#111'><strong>{html.escape(str(value))}</strong></td></tr>"
            for label, value in rows
            if value
        )
        calendar_button = ""
        if add_to_calendar_url:
            calendar_button = (
                "<p style='margin:24px 0'>"
                f"<a href='{html.escape(add_to_calendar_url)}' "
                "style='background:#111;color:#fff;text-decoration:none;padding:10px 14px;border-radius:6px;display:inline-block'>"
                "Add to Google Calendar</a></p>"
            )
        body = f"""
<div style="font-family:Arial,sans-serif;line-height:1.45;color:#111;max-width:560px">
  <p>Hi {html.escape(guest_name)},</p>
  <p>{html.escape(organizer_name)} booked dinner and invited you.</p>
  <h2 style="margin:18px 0 10px;font-size:20px">{html.escape(venue_name)}</h2>
  <table style="border-collapse:collapse;margin:8px 0 18px">
    {detail_rows}
  </table>
  {calendar_button}
  <p style="color:#555;font-size:13px;margin-top:24px">Sent by Ophelia after the booking was confirmed.</p>
</div>
""".strip()

        result = self._execute_tool(
            self.gmail_toolkit,
            "GMAIL_SEND_EMAIL",
            {
                "recipient_email": recipient,
                "subject": subject,
                "body": body,
                "is_html": True,
            },
            step="SEND_GUEST_INVITE_EMAIL",
        )
        return {
            "sent": True,
            "recipient_email": recipient,
            "subject": subject,
            "add_to_calendar_url": add_to_calendar_url,
            "message": "Guest invitation email sent through Gmail.",
            "raw": result,
        }

    def create_event(self, *, event: dict[str, Any]) -> dict[str, Any]:
        start = str(event.get("start") or "")
        end = str(event.get("end") or "")
        start_dt = dt.datetime.fromisoformat(start.replace("Z", "+00:00")) if start else None
        end_dt = dt.datetime.fromisoformat(end.replace("Z", "+00:00")) if end else None
        duration_minutes = int((end_dt - start_dt).total_seconds() // 60) if start_dt and end_dt else 90
        attendees = []
        for attendee in event.get("attendees", []):
            if isinstance(attendee, str) and attendee.strip():
                attendees.append(attendee.strip())
            elif isinstance(attendee, dict) and attendee.get("email"):
                attendees.append(str(attendee["email"]).strip())
        arguments = {
            "calendar_id": "primary",
            "summary": event.get("title") or "Ophelia booking",
            "start_datetime": start,
            "timezone": os.getenv("OPHELIA_TIMEZONE", "America/New_York"),
            "event_duration_hour": duration_minutes // 60,
            "event_duration_minutes": duration_minutes % 60,
            "location": event.get("location") or "",
            "description": event.get("description") or "",
            "attendees": attendees,
            "send_updates": "all",
            "create_meeting_room": False,
        }
        result = self._execute_tool(
            self.toolkit,
            "GOOGLECALENDAR_CREATE_EVENT",
            arguments,
            step="CREATE_CALENDAR_EVENT",
        )
        event_id = ""
        html_link = ""
        for item in _iter_values(result):
            event_id = event_id or str(item.get("id") or item.get("event_id") or "")
            html_link = html_link or str(item.get("htmlLink") or item.get("html_link") or "")
        return {
            "created": True,
            "event_id": event_id,
            "html_link": html_link,
            "attendee_emails": attendees,
            "message": "Google Calendar event created." if not attendees else "Google Calendar event created and attendee invitations were requested.",
            "raw": result,
        }
