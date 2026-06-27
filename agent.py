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

from memory_store import MemoryStore, build_memory_context
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
VERTICAL_POLICIES = {
    "dining": {
        "label": "restaurant",
        "result_label": "Restaurant Search Result",
        "required_fields": ("term", "location", "booking_datetime", "party_size"),
        "default_party_size": None,
        "include_party_size_in_search": True,
        "include_party_size_in_booking": True,
        "needs_password": False,
        "needs_full_billing_address": False,
    },
    "fitness": {
        "label": "fitness class",
        "result_label": "Fitness Search Result",
        "required_fields": ("term", "location", "booking_datetime"),
        "default_party_size": 1,
        "include_party_size_in_search": False,
        "include_party_size_in_booking": False,
        "needs_password": True,
        "needs_full_billing_address": True,
    },
}


class BookingState(TypedDict, total=False):
    org_id: str
    user_id: str
    user_profile: dict[str, Any]
    memory_events: list[dict[str, Any]]
    memory_context: str
    memory_recorded: bool
    profile_update_recorded: bool
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
    So this function will basically help me to append it.
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
    desired locations. Returns (search_location, desired_location_key), which is stored as constants for now.
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
        explicit_location = extract_explicit_city_state(location_phrase)
        return (explicit_location, "") if explicit_location else ("", "")

    _, selected = max(alias_candidates, key=lambda item: item[0])
    return selected["search_location"], selected["key"]


def title_city_state(location: str) -> str:
    parts = [part.strip() for part in location.split(",", 1)]
    if len(parts) != 2:
        return location.strip()
    city, state = parts
    return f"{city.title()}, {state.upper()}"


def extract_explicit_city_state(text: str) -> str:
    """
    Extract a scalable custom city/state location, e.g. "New Haven, CT".

    The predefined DESIRED_LOCATIONS list is useful for the MVP demo, but partners
    can also pass a clear city/state that is not in that list. In that case, we
    should use the explicit user-provided location instead of forcing clarification.
    """
    source = (text or "").strip()
    match = re.search(r"\b([A-Za-z][A-Za-z .'-]+,\s*[A-Za-z]{2})\b", source)
    if not match:
        return ""
    return title_city_state(match.group(1))


def split_named_place_and_location(
    *,
    term: str,
    raw_location: str,
    user_request: str,
) -> tuple[str, str]:
    """
    Handles cases like:

      "book a table at South Bay in New Haven, CT"

    The LLM sometimes puts "South Bay in New Haven, CT" into location and leaves
    term empty. Ophelia wants term="South Bay" and location="New Haven, CT".
    """
    sources = [user_request or "", raw_location or ""]
    patterns = [
        r"\bat\s+(?P<place>.+?)\s+(?:in|near)\s+(?P<location>[A-Za-z][A-Za-z .'-]+,\s*[A-Za-z]{2})\b",
        r"^(?P<place>.+?)\s+(?:in|near)\s+(?P<location>[A-Za-z][A-Za-z .'-]+,\s*[A-Za-z]{2})\b",
    ]
    for source in sources:
        normalized = re.sub(r"\s+", " ", source).strip()
        for pattern in patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if not match:
                continue
            inferred_term = (term or match.group("place")).strip(" .,:;")
            inferred_location = title_city_state(match.group("location"))
            return inferred_term.title(), inferred_location

    explicit_location = extract_explicit_city_state(raw_location) or extract_explicit_city_state(user_request)
    return term, explicit_location


def location_from_answer(answer: Any, fallback_location: str = "", fallback_key: str = "") -> tuple[str, str]:
    '''
    now this function helps me to extract location when the user hasn't mentioned it clearly in their request
    and the clarify node fires to ask them extra details and location is one of them.
    '''
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


def vertical_policy(vertical: str | None) -> dict[str, Any]:
    return VERTICAL_POLICIES.get((vertical or "dining").lower(), VERTICAL_POLICIES["dining"])


'''
now as there are minor changes when it comes to building payloads for verticals,
I have used VERTICAL_POLICIES, which is a dictionary that stores some context on how to build 
request payloads for the respective verticals
'''

def required_missing(state: BookingState) -> list[str]:
    missing: list[str] = []
    policy = vertical_policy(state.get("vertical"))
    for field in policy["required_fields"]:
        if not state.get(field):
            missing.append(field)
    if state.get("booking_datetime") and "datetime" not in missing and not has_explicit_time(state):
        missing.append("datetime")
    if (
        state.get("vertical") == "dining"
        and state.get("term")
        and is_vague_dining_term(str(state.get("term")))
        and dining_preference_cuisines(state)
        and "term" not in missing
    ):
        missing.append("term")
    return missing


def build_search_payload(state: BookingState) -> dict[str, Any]:
    vertical = (state.get("vertical") or "dining").lower()
    policy = vertical_policy(vertical)
    party_size = state.get("party_size") or policy.get("default_party_size")
    payload: dict[str, Any] = {
        "vertical": vertical,
        "term": state["term"],
        "location": state["location"],
        "datetime": state["booking_datetime"],
    }
    if party_size and policy.get("include_party_size_in_search"):
        payload["party_size"] = int(party_size)
    return payload


def has_explicit_time(state: BookingState) -> bool:
    source = f"{state.get('datetime_phrase') or ''} {state.get('user_request') or ''}".lower()
    time_patterns = [
        r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b",
        r"\b\d{1,2}:\d{2}\b",
        r"\b(noon|midnight)\b",
    ]
    return any(re.search(pattern, source) for pattern in time_patterns)


def is_vague_dining_term(term: str) -> bool:
    normalized = re.sub(r"[^a-z\s]", "", (term or "").lower()).strip()
    vague_terms = {
        "birthday dinner",
        "dinner",
        "birthday",
        "restaurant",
        "place",
        "meal",
        "food",
        "date night",
        "celebration",
    }
    return normalized in vague_terms or normalized.endswith(" dinner") and normalized.split()[0] in {"birthday", "anniversary", "celebration"}


def profile_party_size(profile: dict[str, Any]) -> int | None:
    if not isinstance(profile, dict):
        return None
    total = 1
    relationship = str(profile.get("relationship_status") or "").lower()
    if relationship in {"married", "partnered"}:
        total += 1
    kids = profile.get("kids") if isinstance(profile.get("kids"), dict) else {}
    if kids.get("has_kids"):
        try:
            total += int(kids.get("count") or 0)
        except (TypeError, ValueError):
            pass
    return total if total > 1 else None


def dining_preference_cuisines(state: BookingState) -> list[str]:
    cuisines: list[str] = []
    preferences = state.get("preferences") if isinstance(state.get("preferences"), dict) else {}
    for source in (
        preferences.get("cuisines"),
        preferences.get("dining", {}).get("cuisines") if isinstance(preferences.get("dining"), dict) else None,
        state.get("user_profile", {}).get("preferences", {}).get("dining", {}).get("cuisines")
        if isinstance(state.get("user_profile"), dict)
        else None,
    ):
        if isinstance(source, list):
            for cuisine in source:
                cuisine_text = str(cuisine).strip()
                if cuisine_text and cuisine_text not in cuisines:
                    cuisines.append(cuisine_text)
    return cuisines


def dining_preference_neighborhoods(state: BookingState) -> list[str]:
    neighborhoods: list[str] = []
    preferences = state.get("preferences") if isinstance(state.get("preferences"), dict) else {}
    for source in (
        preferences.get("neighborhoods"),
        preferences.get("dining", {}).get("neighborhoods") if isinstance(preferences.get("dining"), dict) else None,
        state.get("user_profile", {}).get("preferences", {}).get("dining", {}).get("neighborhoods")
        if isinstance(state.get("user_profile"), dict)
        else None,
    ):
        if isinstance(source, list):
            for neighborhood in source:
                neighborhood_text = str(neighborhood).strip()
                if neighborhood_text and neighborhood_text not in neighborhoods:
                    neighborhoods.append(neighborhood_text)
    return neighborhoods


def term_from_answer(answer: Any, state: BookingState) -> str:
    raw = str(answer or "").strip()
    if raw.isdigit():
        cuisines = dining_preference_cuisines(state)
        idx = int(raw) - 1
        if 0 <= idx < len(cuisines):
            return cuisines[idx]
    return raw


def payment_payload_from_details(details: dict[str, Any], include_full_billing_address: bool = False) -> dict[str, Any]:
    payment = {
        "card_number": details.get("card_number"),
        "exp_month": details.get("card_exp_month"),
        "exp_year": details.get("card_exp_year"),
        "cvv": details.get("card_cvv"),
        "name_on_card": details.get("card_name"),
        "postal_code": details.get("card_postal"),
    }
    if include_full_billing_address:
        payment["address_line1"] = details.get("billing_address_line1")
        payment["city"] = details.get("billing_city")
        payment["state"] = details.get("billing_state")
        payment["country"] = details.get("billing_country") or "US"
    return payment


def payment_fields_for_policy(policy: dict[str, Any]) -> list[str]:
    fields = ["card_number", "card_exp_month", "card_exp_year", "card_cvv", "card_name", "card_postal"]
    if policy.get("needs_full_billing_address"):
        fields.extend(["billing_address_line1", "billing_city", "billing_state", "billing_country"])
    return fields


'''
now this is to help with context for queries like: "Book me birthday dinner".

This will help store booking details (this is actually helpful for the hallucination usecase)
'''

def booking_memory_content(state: BookingState) -> str:
    vertical = state.get("vertical") or "dining"
    venue = state.get("selected_venue", {})
    venue_name = venue.get("name") or "selected result"
    term = state.get("term") or vertical
    location = state.get("location") or "unknown location"
    status = state.get("booking_status") or "unknown"
    if vertical == "fitness":
        class_time = venue.get("metadata", {}).get("class_time") if isinstance(venue.get("metadata"), dict) else ""
        time_note = f" at {class_time}" if class_time else ""
        return f"User attempted {term} fitness booking at {venue_name}{time_note} near {location}; final status {status}."
    return f"User attempted {term} dining booking at {venue_name} near {location}; final status {status}."


def extract_profile_update_from_state(state: BookingState) -> dict[str, Any]:
    user_request = (state.get("user_request") or "").lower()
    update: dict[str, Any] = {}

    if re.search(r"\b(my )?(wife|husband|spouse|partner)\b", user_request):
        update["relationship_status"] = "married" if re.search(r"\b(wife|husband|spouse)\b", user_request) else "partnered"

    kids_count = extract_kids_count(user_request)
    if kids_count is not None:
        update["kids"] = {"has_kids": kids_count > 0, "count": kids_count}

    pet_types = extract_pet_types(user_request)
    if pet_types:
        update["pets"] = {"has_pets": True, "types": pet_types}

    vertical = state.get("vertical")
    preferences: dict[str, Any] = {}
    term = state.get("term")
    if vertical == "dining":
        cuisines = dining_preference_cuisines(state)
        if term and not is_vague_dining_term(str(term)):
            cuisines.append(str(term))
        for cuisine in cuisines:
            cuisine_text = str(cuisine).strip()
            if cuisine_text:
                preferences.setdefault("dining", {}).setdefault("cuisines", []).append(cuisine_text)
    if vertical == "fitness" and term:
        preferences.setdefault("fitness", {}).setdefault("activities", []).append(term)

    location_label = location_label_from_key(state.get("desired_location_key"))
    if vertical == "dining" and location_label:
        preferences.setdefault("dining", {}).setdefault("neighborhoods", []).append(location_label)
    if vertical == "dining":
        for neighborhood in dining_preference_neighborhoods(state):
            preferences.setdefault("dining", {}).setdefault("neighborhoods", []).append(neighborhood)

    if preferences:
        update["preferences"] = preferences

    return update


def extract_kids_count(text: str) -> int | None:
    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
    }
    match = re.search(r"\b(\d+|one|two|three|four|five|six)\s+(kids?|children)\b", text)
    if match:
        raw = match.group(1)
        return int(raw) if raw.isdigit() else number_words.get(raw)
    if re.search(r"\b(my )?(kid|kids|child|children)\b", text):
        return 1
    return None


def extract_pet_types(text: str) -> list[str]:
    pet_types: list[str] = []
    for pet in ("dog", "cat", "bird", "rabbit"):
        if re.search(rf"\b{pet}s?\b", text):
            pet_types.append(pet)
    if not pet_types and re.search(r"\bpet\b", text):
        pet_types.append("pet")
    return pet_types


def location_label_from_key(location_key: str | None) -> str:
    if not location_key:
        return ""
    for location in DESIRED_LOCATIONS:
        if location["key"] == location_key:
            return str(location["label"])
    return ""


def summary_booking_details(state: BookingState) -> str:
    policy = vertical_policy(state.get("vertical"))
    selected = state.get("selected_venue", {}) or {}
    metadata = selected.get("metadata") if isinstance(selected.get("metadata"), dict) else {}
    provider = selected_provider(state)

    lines = ["Booking Details:"]
    lines.append(f"- Vertical: {state.get('vertical') or 'dining'}")
    lines.append(f"- {policy['result_label']}: {selected.get('name') or 'Unknown result'}")
    studio_name = metadata.get("studio_name")
    if studio_name:
        lines.append(f"- Studio/Venue: {studio_name}")
    address = ", ".join(
        str(selected.get(key, "")).strip()
        for key in ("address", "city", "state")
        if selected.get(key)
    )
    if address:
        lines.append(f"- Address: {address}")
    if provider:
        lines.append(f"- Provider: {provider}")
    if state.get("customer_name"):
        lines.append(f"- Guest Name: {state['customer_name']}")
    if state.get("booking_datetime"):
        lines.append(f"- Reserved Date & Time: {state['booking_datetime']}")
    if state.get("party_size") and policy.get("include_party_size_in_booking"):
        lines.append(f"- Party Size: {state['party_size']}")

    for label, key in (
        ("Class Time", "class_time"),
        ("Class Date", "date"),
        ("Price", "price"),
        ("Category", "category"),
        ("Distance", "distance"),
    ):
        value = metadata.get(key)
        if value:
            lines.append(f"- {label}: {value}")

    if state.get("booking_id"):
        lines.append(f"- Booking ID: {state['booking_id']}")
    lines.append(f"- Final status: {state.get('booking_status', 'unknown')}")
    return "\n".join(lines) + "\n"



'''
this is for showing user a list of venues to choose from (for now I am only showing a max of 5, kinda helps 
with choice paralysis)
'''
def venue_display(venue: dict[str, Any]) -> str:
    address = ", ".join(
        str(venue.get(key, "")).strip()
        for key in ("address", "city", "state")
        if venue.get(key)
    )
    provider = venue.get("provider") or venue.get("source") or "unknown provider"
    metadata = venue.get("metadata") if isinstance(venue.get("metadata"), dict) else {}
    class_time = metadata.get("class_time")
    price = metadata.get("price") or venue.get("price")
    extra = ""
    if class_time:
        extra += f" — {class_time}"
    if price:
        extra += f" — {price}"
    return f"{venue.get('name', 'Unknown venue')} ({venue.get('id')}) — {address}{extra} [{provider}]"


def selected_provider(state: BookingState) -> str:
    return str(
        state.get("selected_venue", {}).get("provider")
        or state.get("selected_venue", {}).get("source")
        or ""
    ).strip().lower()


''' 
this is for processing purposes in the availablity node.
'''

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
    memory_store = MemoryStore.from_env()
    default_org_id = os.getenv("OPHELIA_ORG_ID", "demo_org")
    default_user_id = os.getenv("OPHELIA_USER_ID", "demo_user")
    model = ChatGroq(model_name=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))

    def load_user_context_node(state: BookingState) -> dict[str, Any]:
        org_id = state.get("org_id") or default_org_id
        user_id = state.get("user_id") or default_user_id
        vertical = state.get("vertical") or None
        logger.info("load_user_context_node: loading memory for org_id=%s user_id=%s", org_id, user_id)
        memory_store.ensure_user(org_id=org_id, user_id=user_id)
        profile = memory_store.get_profile(org_id=org_id, user_id=user_id)
        events = memory_store.recent_memory_events(org_id=org_id, user_id=user_id, vertical=vertical)
        return {
            "org_id": org_id,
            "user_id": user_id,
            "user_profile": profile,
            "memory_events": events,
            "memory_context": build_memory_context(profile, events),
        }

    async def extract_intent_node(state: BookingState) -> dict[str, Any]:
        logger.info("extract_intent_node: parsing user request")
        user_request = state.get("user_request", "")
        memory_context = state.get("memory_context", "No user memory available yet.")
        prompt = f"""
                You extract booking intent for the Ophelia API.

                Return ONLY valid JSON with these keys:
                - vertical: "dining" or "fitness"
                - term: restaurant/cuisine/activity/event search term
                - location: city/neighborhood/state/country text
                - datetime_phrase: exact natural language date/time phrase from the user, e.g. "today at 7pm"
                - party_size: integer number of people/tickets when present, otherwise null
                - preferences: object of soft preferences
                - missing_fields: array of missing required fields for the selected vertical

                Rules:
                - MVP supports dining and fitness.
                - Do not invent missing values.
                - For "Italian place in SoHo" term is "Italian" and location is "SoHo".
                - For "table at South Bay in New Haven, CT" term is "South Bay" and location is "New Haven, CT".
                - For "pilates class near SoHo" vertical is "fitness" and term is "pilates".
                - Keep datetime_phrase as natural language; do not normalize it.
                - Use user memory only to refine soft preferences. Do not invent booking-critical fields.

                Safe user memory:
                {memory_context}

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

        vertical = (parsed.get("vertical") or "dining").lower()

        # I'm doing this temporarily for the MVP, but I'll have proper vertical support in the future
        # if vertical not in VERTICAL_POLICIES:
        #     vertical = "dining"
        datetime_phrase = parsed.get("datetime_phrase") or ""
        booking_datetime = normalize_datetime(datetime_phrase, user_request)
        raw_location = parsed.get("location") or ""
        term = parsed.get("term") or ""
        term, inferred_custom_location = split_named_place_and_location(
            term=term,
            raw_location=raw_location,
            user_request=user_request,
        )
        location, desired_location_key = parse_desired_location(inferred_custom_location or raw_location, user_request)
        if inferred_custom_location and not desired_location_key:
            location = inferred_custom_location
        party_size = parsed.get("party_size")
        try:
            party_size = int(party_size) if party_size is not None else None
        except (TypeError, ValueError):
            party_size = None

        update: BookingState = {
            "vertical": vertical,
            "term": term,
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
            "memory_recorded": False,
            "profile_update_recorded": False,
        }
        if booking_datetime:
            update["booking_datetime"] = booking_datetime
        if party_size:
            update["party_size"] = party_size
        elif vertical_policy(vertical).get("default_party_size"):
            update["party_size"] = int(vertical_policy(vertical)["default_party_size"])

        # if i have any missing values this where I call required_missing to
        # figure out the missing values for that particular vertical and then send
        # control to clarify node to do the needful
        update["missing_fields"] = required_missing(update)
        logger.info("extract_intent_node: extracted intent=%s", redact(update))
        return update

    def update_user_profile_node(state: BookingState) -> dict[str, Any]:
        if state.get("profile_update_recorded"):
            return {}

        profile_update = extract_profile_update_from_state(state)
        if not profile_update:
            return {}

        org_id = state.get("org_id") or default_org_id
        user_id = state.get("user_id") or default_user_id
        logger.info("update_user_profile_node: merging profile update=%s", redact(profile_update))
        try:
            profile = memory_store.merge_profile_update(org_id=org_id, user_id=user_id, update=profile_update)
            memory_store.add_memory_event(
                org_id=org_id,
                user_id=user_id,
                vertical=state.get("vertical"),
                source="user_request",
                memory_type="profile_update",
                content="User shared non-sensitive profile/preferences during a booking request.",
                metadata=profile_update,
                confidence=0.85,
            )
            events = memory_store.recent_memory_events(org_id=org_id, user_id=user_id, vertical=state.get("vertical"))
            return {
                "user_profile": profile,
                "memory_events": events,
                "memory_context": build_memory_context(profile, events),
                "profile_update_recorded": True,
            }
        except Exception as exc:
            logger.warning("update_user_profile_node: failed to update profile: %s", exc)
            return {}

    def clarify_node(state: BookingState) -> dict[str, Any]:
        missing = required_missing(state)
        logger.info("clarify_node: missing fields=%s", missing)
        cuisine_choices = dining_preference_cuisines(state) if "term" in missing else []
        suggested_party_size = profile_party_size(state.get("user_profile", {})) if "party_size" in missing else None
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
                "term_choices": [
                    f"{idx}. {cuisine}" for idx, cuisine in enumerate(cuisine_choices, start=1)
                ],
                "field_suggestions": {
                    "term": "I remember you like: " + ", ".join(cuisine_choices) if cuisine_choices else "",
                    "party_size": f"Based on your profile, this may be for {suggested_party_size} people. Confirm or enter another number."
                    if suggested_party_size
                    else "",
                    "datetime": "Please include both date and time, e.g. 'tomorrow at 7 PM'.",
                },
            }
        )

        user_request = state.get("user_request", "")
        term = term_from_answer(answers.get("term"), state) or state.get("term", "")
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


    '''
    these are function for conditional edges of the graph
    '''
    def route_after_intent(state: BookingState) -> Literal["clarify", "search"]:
        return "clarify" if required_missing(state) else "search"

    def route_after_search(state: BookingState) -> Literal["select_venue", "summary"]:
        return "summary" if state.get("booking_status") in TERMINAL_STATUSES else "select_venue"

    def route_after_availability(state: BookingState) -> Literal["preflight_and_consent", "summary"]:
        return "summary" if state.get("booking_status") in TERMINAL_STATUSES else "preflight_and_consent"

    async def search_node(state: BookingState) -> dict[str, Any]:
        payload = build_search_payload(state)
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

        provider = selected_provider(state)
        if provider == "mindbody":
            selected_venue = state.get("selected_venue", {})
            metadata = selected_venue.get("metadata") if isinstance(selected_venue.get("metadata"), dict) else {}
            class_time = metadata.get("class_time")
            logger.info("availability_node: using Mindbody inline class metadata class_time=%s", class_time)
            return {
                "availability": {
                    "source": "venues/search.inline",
                    "provider": "mindbody",
                    "class_time": class_time,
                    "available": True,
                }
            }

        if provider == "opentable":
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
            "party_size": int(state.get("party_size") or vertical_policy(state.get("vertical")).get("default_party_size") or 1),
            "datetime": format_api_datetime(state["booking_datetime"]),
            "window_minutes": 60,
        }
        logger.info("availability_node: payload=%s", redact(payload))

        # I am have this for a graceful fallback if neither of the above works
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
        policy = vertical_policy(state.get("vertical"))
        consent = interrupt(
            {
                "message": "Please confirm before I create a real booking.",
                "booking_details": {
                    policy["label"]: venue_display(venue),
                    "datetime": state.get("booking_datetime"),
                    "party_size": state.get("party_size"),
                    "vertical": state.get("vertical", "dining"),
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
        vertical = (state.get("vertical") or "dining").lower()
        policy = vertical_policy(vertical)

        # this my solution for akamai blocks using Redis can_book and get_cool_off_period
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

            

        fields_needed = ["name", "email", "phone", *payment_fields_for_policy(policy)]
        if policy.get("needs_password"):
            fields_needed.append("password")

        details = interrupt(
            {
                "message": "Please provide the customer details for this booking request.",
                "fields_needed": fields_needed,
            }
        )

        idempotency_key = state.get("idempotency_key") or str(uuid.uuid4())
        payload: dict[str, Any] = {
            "vertical": vertical,
            "venue_id": venue_id,
            "datetime": state["booking_datetime"],
            "customer": {
                "name": details["name"],
                "email": details["email"],
                "phone_number": details["phone"],
            },
            "metadata": {},
        }
        if policy.get("include_party_size_in_booking"):
            payload["party_size"] = int(state.get("party_size") or policy.get("default_party_size") or 1)
        if details.get("card_number"):
            payload["metadata"]["payment"] = payment_payload_from_details(
                details,
                include_full_billing_address=bool(policy.get("needs_full_billing_address")),
            )
        if policy.get("needs_password"):
            payload["metadata"]["password"] = details.get("password")

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
            policy = vertical_policy(state.get("vertical"))
            fields_needed = payment_fields_for_policy(policy)
            prompt = {
                "message": f"Booking {booking_id} requires payment details.",
                "booking_id": booking_id,
                "next_action": next_action,
                "fields_needed": fields_needed,
            }
            payment = interrupt(prompt)
            payload = {"payment": payment_payload_from_details(payment, bool(policy.get("needs_full_billing_address")))}
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
        selected = state.get("selected_venue", {}) or {}
        venue_name = selected.get("name") or "Unknown result"
        selected_metadata = selected.get("metadata") if isinstance(selected.get("metadata"), dict) else {}
        booking_details = summary_booking_details(state)
        grounded = {
            "vertical": state.get("vertical") or "dining",
            "status": status,
            "booking_id": state.get("booking_id"),
            "selected_result": {
                "id": state.get("selected_venue_id"),
                "name": venue_name,
                "provider": selected_provider(state),
                "metadata": selected_metadata,
            },
            "datetime": state.get("booking_datetime"),
            "party_size": state.get("party_size") if vertical_policy(state.get("vertical")).get("include_party_size_in_booking") else None,
            "error": state.get("error"),
            "booking_response": redact(state.get("booking_response", {})),
        }
        prompt = f"""
                    Write a short user-facing booking result.
                    Use only the grounded JSON. Do not imply success unless status is confirmed.
                    If status is verification_required, explicitly say the booking could not be verified as confirmed yet.
                    If failed/error/rate_limited/expired, be honest and include the sanitized error code if present.
                    Use the selected_result and Booking Details keys that are present.
                    Do not mention missing fields, and do not invent a guest name, party size, class time, venue, or confirmation code.

                    {booking_details}

                    Grounded JSON:
                    {json.dumps(grounded, indent=2)}
                    """.strip()
        resp = await model.ainvoke([("user", prompt)])
        print("\n" + "=" * 60)
        print(resp.content)
        print("=" * 60 + "\n")
        update: BookingState = {"final_summary": resp.content, "booking_succeeded": succeeded}

        if state.get("booking_id") and not state.get("memory_recorded"):
            metadata = {
                "booking_id": state.get("booking_id"),
                "status": status,
                "provider": selected_provider(state),
                "term": state.get("term"),
                "venue_id": state.get("selected_venue_id"),
                "venue_name": venue_name,
                "location": state.get("location"),
                "datetime": state.get("booking_datetime"),
                "party_size": state.get("party_size"),
                "error_code": booking_error_code(state.get("booking_response", {})),
            }
            try:
                memory_store.add_memory_event(
                    org_id=state.get("org_id") or default_org_id,
                    user_id=state.get("user_id") or default_user_id,
                    vertical=state.get("vertical") or "dining",
                    source="booking_result",
                    memory_type="booking_history",
                    content=booking_memory_content(state),
                    metadata=metadata,
                )
                update["memory_recorded"] = True
                logger.info("summary_node: wrote booking memory event metadata=%s", redact(metadata))
            except Exception as exc:
                logger.warning("summary_node: failed to write memory event: %s", exc)

        return update

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
    workflow.add_node("load_user_context", load_user_context_node)
    workflow.add_node("extract_intent", extract_intent_node)
    workflow.add_node("update_user_profile", update_user_profile_node)
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

    workflow.add_edge(START, "load_user_context")
    workflow.add_edge("load_user_context", "extract_intent")
    workflow.add_edge("extract_intent", "update_user_profile")
    workflow.add_conditional_edges("update_user_profile", route_after_intent, {"clarify": "clarify", "search": "search"})
    workflow.add_edge("clarify", "update_user_profile")
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
        {"cancel": "cancel", "new_booking": "load_user_context", "end": END},
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
    field_suggestions = payload.get("field_suggestions") if isinstance(payload.get("field_suggestions"), dict) else {}

    field_labels = {
        "location": "location",
        "term": "cuisine/activity",
        "datetime": "date/time",
        "party_size": "party size",
        "selection": "venue selection",
        "approved": "approval",
        "otp_code": "OTP code",
    }
    if len(fields) > 1:
        needed = ", ".join(field_labels.get(field, field.replace("_", " ")) for field in fields)
        print(f"\nI'll ask for these one at a time: {needed}.")

    for index, field in enumerate(fields, start=1):
        label = field_labels.get(field, field.replace("_", " ").title())
        if len(fields) > 1:
            print(f"\n[{index}/{len(fields)}] {label.title()}")
        suggestion = field_suggestions.get(field)
        if suggestion:
            print(suggestion)
        if field == "approved":
            answers[field] = input("Approve booking? Type yes/no: ").strip()
        elif field == "selection":
            answers[field] = input("Selection number: ").strip()
        elif field == "location":
            if payload.get("location_choices"):
                print("Location choices:")
                for choice in payload["location_choices"]:
                    print(choice)
            answers[field] = input("Location number or custom location: ").strip()
        elif field == "term":
            if payload.get("term_choices"):
                print("Cuisine/activity choices:")
                for choice in payload["term_choices"]:
                    print(choice)
            answers[field] = input("Cuisine/activity number or custom search term: ").strip()
        elif field == "card_cvv":
            answers[field] = input(f"{label.title()}: ").strip()
        elif field == "otp_code":
            answers[field] = input("OTP code: ").strip()
        else:
            answers[field] = input(f"{label.title()}: ").strip()
    return answers


if __name__ == "__main__":
    asyncio.run(main())
