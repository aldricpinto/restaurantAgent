import json
import os
from typing import Any

from sqlalchemy import create_engine, or_, select
from sqlalchemy.orm import Session, sessionmaker

from models import Base, User, UserMemoryEvent, UserProfile


DEFAULT_PROFILE: dict[str, Any] = {
    "home_location": "New York, NY",
    "relationship_status": "",
    "kids": {"has_kids": False, "count": 0},
    "pets": {"has_pets": False, "types": []},
    "preferences": {
        "dining": {
            "cuisines": [],
            "neighborhoods": [],
        },
        "fitness": {
            "activities": [],
            "time_preferences": [],
        },
        # "travel": {
        #     "hotel_style": [],
        #     "seat_preference": "",
        # },
    },
}


class MemoryStore:


    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url, future=True)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False, future=True)
        Base.metadata.create_all(self.engine)

    @classmethod
    def from_env(cls) -> "MemoryStore":
        database_url = os.getenv("OPHELIA_MEMORY_DATABASE_URL")
        if not database_url:
            db_path = os.getenv("OPHELIA_MEMORY_DB", "ophelia_agent_memory.db")
            database_url = f"sqlite:///{db_path}"
        return cls(database_url)

    def ensure_user(
        self,
        *,
        org_id: str,
        user_id: str,
        display_name: str | None = None,
        email: str | None = None,
        phone_number: str | None = None,
    ) -> None:
        with self.SessionLocal() as session:
            user = self._get_user(session, org_id=org_id, user_id=user_id)
            if user is None:
                user = User(
                    org_id=org_id,
                    user_id=user_id,
                    display_name=display_name,
                    email=email,
                    phone_number=phone_number,
                )
                session.add(user)
            else:
                if display_name:
                    user.display_name = display_name
                if email:
                    user.email = email
                if phone_number:
                    user.phone_number = phone_number

            profile = self._get_profile_row(session, org_id=org_id, user_id=user_id)
            if profile is None:
                session.add(
                    UserProfile(
                        org_id=org_id,
                        user_id=user_id,
                        profile_json=json.dumps(DEFAULT_PROFILE),
                    )
                )
            session.commit()

    def get_profile(self, *, org_id: str, user_id: str) -> dict[str, Any]:
        with self.SessionLocal() as session:
            profile = self._get_profile_row(session, org_id=org_id, user_id=user_id)
            if profile is None:
                return {}
            try:
                parsed = json.loads(profile.profile_json)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}

    def upsert_profile(self, *, org_id: str, user_id: str, profile: dict[str, Any]) -> None:
        with self.SessionLocal() as session:
            row = self._get_profile_row(session, org_id=org_id, user_id=user_id)
            if row is None:
                row = UserProfile(org_id=org_id, user_id=user_id, profile_json=json.dumps(profile))
                session.add(row)
            else:
                row.profile_json = json.dumps(profile)
            session.commit()

    def merge_profile_update(self, *, org_id: str, user_id: str, update: dict[str, Any]) -> dict[str, Any]:
        profile = self.get_profile(org_id=org_id, user_id=user_id) or dict(DEFAULT_PROFILE)
        merged = merge_profile_dict(profile, update)
        self.upsert_profile(org_id=org_id, user_id=user_id, profile=merged)
        return merged

    def recent_memory_events(
        self,
        *,
        org_id: str,
        user_id: str,
        vertical: str | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        with self.SessionLocal() as session:
            stmt = (
                select(UserMemoryEvent)
                .where(UserMemoryEvent.org_id == org_id, UserMemoryEvent.user_id == user_id)
                .order_by(UserMemoryEvent.created_at.desc(), UserMemoryEvent.id.desc())
                .limit(limit)
            )
            if vertical:
                stmt = stmt.where(or_(UserMemoryEvent.vertical == vertical, UserMemoryEvent.vertical.is_(None)))

            rows = list(session.scalars(stmt))

        return [memory_event_to_dict(row) for row in rows]

    def search_memory_events(
        self,
        *,
        org_id: str,
        user_id: str,
        vertical: str | None = None,
        memory_type: str | None = None,
        query: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        with self.SessionLocal() as session:
            stmt = (
                select(UserMemoryEvent)
                .where(UserMemoryEvent.org_id == org_id, UserMemoryEvent.user_id == user_id)
                .order_by(UserMemoryEvent.created_at.desc(), UserMemoryEvent.id.desc())
                .limit(limit)
            )
            if vertical:
                stmt = stmt.where(or_(UserMemoryEvent.vertical == vertical, UserMemoryEvent.vertical.is_(None)))
            if memory_type:
                stmt = stmt.where(UserMemoryEvent.memory_type == memory_type)
            if query:
                pattern = f"%{query.strip()}%"
                stmt = stmt.where(
                    or_(
                        UserMemoryEvent.content.ilike(pattern),
                        UserMemoryEvent.metadata_json.ilike(pattern),
                    )
                )

            rows = list(session.scalars(stmt))

        return [memory_event_to_dict(row) for row in rows]

    def add_memory_event(
        self,
        *,
        org_id: str,
        user_id: str,
        vertical: str | None,
        source: str,
        memory_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        confidence: float = 1.0,
        expires_at: Any | None = None,
    ) -> None:
        with self.SessionLocal() as session:
            session.add(
                UserMemoryEvent(
                    org_id=org_id,
                    user_id=user_id,
                    vertical=vertical,
                    source=source,
                    memory_type=memory_type,
                    content=content,
                    metadata_json=json.dumps(metadata or {}),
                    confidence=confidence,
                    expires_at=expires_at,
                )
            )
            session.commit()

    def _get_user(self, session: Session, *, org_id: str, user_id: str) -> User | None:
        return session.scalar(select(User).where(User.org_id == org_id, User.user_id == user_id))

    def _get_profile_row(self, session: Session, *, org_id: str, user_id: str) -> UserProfile | None:
        return session.scalar(
            select(UserProfile).where(UserProfile.org_id == org_id, UserProfile.user_id == user_id)
        )


def memory_event_to_dict(row: UserMemoryEvent) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if row.metadata_json:
        try:
            parsed = json.loads(row.metadata_json)
            metadata = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            metadata = {}
    return {
        "vertical": row.vertical,
        "source": row.source,
        "memory_type": row.memory_type,
        "content": row.content,
        "metadata": metadata,
        "confidence": row.confidence,
        "created_at": row.created_at.isoformat() if row.created_at else "",
    }


def build_memory_context(profile: dict[str, Any], events: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    if profile:
        home = profile.get("home_location")
        relationship = profile.get("relationship_status")
        kids = profile.get("kids") if isinstance(profile.get("kids"), dict) else {}
        pets = profile.get("pets") if isinstance(profile.get("pets"), dict) else {}
        preferences = profile.get("preferences") if isinstance(profile.get("preferences"), dict) else {}

        if home:
            lines.append(f"Home/default location: {home}")
        if relationship:
            lines.append(f"Relationship/family context: {relationship}")
        if kids.get("has_kids"):
            lines.append(f"Has kids: yes; count={kids.get('count', 'unknown')}")
        if pets.get("has_pets"):
            lines.append(f"Has pets: yes; types={pets.get('types', [])}")
        if preferences:
            lines.append(f"Known non-secret preferences by vertical: {json.dumps(preferences)}")

    if events:
        lines.append("Recent memory/events:")
        for event in events[:8]:
            lines.append(f"- [{event.get('vertical') or 'general'}] {event.get('content')}")

    return "\n".join(lines).strip() or "No user memory available yet."


def merge_profile_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(base or DEFAULT_PROFILE))
    for key, value in (update or {}).items():
        if value in (None, "", [], {}):
            continue
        if key == "preferences" and isinstance(value, dict):
            preferences = merged.setdefault("preferences", {})
            for vertical, vertical_update in value.items():
                if not isinstance(vertical_update, dict):
                    continue
                current_vertical = preferences.setdefault(vertical, {})
                for pref_key, pref_value in vertical_update.items():
                    if isinstance(pref_value, list):
                        current_values = current_vertical.setdefault(pref_key, [])
                        if not isinstance(current_values, list):
                            current_values = []
                        for item in pref_value:
                            if item and item not in current_values:
                                current_values.append(item)
                        current_vertical[pref_key] = current_values
                    elif pref_value not in (None, ""):
                        current_vertical[pref_key] = pref_value
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_profile_dict(merged[key], value)
            continue
        merged[key] = value
    return merged
