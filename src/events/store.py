from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.db.postgres import execute_query, execute_write, execute_insert


_EVENT_COLUMNS = (
    "id",
    "date",
    "time",
    "person",
    "direction",
    "track_id",
    "camera_id",
    "confidence",
    "snapshot_path",
    "event_id",
)


@dataclass
class Event:
    id: int = 0
    date: str = ""
    time: str = ""
    person: str = ""
    direction: str = ""  # entry | exit
    track_id: int = 0
    camera_id: str = "cam_01"
    confidence: float = 0.0
    snapshot_path: str = ""
    global_id: str = ""
    fsm_path: List[str] = field(default_factory=list)
    event_id: str = ""


class EventsStore:
    """PostgreSQL event log with indexed fetch helpers."""

    def __init__(self, db_path: str = "") -> None:
        pass

    def insert(self, event: Event) -> int:
        return execute_insert(
            """
            INSERT INTO events(date, time, person, direction, track_id, camera_id, confidence, snapshot_path, event_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                event.date,
                event.time,
                event.person,
                event.direction,
                event.track_id,
                event.camera_id,
                event.confidence,
                event.snapshot_path,
                event.event_id or None,
            ),
        )

    def delete(self, event_id: int) -> bool:
        return execute_write("DELETE FROM events WHERE id = %s", (event_id,)) > 0

    def update_direction(self, event_id: int, direction: str) -> bool:
        return execute_write("UPDATE events SET direction = %s WHERE id = %s", (direction, event_id)) > 0

    def update_person(self, event_id: int, person: str) -> bool:
        return execute_write("UPDATE events SET person = %s WHERE id = %s", (person, event_id)) > 0

    def get(self, event_id: int) -> Optional[Dict[str, Any]]:
        rows = execute_query(
            f"SELECT {', '.join(_EVENT_COLUMNS)} FROM events WHERE id = %s",
            (event_id,),
        )
        return rows[0] if rows else None

    def get_by_event_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        if not event_id:
            return None
        rows = execute_query(
            f"SELECT {', '.join(_EVENT_COLUMNS)} FROM events WHERE event_id = %s LIMIT 1",
            (event_id,),
        )
        return rows[0] if rows else None

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, int(limit))
        return execute_query(
            f"SELECT {', '.join(_EVENT_COLUMNS)} FROM events ORDER BY id DESC LIMIT %s",
            (limit,),
        )

    def by_person(
        self,
        person: str,
        *,
        date: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        limit = max(1, int(limit))
        if date:
            return execute_query(
                f"SELECT {', '.join(_EVENT_COLUMNS)} FROM events WHERE person = %s AND date = %s ORDER BY id DESC LIMIT %s",
                (person, date, limit),
            )
        return execute_query(
            f"SELECT {', '.join(_EVENT_COLUMNS)} FROM events WHERE person = %s ORDER BY id DESC LIMIT %s",
            (person, limit),
        )

    def count(self) -> int:
        rows = execute_query("SELECT COUNT(*) AS n FROM events")
        return int(rows[0]["n"]) if rows else 0

    def direction_counts(self, date: Optional[str] = None) -> Dict[str, int]:
        if date:
            rows = execute_query(
                "SELECT direction, COUNT(*) AS n FROM events WHERE date = %s GROUP BY direction",
                (date,),
            )
        else:
            rows = execute_query("SELECT direction, COUNT(*) AS n FROM events GROUP BY direction")
        return {r["direction"]: int(r["n"]) for r in rows}

    def close(self) -> None:
        pass

    def __enter__(self) -> "EventsStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @staticmethod
    def now_parts(dt: Optional[datetime] = None) -> tuple[str, str]:
        dt = dt or datetime.now()
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
