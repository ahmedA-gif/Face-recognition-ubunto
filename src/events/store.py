from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


class EventsStore:
    """SQLite event log with indexed, row-factory-backed fetch helpers."""

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                person TEXT NOT NULL,
                direction TEXT NOT NULL,
                track_id INTEGER,
                camera_id TEXT,
                confidence REAL,
                snapshot_path TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_id_desc ON events(id DESC)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_date_person ON events(date, person)"
        )
        self._conn.commit()

    def insert(self, event: Event) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO events(date, time, person, direction, track_id, camera_id, confidence, snapshot_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def delete(self, event_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def update_direction(self, event_id: int, direction: str) -> bool:
        cur = self._conn.execute(
            "UPDATE events SET direction = ? WHERE id = ?",
            (direction, event_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get(self, event_id: int) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            f"SELECT {', '.join(_EVENT_COLUMNS)} FROM events WHERE id = ?",
            (event_id,),
        ).fetchone()
        return _row_to_dict(row) if row is not None else None

    def recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, int(limit))
        rows = self._conn.execute(
            f"""
            SELECT {', '.join(_EVENT_COLUMNS)}
            FROM events ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def by_person(
        self,
        person: str,
        *,
        date: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        limit = max(1, int(limit))
        if date:
            rows = self._conn.execute(
                f"""
                SELECT {', '.join(_EVENT_COLUMNS)}
                FROM events WHERE person = ? AND date = ?
                ORDER BY id DESC LIMIT ?
                """,
                (person, date, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"""
                SELECT {', '.join(_EVENT_COLUMNS)}
                FROM events WHERE person = ?
                ORDER BY id DESC LIMIT ?
                """,
                (person, limit),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"])

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> "EventsStore":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @staticmethod
    def now_parts(dt: Optional[datetime] = None) -> tuple[str, str]:
        dt = dt or datetime.now()
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
