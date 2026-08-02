from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {k: row[k] for k in row.keys()}


class AttendanceDB:
    """SQLite abstraction for attendance records and boundary calibrations."""

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                person_id TEXT NOT NULL,
                person_name TEXT,
                check_in_time TEXT,
                check_out_time TEXT,
                status TEXT,
                work_hours REAL,
                confidence REAL,
                camera_id TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, person_id)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS boundary_calibration_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                line_coords TEXT,
                cluster_confidence REAL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance_logs(date)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_attendance_person ON attendance_logs(person_id, date)"
        )
        self._conn.commit()

    def upsert_log(
        self,
        date: str,
        person_id: str,
        person_name: str,
        check_in: Optional[str],
        check_out: Optional[str],
        status: str,
        work_hours: Optional[float],
        confidence: float,
        camera_id: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO attendance_logs
                (date, person_id, person_name, check_in_time, check_out_time,
                 status, work_hours, confidence, camera_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(date, person_id) DO UPDATE SET
                person_name     = excluded.person_name,
                check_in_time   = COALESCE(excluded.check_in_time, attendance_logs.check_in_time),
                check_out_time  = COALESCE(excluded.check_out_time, attendance_logs.check_out_time),
                status          = excluded.status,
                work_hours      = excluded.work_hours,
                confidence      = excluded.confidence,
                camera_id       = excluded.camera_id,
                updated_at      = CURRENT_TIMESTAMP
            """,
            (date, person_id, person_name, check_in, check_out, status, work_hours, confidence, camera_id),
        )
        self._conn.commit()

    def get_log(self, date: str, person_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            """
            SELECT date, person_id, person_name, check_in_time, check_out_time,
                   status, work_hours, confidence, camera_id
            FROM attendance_logs WHERE date = ? AND person_id = ?
            """,
            (date, person_id),
        ).fetchone()
        return _row_to_dict(row) if row is not None else None

    def daily_logs(self, date: Optional[str] = None) -> List[Dict[str, Any]]:
        date = date or datetime.now().strftime("%Y-%m-%d")
        rows = self._conn.execute(
            """
            SELECT date, person_id, person_name, check_in_time, check_out_time,
                   status, work_hours, confidence, camera_id
            FROM attendance_logs WHERE date = ?
            ORDER BY check_in_time IS NULL, check_in_time
            """,
            (date,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def insert_calibration(self, line_coords: Dict[str, float], confidence: float) -> int:
        cur = self._conn.execute(
            "INSERT INTO boundary_calibration_history(line_coords, cluster_confidence) VALUES (?, ?)",
            (json.dumps(line_coords), float(confidence)),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def calibrations(self, limit: int = 20) -> List[Dict[str, Any]]:
        limit = max(1, int(limit))
        rows = self._conn.execute(
            """
            SELECT id, timestamp, line_coords, cluster_confidence
            FROM boundary_calibration_history
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            coords = r["line_coords"]
            if isinstance(coords, str):
                try:
                    coords = json.loads(coords)
                except json.JSONDecodeError:
                    pass
            out.append(
                {
                    "id": r["id"],
                    "timestamp": r["timestamp"],
                    "line_coords": coords,
                    "cluster_confidence": r["cluster_confidence"],
                }
            )
        return out

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> "AttendanceDB":
        return self

    def __exit__(self, *_) -> None:
        self.close()
