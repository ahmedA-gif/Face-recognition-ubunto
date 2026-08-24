from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.db.postgres import execute_query, execute_write, execute_insert


class AttendanceDB:
    """PostgreSQL abstraction for attendance records and boundary calibrations."""

    def __init__(self, db_path: str = "") -> None:
        pass

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
        clear_checkout: bool = False,
    ) -> None:
        if clear_checkout:
            checkout_sql = "NULL"
        else:
            checkout_sql = "COALESCE(excluded.check_out_time, attendance_logs.check_out_time)"

        execute_write(
            f"""
            INSERT INTO attendance_logs
                (date, person_id, person_name, check_in_time, check_out_time,
                 status, work_hours, confidence, camera_id, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT(date, person_id) DO UPDATE SET
                person_name     = excluded.person_name,
                check_in_time   = COALESCE(excluded.check_in_time, attendance_logs.check_in_time),
                check_out_time  = {checkout_sql},
                status          = excluded.status,
                work_hours      = excluded.work_hours,
                confidence      = excluded.confidence,
                camera_id       = excluded.camera_id,
                updated_at      = NOW()
            """,
            (date, person_id, person_name, check_in, check_out, status, work_hours, confidence, camera_id),
        )

    def get_log(self, date: str, person_id: str) -> Optional[Dict[str, Any]]:
        rows = execute_query(
            """
            SELECT date, person_id, person_name, check_in_time, check_out_time,
                   status, work_hours, confidence, camera_id
            FROM attendance_logs WHERE date = %s AND person_id = %s
            """,
            (date, person_id),
        )
        return rows[0] if rows else None

    def daily_logs(self, date: Optional[str] = None) -> List[Dict[str, Any]]:
        date = date or datetime.now().strftime("%Y-%m-%d")
        return execute_query(
            """
            SELECT date, person_id, person_name, check_in_time, check_out_time,
                   status, work_hours, confidence, camera_id
            FROM attendance_logs WHERE date = %s
            ORDER BY check_in_time IS NULL, check_in_time
            """,
            (date,),
        )

    def insert_calibration(self, line_coords: Dict[str, float], confidence: float) -> int:
        return execute_insert(
            "INSERT INTO boundary_calibration_history(line_coords, cluster_confidence) VALUES (%s, %s) RETURNING id",
            (json.dumps(line_coords), float(confidence)),
        )

    def calibrations(self, limit: int = 20) -> List[Dict[str, Any]]:
        limit = max(1, int(limit))
        rows = execute_query(
            """
            SELECT id, timestamp, line_coords, cluster_confidence
            FROM boundary_calibration_history
            ORDER BY id DESC LIMIT %s
            """,
            (limit,),
        )
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
        pass

    def __enter__(self) -> "AttendanceDB":
        return self

    def __exit__(self, *_) -> None:
        self.close()
