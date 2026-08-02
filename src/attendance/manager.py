from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.attendance.db import AttendanceDB
from src.events.store import Event


class AttendanceManager:
    """Daily attendance state machine, keyed strictly on ``global_person_id``.

    States per person per day: NOT_ARRIVED → CHECKED_IN ⇄ ON_BREAK → CHECKED_OUT.

    - First entry  → Check-In (with Late / On Time status).
    - Last exit    → Check-Out (with Early Departure / On Time status + hours).
    - Multi-track dedup: a person's events are ignored for ``debounce_minutes``
      after the last processed event (fragmented ByteTrack IDs collapse here).
    """

    def __init__(
        self,
        db: AttendanceDB,
        shift_start: str = "09:00",
        shift_end: str = "17:00",
        late_threshold_mins: int = 15,
        early_exit_mins: int = 15,
        debounce_minutes: float = 2.0,
        camera_id: str = "cam_01",
    ) -> None:
        self.db = db
        self.shift_start = shift_start
        self.shift_end = shift_end
        self.late_mins = int(late_threshold_mins)
        self.early_mins = int(early_exit_mins)
        self.debounce_sec = float(debounce_minutes) * 60.0
        self.camera_id = camera_id

        self._state: Dict[str, Dict[str, Dict[str, Any]]] = {}  # date -> person -> state

    # ── public API ─────────────────────────────────────────────────────────────

    def process_events(self, events: List[Event]) -> None:
        for e in events:
            self._process_one(e)

    def summary(self, date: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.db.daily_logs(date)

    # ── internals ──────────────────────────────────────────────────────────────

    def _state_for(self, e: Event) -> Dict[str, Any]:
        states = self._state.setdefault(e.date, {})
        return states.setdefault(
            e.person,
            {
                "state": "NOT_ARRIVED",
                "name": e.person,
                "check_in": None,
                "check_out": None,
                "last_event": 0.0,
                "conf": 0.0,
            },
        )

    def _process_one(self, e: Event) -> None:
        st = self._state_for(e)
        now = time.time()

        # Multi-track / multi-event dedup window per identity
        if st["last_event"] > 0 and now - st["last_event"] < self.debounce_sec:
            st["last_event"] = now
            return

        st["conf"] = max(st["conf"], float(e.confidence or 0.0))

        if e.direction == "entry":
            if st["state"] == "NOT_ARRIVED":
                st["state"] = "CHECKED_IN"
                st["check_in"] = e.time
                status = self._checkin_status(e.date, e.time)
                self.db.upsert_log(
                    date=e.date, person_id=e.person, person_name=e.person,
                    check_in=e.time, check_out=None, status=status,
                    work_hours=None, confidence=st["conf"], camera_id=self.camera_id,
                )
                print(f"[ATTENDANCE] {e.date} | {e.person} | CHECK-IN {e.time} ({status})")
            elif st["state"] == "ON_BREAK":
                st["state"] = "CHECKED_IN"
                print(f"[ATTENDANCE] {e.date} | {e.person} | BACK FROM BREAK {e.time}")

        elif e.direction == "exit":
            if st["state"] in ("CHECKED_IN", "ON_BREAK"):
                st["state"] = "CHECKED_OUT"
                st["check_out"] = e.time
                status, hours = self._checkout_status(e.date, st)
                self.db.upsert_log(
                    date=e.date, person_id=e.person, person_name=e.person,
                    check_in=st["check_in"], check_out=e.time, status=status,
                    work_hours=hours, confidence=st["conf"], camera_id=self.camera_id,
                )
                hours_s = f"{hours:.2f}h" if hours is not None else "?"
                print(f"[ATTENDANCE] {e.date} | {e.person} | CHECK-OUT {e.time} ({status}, {hours_s})")
            elif st["state"] == "NOT_ARRIVED":
                print(f"[ATTENDANCE] {e.date} | {e.person} | exit without check-in (ignored)")

        st["last_event"] = now

    # ── status logic ───────────────────────────────────────────────────────────

    def _checkin_status(self, date: str, time_str: str) -> str:
        shift = self._to_minutes(self.shift_start)
        arrive = self._to_minutes(time_str)
        if shift is None or arrive is None:
            return "On Time"
        return "Late" if arrive > shift + self.late_mins else "On Time"

    def _checkout_status(self, date: str, st: Dict[str, Any]) -> tuple[str, Optional[float]]:
        ci = self._to_minutes(st.get("check_in"))
        co = self._to_minutes(st.get("check_out"))
        if ci is None or co is None:
            return "Present", None
        hours = (co - ci) / 60.0
        shift_end = self._to_minutes(self.shift_end)
        early = shift_end is not None and co < shift_end - self.early_mins
        late = ci > self._to_minutes(self.shift_start) + self.late_mins if self._to_minutes(self.shift_start) else False
        if late:
            return "Late", round(hours, 2)
        if early:
            return "Early Departure", round(hours, 2)
        return "On Time", round(hours, 2)

    @staticmethod
    def _to_minutes(hm_time: Optional[str]) -> Optional[int]:
        if not hm_time or len(hm_time) < 5:
            return None
        try:
            h, m = int(hm_time[0:2]), int(hm_time[3:5])
            return h * 60 + m
        except ValueError:
            return None
