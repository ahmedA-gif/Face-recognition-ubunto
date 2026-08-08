from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.attendance.db import AttendanceDB
from src.events.store import Event


class AttendanceManager:
    """Daily attendance state machine, keyed strictly on person identity.

    States per person per day: NOT_ARRIVED → CHECKED_IN ⇄ ON_BREAK → CHECKED_OUT.

    - First entry  → Check-In (with Late / On Time status).
    - Exit while checked in → Check-Out.
    - Re-entry after checkout → person is back (clear checkout, CHECKED_IN).
    - Same-direction dedup only: exit after entry is never blocked by debounce.
    - State is hydrated from the DB so process restarts do not lose open visits.
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

    def process_events(self, events: List[Event]) -> List[Dict[str, Any]]:
        """Process events and return outcome details for each recorded one."""
        outcomes: List[Dict[str, Any]] = []
        for e in events:
            out = self._process_one(e)
            if out is not None:
                outcomes.append(out)
        return outcomes

    def summary(self, date: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.db.daily_logs(date)

    # ── internals ──────────────────────────────────────────────────────────────

    def _state_for(self, e: Event) -> Dict[str, Any]:
        states = self._state.setdefault(e.date, {})
        if e.person in states:
            return states[e.person]

        # Hydrate from DB so a process restart still knows who is checked in.
        st = self._hydrate_from_db(e.date, e.person)
        states[e.person] = st
        return st

    def _hydrate_from_db(self, date: str, person: str) -> Dict[str, Any]:
        row = self.db.get_log(date, person)
        if row is None:
            return {
                "state": "NOT_ARRIVED",
                "name": person,
                "check_in": None,
                "check_out": None,
                "last_event": 0.0,
                "last_direction": None,
                "conf": 0.0,
            }

        check_in = row.get("check_in_time")
        check_out = row.get("check_out_time")
        if check_in and check_out:
            state = "CHECKED_OUT"
        elif check_in:
            state = "CHECKED_IN"
        else:
            state = "NOT_ARRIVED"

        return {
            "state": state,
            "name": row.get("person_name") or person,
            "check_in": check_in,
            "check_out": check_out,
            "last_event": 0.0,
            "last_direction": "exit" if state == "CHECKED_OUT" else ("entry" if state == "CHECKED_IN" else None),
            "conf": float(row.get("confidence") or 0.0),
        }

    def _process_one(self, e: Event) -> Optional[Dict[str, Any]]:
        st = self._state_for(e)
        now = time.time()

        # Same-direction multi-track dedup only. Never block exit after entry
        # (or entry after exit) — that was wiping legitimate check-outs.
        if (
            st["last_event"] > 0
            and now - st["last_event"] < self.debounce_sec
            and st.get("last_direction") == e.direction
        ):
            st["last_event"] = now
            return None

        st["conf"] = max(st["conf"], float(e.confidence or 0.0))

        if e.direction == "entry":
            if st["state"] == "NOT_ARRIVED":
                st["state"] = "CHECKED_IN"
                st["check_in"] = e.time
                st["check_out"] = None
                status = self._checkin_status(e.date, e.time)
                self.db.upsert_log(
                    date=e.date, person_id=e.person, person_name=e.person,
                    check_in=e.time, check_out=None, status=status,
                    work_hours=None, confidence=st["conf"], camera_id=self.camera_id,
                )
                print(f"[ATTENDANCE] {e.date} | {e.person} | CHECK-IN {e.time} ({status})")
                st["last_event"] = now
                st["last_direction"] = "entry"
                return {
                    "person": e.person, "date": e.date, "time": e.time,
                    "action": "check_in", "status": status, "work_hours": None,
                }
            elif st["state"] == "ON_BREAK":
                st["state"] = "CHECKED_IN"
                print(f"[ATTENDANCE] {e.date} | {e.person} | BACK FROM BREAK {e.time}")
                st["last_event"] = now
                st["last_direction"] = "entry"
                return None
            elif st["state"] == "CHECKED_OUT":
                # Same-day re-entry: person is back; clear checkout, stay present.
                st["state"] = "CHECKED_IN"
                st["check_out"] = None
                status = self._checkin_status(e.date, st["check_in"] or e.time)
                self.db.upsert_log(
                    date=e.date, person_id=e.person, person_name=e.person,
                    check_in=st["check_in"] or e.time, check_out=None, status=status,
                    work_hours=None, confidence=st["conf"], camera_id=self.camera_id,
                    clear_checkout=True,
                )
                print(f"[ATTENDANCE] {e.date} | {e.person} | RE-ENTRY {e.time} (back on site)")
                st["last_event"] = now
                st["last_direction"] = "entry"
                return {
                    "person": e.person, "date": e.date, "time": e.time,
                    "action": "re_entry", "status": status, "work_hours": None,
                }
            # Already CHECKED_IN: ignore duplicate entry (debounce covers most cases).
            st["last_event"] = now
            st["last_direction"] = "entry"
            return None

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
                st["last_event"] = now
                st["last_direction"] = "exit"
                return {
                    "person": e.person, "date": e.date, "time": e.time,
                    "action": "check_out", "status": status, "work_hours": hours,
                }
            elif st["state"] == "NOT_ARRIVED":
                print(f"[ATTENDANCE] {e.date} | {e.person} | exit without check-in (ignored)")
            elif st["state"] == "CHECKED_OUT":
                # Duplicate exit — ignore.
                pass

        st["last_event"] = now
        st["last_direction"] = e.direction
        return None

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
