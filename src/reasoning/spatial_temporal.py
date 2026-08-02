from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.events.store import Event


@dataclass
class Verdict:
    event: Event
    action: str                       # "accept" | "reject" | "flip"
    direction: str                    # final direction after verdict
    note: str = ""
    void_previous_id: int = 0         # if rejecting a u-turn, also delete this id


class SpatialTemporalReasoning:
    """Applies contextual rules on top of raw line-crossing events.

    - Time-of-day windows: while the boundary is still uncalibrated,
      morning crossings are biased toward Check-In, evening toward Check-Out.
    - U-turn / loitering filter: a track crossing back within ``uturn_sec``
      invalidates both crossings (both events are deleted).
    - Anti-tailgating alert: two accepted crossings of different people in
      near-identical time are flagged.
    - Guest fallback: raw ``Unknown`` / ``ID:#`` labels get persistent
      ``Guest#NNN`` identities so attendance never loses a record.
    """

    def __init__(
        self,
        morning_window: Tuple[str, str] = ("07:00", "11:00"),
        evening_window: Tuple[str, str] = ("16:00", "20:00"),
        uturn_sec: float = 3.0,
        tailgate_sec: float = 2.0,
        window_bias: bool = False,
        enabled: bool = True,
    ) -> None:
        self.morning = morning_window
        self.evening = evening_window
        self.uturn_sec = float(uturn_sec)
        self.tailgate_sec = float(tailgate_sec)
        self.window_bias = bool(window_bias)
        self.enabled = enabled

        self._track_events: Dict[int, deque] = {}   # track_id -> accepted events
        self._guest_map: Dict[int, str] = {}
        self._guest_counter = 0
        self.alerts: List[str] = []

    # ── public API ─────────────────────────────────────────────────────────────

    def verify(
        self,
        events: List[Event],
        boundary_conf: float = 0.0,
        boundary_learned: bool = False,
    ) -> List[Verdict]:
        self.alerts = []
        if not self.enabled:
            return [Verdict(e, "accept", e.direction) for e in events]

        now = time.time()
        verdicts: List[Verdict] = []
        for e in events:
            self._ensure_person(e)

            action, direction, note = "accept", e.direction, ""
            void_id = 0

            # ── U-turn / loitering filter ────────────────────────────────
            prev = self._track_events.get(e.track_id)
            if prev:
                for p in prev:
                    if p.direction and p.direction != e.direction and abs(self._ts(p) - self._ts(e)) < self.uturn_sec:
                        action, note, void_id = "reject", "u-turn/loitering", p.id
                        break

            # ── Time-of-day bias (OPT-IN; off by default) ────────────────
            # Can only override the direction while the boundary is
            # uncalibrated. Default OFF: geometry-derived direction (the
            # configured seed line) is treated as ground truth, so an
            # evening "entry" stays an entry and always produces a check-in.
            if (
                action == "accept"
                and self.window_bias
                and not boundary_learned
                and boundary_conf < 0.5
            ):
                if self._in_window(e.time, self.morning) and e.direction == "exit":
                    action, direction, note = "flip", "entry", "morning-window-bias"
                elif self._in_window(e.time, self.evening) and e.direction == "entry":
                    action, direction, note = "flip", "exit", "evening-window-bias"

            verdicts.append(Verdict(e, action, direction, note, void_previous_id=void_id))

            if action == "accept":
                e.direction = direction
                q = prev if prev is not None else deque(maxlen=4)
                q.append(e)
                self._track_events[e.track_id] = q
                # ── anti-tailgating ─────────────────────────────────────
                for tid2, q2 in self._track_events.items():
                    if tid2 == e.track_id:
                        continue
                    for p in q2:
                        if abs(self._ts(p) - self._ts(e)) < self.tailgate_sec and p.person != e.person:
                            self.alerts.append(
                                f"TAILGATE {p.person} + {e.person} @ {e.time}"
                            )
        return verdicts

    # ── internals ──────────────────────────────────────────────────────────────

    def _ensure_person(self, e: Event) -> None:
        """Give raw Unknown / ID-labelled events persistent Guest identities."""
        if e.person and "Unknown" not in e.person and not e.person.startswith("ID:"):
            return
        gid = self._guest_map.get(e.track_id)
        if gid is None:
            self._guest_counter += 1
            gid = f"Guest#{self._guest_counter:03d}"
            self._guest_map[e.track_id] = gid
        e.person = gid

    @staticmethod
    def _ts(e: Event) -> float:
        dt = datetime.now().replace(
            hour=int(e.time[0:2]), minute=int(e.time[3:5]), second=int(e.time[6:8])
        )
        return dt.timestamp()

    @staticmethod
    def _in_window(time_str: str, window: Tuple[str, str]) -> bool:
        hm = time_str[0:5]
        return window[0] <= hm <= window[1]
