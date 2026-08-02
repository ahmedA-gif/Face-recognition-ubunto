from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.events.store import Event, EventsStore
from src.tracking.bytetrack import Track
from src.utils.geometry import _cross, line_crossed


@dataclass
class EntryExitEngine:
    """
    Produces entry/exit events when a track centroid crosses the virtual line.

    Hysteresis: a crossing only fires after the centroid has moved ``hysteresis_px``
    beyond the line into the opposite zone. While inside the ±margin band the
    previous zone is sticky, so centroid jitter (vertical bobbing near a
    diagonal line) cannot produce false double-events.
    """

    line_norm: Dict[str, float]  # x1,y1,x2,y2 in 0..1
    entry_direction: str = "A_to_B"  # A_to_B => entry
    debounce_sec: float = 1.5
    min_track_frames: int = 8
    hysteresis_px: float = 12.0
    camera_id: str = "cam_01"
    counts: Dict[str, int] = field(default_factory=lambda: {"entry": 0, "exit": 0, "present": 0})

    def __post_init__(self) -> None:
        self._zones: Dict[int, str] = {}  # track_id -> current zone (A|B)

    def _abs_line(self, frame_w: int, frame_h: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (
            (self.line_norm["x1"] * frame_w, self.line_norm["y1"] * frame_h),
            (self.line_norm["x2"] * frame_w, self.line_norm["y2"] * frame_h),
        )

    def set_line(self, line_norm: Dict[str, float]) -> None:
        """Swap the boundary line (e.g. after auto-boundary learning).

        Resets per-track zone state so tracks re-warm against the new line
        and cannot fire spurious events on a mid-run line swap.
        """
        self.line_norm = dict(line_norm)
        self._zones.clear()

    def _signed_dist(self, point: Tuple[float, float], line: Tuple[Point, Point]) -> float:
        (x1, y1), (x2, y2) = line
        px, py = point
        dx, dy = x2 - x1, y2 - y1
        length = (dx * dx + dy * dy) ** 0.5 or 1e-9
        return _cross(x1, y1, x2, y2, px, py) / length

    def _zone_for(self, track_id: int, d: float, margin: float) -> str:
        if d > margin:
            return "A"
        if d < -margin:
            return "B"
        return self._zones.get(track_id, "A" if d >= 0 else "B")

    def _direction_from_transition(self, transition: str) -> Optional[str]:
        # transition like "A_to_B"
        if transition == self.entry_direction:
            return "entry"
        a, _, b = transition.partition("_to_")
        opposite = f"{b}_to_{a}"
        if opposite == self.entry_direction or transition != self.entry_direction:
            # if configured entry is A_to_B, then B_to_A is exit
            if transition == "A_to_B":
                return "entry" if self.entry_direction == "A_to_B" else "exit"
            if transition == "B_to_A":
                return "exit" if self.entry_direction == "A_to_B" else "entry"
        return None

    def update(
        self,
        tracks: List[Track],
        frame_shape: Tuple[int, int, int],
        store: EventsStore,
    ) -> List[Event]:
        h, w = frame_shape[:2]
        line = self._abs_line(w, h)
        margin = self.hysteresis_px
        produced: List[Event] = []
        now = time.time()

        for t in tracks:
            d = self._signed_dist(t.centroid, line)

            if t.hits < self.min_track_frames:
                # warm up zone without eventing
                zone = "A" if d >= 0 else "B"
                self._zones[t.track_id] = zone
                t.prev_side = zone
                continue

            zone = self._zone_for(t.track_id, d, margin)
            prev_zone = self._zones.get(t.track_id)
            transition = line_crossed(prev_zone, zone)
            if transition:
                if now - t.last_event_at < self.debounce_sec:
                    self._zones[t.track_id] = zone
                    t.prev_side = zone
                    continue
                direction = self._direction_from_transition(transition)
                if direction is None:
                    self._zones[t.track_id] = zone
                    t.prev_side = zone
                    continue

                date_s, time_s = EventsStore.now_parts()
                person = t.person_name or f"Unknown#{t.track_id}"
                ev = Event(
                    date=date_s,
                    time=time_s,
                    person=person,
                    direction=direction,
                    track_id=t.track_id,
                    camera_id=self.camera_id,
                    confidence=float(t.face_score or t.conf),
                )
                ev.id = store.insert(ev)
                produced.append(ev)
                t.last_event_at = now
                self.counts[direction] = self.counts.get(direction, 0) + 1
                self.counts["present"] = self.counts.get("entry", 0) - self.counts.get("exit", 0)

            self._zones[t.track_id] = zone
            t.prev_side = zone

        return produced


Point = Tuple[float, float]
