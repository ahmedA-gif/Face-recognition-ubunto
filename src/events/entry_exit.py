from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Tuple

from src.events.store import Event, EventsStore
from src.tracking.bytetrack import Track
from src.utils.geometry import (
    foot_point,
    is_near_segment,
    line_crossed,
    signed_distance,
)

Point = Tuple[float, float]
Line = Tuple[Point, Point]


@dataclass
class EntryExitEngine:
    """
    Produces entry/exit events when a track crosses the virtual door segment.

    Design rules that keep counts correct on a door camera:

    - **Foot point**: use bottom-center of the person box, not body centroid,
      so the threshold matches where feet actually cross the door.
    - **Segment gate**: only fire when the foot projection lands on the
      configured door segment (not the infinite line extension).
    - **Hysteresis**: a crossing only fires after the foot has moved
      ``hysteresis_px`` beyond the line into the opposite zone. Inside the
      ±margin band the previous zone is sticky (no jitter flips).
    - **Committed zone**: the previous zone must have been observed deep
      (outside the band) for ``min_track_frames`` before a reverse can fire.
    - **Identity key**: zone / debounce state is keyed by ``global_id`` when
      identity fusion provides one, so ByteTrack fragment IDs do not re-warm
      and double-count the same person.
    - **Sticky debounce**: when a reverse is suppressed by debounce, the zone
      is NOT flipped — so a real exit still fires once debounce expires.
    """

    line_norm: Dict[str, float]  # x1,y1,x2,y2 in 0..1
    entry_direction: str = "A_to_B"  # which transition = entry
    debounce_sec: float = 1.5
    min_track_frames: int = 5
    hysteresis_px: float = 14.0
    camera_id: str = "cam_01"
    use_foot_point: bool = True
    segment_pad: float = 0.12
    require_committed_zone: bool = True
    counts: Dict[str, int] = field(default_factory=lambda: {"entry": 0, "exit": 0, "present": 0})

    def __post_init__(self) -> None:
        # key -> current sticky zone (A|B)
        self._zones: Dict[Hashable, str] = {}
        # key -> frames spent deep in current zone (outside hysteresis band)
        self._deep_frames: Dict[Hashable, int] = {}
        # key -> last event wall time (identity-level debounce)
        self._last_event_at: Dict[Hashable, float] = {}

    def _abs_line(self, frame_w: int, frame_h: int) -> Line:
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
        self._deep_frames.clear()
        # Keep last-event times so debounce still suppresses spam after a swap.

    def _state_key(self, t: Track) -> Hashable:
        gid = t.meta.get("global_id") if t.meta else None
        if gid:
            return ("gid", gid)
        return ("tid", t.track_id)

    def _sample_point(self, t: Track) -> Point:
        if self.use_foot_point:
            return foot_point(t.xyxy)
        return t.centroid

    def _zone_for(self, key: Hashable, d: float, margin: float) -> str:
        if d > margin:
            return "A"
        if d < -margin:
            return "B"
        # Inside the band: sticky previous zone, or raw side if first sighting.
        prev = self._zones.get(key)
        if prev is not None:
            return prev
        return "A" if d >= 0 else "B"

    def _direction_from_transition(self, transition: str) -> Optional[str]:
        if transition == self.entry_direction:
            return "entry"
        a, _, b = transition.partition("_to_")
        if not a or not b:
            return None
        opposite = f"{b}_to_{a}"
        if self.entry_direction == opposite:
            return "exit"
        # Fallback for unexpected entry_direction strings: treat A_to_B as entry
        # only when configured that way; otherwise map opposites consistently.
        if transition in ("A_to_B", "B_to_A"):
            return "entry" if transition == self.entry_direction else "exit"
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
            key = self._state_key(t)
            point = self._sample_point(t)
            d = signed_distance(point, line)
            deep = abs(d) > margin

            # Warm-up: learn zone, never emit. Also skip if still too few hits.
            if t.hits < self.min_track_frames:
                zone = "A" if d >= 0 else "B"
                if deep:
                    self._zones[key] = zone
                    self._deep_frames[key] = self._deep_frames.get(key, 0) + 1
                else:
                    # First sighting inside the band — remember raw side only.
                    self._zones.setdefault(key, zone)
                t.prev_side = self._zones.get(key, zone)
                continue

            # Segment gate: ignore motion that is not across the door opening.
            if not is_near_segment(point, line, pad=self.segment_pad):
                # Still refresh deep-frame counters if we already have a zone and
                # the person is clearly on one side (walking past the door).
                if deep:
                    zone_side = "A" if d > 0 else "B"
                    if self._zones.get(key) == zone_side:
                        self._deep_frames[key] = self._deep_frames.get(key, 0) + 1
                    else:
                        # Off-segment side change: adopt new side without eventing.
                        self._zones[key] = zone_side
                        self._deep_frames[key] = 1
                t.prev_side = self._zones.get(key)
                continue

            zone = self._zone_for(key, d, margin)
            prev_zone = self._zones.get(key)
            transition = line_crossed(prev_zone, zone)

            if transition:
                # Require the old zone to have been "committed" (seen deep long enough).
                if self.require_committed_zone:
                    committed = self._deep_frames.get(key, 0) >= max(1, self.min_track_frames // 2)
                    if not committed:
                        # Adopt new zone without firing — not enough evidence yet.
                        if deep:
                            self._zones[key] = zone
                            self._deep_frames[key] = 1
                        t.prev_side = self._zones.get(key, zone)
                        continue

                last_at = self._last_event_at.get(key, t.last_event_at)
                if now - last_at < self.debounce_sec:
                    # Sticky debounce: do NOT flip zone. Real reverse waits.
                    t.prev_side = prev_zone
                    continue

                direction = self._direction_from_transition(transition)
                if direction is None:
                    if deep:
                        self._zones[key] = zone
                        self._deep_frames[key] = 1
                    t.prev_side = self._zones.get(key, zone)
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
                self._last_event_at[key] = now
                self.counts[direction] = self.counts.get(direction, 0) + 1
                self.counts["present"] = self.counts.get("entry", 0) - self.counts.get("exit", 0)

                # Commit the new zone after a successful event.
                self._zones[key] = zone
                self._deep_frames[key] = 1 if deep else 0
                t.prev_side = zone
                continue

            # No transition: refresh zone + deep-frame counter.
            if deep:
                if self._zones.get(key) == zone:
                    self._deep_frames[key] = self._deep_frames.get(key, 0) + 1
                else:
                    self._zones[key] = zone
                    self._deep_frames[key] = 1
            else:
                self._zones.setdefault(key, zone)

            t.prev_side = self._zones.get(key, zone)

        return produced
