"""Simplified Door Intelligence Engine - Easy to understand, guaranteed to work.

This is a drop-in replacement for door_intelligence.py with a much simpler
state machine that's easier to debug and maintain.

States:
- OUTSIDE: Person is in OUTSIDE zone
- INSIDE: Person is in INSIDE zone  
- DOOR: Person is in DOOR zone

Events:
- ENTRY: When person moves from OUTSIDE/DOOR to INSIDE with inward motion
- EXIT: When person moves from INSIDE/DOOR to OUTSIDE with outward motion

No complex intermediate states (APPROACHING, CROSSING, etc.) - just track
zone and direction.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, Hashable, List, Optional, Tuple

from src.events.store import Event, EventsStore
from src.tracking.bytetrack import Track
from src.tracking.motion import TrackMotion
from src.utils.geometry import (
    Polygon,
    foot_point,
    inward_normal,
    region_for_point,
)


class SimpleState(Enum):
    OUTSIDE = "OUTSIDE"
    INSIDE = "INSIDE"
    DOOR = "DOOR"


@dataclass
class SimplePersonTrack:
    track_id: int
    global_id: str = ""
    current_zone: str = ""
    previous_zone: str = ""
    last_event: Optional[str] = None
    last_seen: float = 0.0
    velocity: Tuple[float, float] = (0.0, 0.0)
    last_seen: float = 0.0
    history: Deque[Tuple[float, float, float, str]] = field(
        default_factory=lambda: deque(maxlen=90)
    )
    region_path: List[str] = field(default_factory=list)


class SimpleDoorIntelligenceEngine:
    """Simplified door intelligence - just track zone transitions with direction."""

    def __init__(
        self,
        zones: Dict[str, Polygon],
        camera_id: str = "cam_01",
        probe: str = "foot",
        min_track_frames: int = 5,
        min_frames_in_zone: int = 3,
        motion_toward_inside_dot: float = 0.25,
        motion_outward_dot: float = 0.25,
        purge_after_sec: float = 10.0,
        min_motion_speed: float = 0.6,
    ) -> None:
        if probe not in ("foot", "centroid"):
            raise ValueError(f"probe must be 'foot' or 'centroid', got {probe!r}")
        self.zones = {k: [list(p) for p in v] for k, v in zones.items()}
        self.camera_id = camera_id
        self.probe = probe
        self.min_track_frames = int(min_track_frames)
        self.min_frames_in_zone = int(min_frames_in_zone)
        self.motion_toward_inside_dot = float(motion_toward_inside_dot)
        self.motion_outward_dot = float(motion_outward_dot)
        self.purge_after_sec = float(purge_after_sec)
        self.min_motion_speed = float(min_motion_speed)

        self.motion = TrackMotion(min_speed=self.min_motion_speed)
        self._tracks: Dict[Hashable, SimplePersonTrack] = {}
        self.counts: Dict[str, int] = {"entry": 0, "exit": 0, "present": 0}
        self._inward_norm: Optional[Tuple[float, float]] = inward_normal(self.zones)
        self._px_zones: Optional[Dict[str, List[List[float]]]] = None

    def update(
        self,
        tracks: List[Track],
        frame_shape: Tuple[int, int, int],
        store: EventsStore,
    ) -> List[Event]:
        now = time.time()
        purge_events = self._purge(now)
        h, w = frame_shape[:2]
        self._px_zones = self._pixel_zones(w, h)

        produced: List[Event] = []
        for ev in purge_events:
            ev.id = store.insert(ev)
            produced.append(ev)
            self.counts[ev.direction] = self.counts.get(ev.direction, 0) + 1
            self.counts["present"] = self.counts.get("entry", 0) - self.counts.get("exit", 0)

        alive: set = set()
        for t in tracks:
            key = self._state_key(t)
            alive.add(key)
            ft = self._tracks.get(key)
            if ft is None:
                ft = SimplePersonTrack(track_id=t.track_id)
                self._tracks[key] = ft
            self._sync(ft, t, now)

            point = self._probe_point(t)
            zone = self._zone_for(point)
            vx, vy = self.motion.update(key, point[0], point[1], now)
            ft.velocity = (vx, vy)
            ft.last_seen = now
            ft.history.append((now, point[0], point[1], zone))

            direction = self._motion_direction(vx, vy)
            self._append_zone_path(ft, zone)

            if t.hits < self.min_track_frames:
                # Warm-up: learn the zone, never emit
                ft.state = SimpleState.OUTSIDE if zone in ("OUTSIDE", "DOOR") else SimpleState.INSIDE
                ft.current_zone = zone
                ft.previous_zone = zone
            else:
                event = self._step(ft, zone, direction, now)
                t.meta["fsm_state"] = ft.current_zone  # Use zone as state for simplicity
                t.meta["zone"] = zone
                t.prev_side = zone

                if event is not None:
                    event.id = store.insert(event)
                    produced.append(event)
                    self.counts[event.direction] = self.counts.get(event.direction, 0) + 1
                    self.counts["present"] = self.counts.get("entry", 0) - self.counts.get("exit", 0)
                    ft.last_event = event.direction.upper()

        self._forget_removed(alive)
        return produced

    # --- state helpers -----------------------------------------------------------

    def _state_key(self, t: Track) -> Hashable:
        gid = t.meta.get("global_id") if t.meta else None
        if gid:
            return ("gid", gid)
        return ("tid", t.track_id)

    def _sync(self, ft: SimplePersonTrack, t: Track, now: float) -> None:
        ft.track_id = t.track_id
        gid = t.meta.get("global_id") if t.meta else None
        if gid:
            ft.global_id = gid
        if t.person_name and not t.person_name.startswith(("Unknown", "ID:")):
            ft.face_id = t.person_name
        ft.confidence = float(t.face_score or t.conf)

    def _probe_point(self, t: Track):
        if self.probe == "centroid":
            return t.centroid
        return foot_point(t.xyxy)

    def _zone_for(self, point) -> str:
        zones = self._px_zones if self._px_zones is not None else self.zones
        return region_for_point(point, zones) or "OUTSIDE"

    def _pixel_zones(self, w: float, h: float) -> Dict[str, List[List[float]]]:
        return {
            name: [[x * w, y * h] for (x, y) in poly]
            for name, poly in self.zones.items()
        }

    def _motion_direction(self, vx: float, vy: float) -> str:
        speed = (vx * vx + vy * vy) ** 0.5
        if speed < self.min_motion_speed:
            return "ambiguous"
        n = self._inward_norm
        if n is None:
            return "ambiguous"
        dot = (vx * n[0] + vy * n[1]) / speed
        if dot >= self.motion_toward_inside_dot:
            return "inward"
        if dot <= -self.motion_outward_dot:
            return "outward"
        return "ambiguous"

    def _append_zone_path(self, ft: SimplePersonTrack, zone: str) -> None:
        if zone != ft.current_zone:
            if ft.current_zone:
                if not ft.region_path or ft.region_path[-1] != ft.current_zone:
                    ft.region_path.append(ft.current_zone)
            ft.previous_zone = ft.current_zone
            ft.current_zone = zone

    # --- FSM ----------------------------------------------------------------

    def _step(
        self,
        ft: SimplePersonTrack,
        zone: str,
        direction: str,
        now: float,
    ) -> Optional[Event]:
        """Simple FSM: just track zone transitions with direction."""
        prev_zone = ft.current_zone

        # If zone didn't change, no event possible
        if zone == prev_zone:
            ft.current_zone = zone
            return None

        # Zone changed - check for valid transitions
        # Valid entry: OUTSIDE/DOOR -> INSIDE with inward motion
        # Valid exit: INSIDE/DOOR -> OUTSIDE with outward motion

        if zone == "INSIDE" and direction == "inward":
            # Moving into inside - this is ENTRY
            # But only if we came from OUTSIDE or DOOR
            if prev_zone in ("OUTSIDE", "DOOR"):
                return self._fire(ft, "entry", zone)

        elif zone == "OUTSIDE" and direction == "outward":
            # Moving into outside - this is EXIT
            # But only if we came from INSIDE or DOOR
            if prev_zone in ("INSIDE", "DOOR"):
                return self._fire(ft, "exit", zone)

        # Update current zone
        ft.current_zone = zone
        return None

    # --- events --------------------------------------------------------------

    def _fire(self, ft: SimplePersonTrack, direction: str, zone: str) -> Event:
        date_s, time_s = EventsStore.now_parts()
        person = ft.face_id or ft.global_id or f"Unknown#{ft.track_id}"
        path = list(ft.region_path)
        if zone and (not path or path[-1] != zone):
            path.append(zone)
        return Event(
            date=date_s,
            time=time_s,
            person=person,
            direction=direction,
            track_id=ft.track_id,
            camera_id=self.camera_id,
            confidence=ft.confidence,
            global_id=ft.global_id,
            fsm_path=path,
            event_id=str(uuid.uuid4()),
        )

    # --- lifecycle ------------------------------------------------------------

    def _purge(self, now: float) -> List[Event]:
        stale = [
            key
            for key, ft in self._tracks.items()
            if ft.last_seen and now - ft.last_seen > self.purge_after_sec
        ]
        events: List[Event] = []
        for key in stale:
            ft = self._tracks[key]
            # If track was last seen in INSIDE, generate implicit EXIT
            if ft.current_zone == "INSIDE":
                events.append(self._fire(ft, "exit", "OUTSIDE"))
            self.motion.forget(key)
            self._tracks.pop(key, None)
        return events

    def _forget_removed(self, alive: set) -> None:
        for key in [k for k in self._tracks if k not in alive]:
            if self._tracks[key].last_seen == 0.0:
                continue
            # Leave recently-active tracks in place so short tracker gaps do
            # not wipe FSM state; _purge handles the long timeout.
