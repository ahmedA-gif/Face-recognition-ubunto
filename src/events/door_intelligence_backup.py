"""Door Intelligence Engine — polygon-based entry/exit FSM.

Replaces the single virtual line with three regions (OUTSIDE / DOOR / INSIDE)
and a per-track finite state machine. Every person is always assigned to a
region via their **foot point**; direction comes from Kalman-smoothed velocity
dotted with the door's inward normal. Events are emitted per identity
(``global_id`` when identity fusion is active, else raw ``track_id``), so ten
people walking in together produce ten independent, correctly-attributed
events — never one global frame rule.

Stand-at-door, peek, u-turn and corridor-edge jitter are suppressed by the
dwell timer, min-frame gates, motion-direction gate and the event lock.
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


class TrackState(Enum):
    UNKNOWN = 0
    OUTSIDE = 1
    APPROACHING_DOOR = 2
    CROSSING_IN = 3
    INSIDE = 4
    APPROACHING_EXIT = 5
    CROSSING_OUT = 6
    LOCKED_AFTER_ENTRY = 7
    LOCKED_AFTER_EXIT = 8


@dataclass
class PersonTrack:
    track_id: int
    global_id: str = ""
    current_zone: str = ""
    previous_zone: str = ""
    state: TrackState = TrackState.UNKNOWN
    last_event: Optional[str] = None       # 'ENTRY' | 'EXIT' | None
    last_seen: float = 0.0
    velocity: Tuple[float, float] = (0.0, 0.0)
    dwell_time_in_door: float = 0.0
    is_event_locked: bool = False
    face_id: Optional[str] = None
    confidence: float = 0.0

    door_enter_time: float = 0.0           # when zone first became DOOR
    crossing_start_time: float = 0.0
    inside_frames: int = 0
    outside_frames: int = 0
    inward_frames: int = 0
    outward_frames: int = 0

    history: Deque[Tuple[float, float, float, str]] = field(
        default_factory=lambda: deque(maxlen=90)
    )
    region_path: List[str] = field(default_factory=list)  # compressed zone path


class DoorIntelligenceEngine:
    """Per-track polygon FSM. Drop-in for ``EntryExitEngine.update``."""

    def __init__(
        self,
        zones: Dict[str, Polygon],
        camera_id: str = "cam_01",
        probe: str = "foot",
        min_track_frames: int = 5,
        min_dwell_door_sec: float = 0.15,
        min_inside_frames: int = 3,
        min_outside_frames: int = 3,
        lock_after_event: bool = True,
        motion_toward_inside_dot: float = 0.25,
        motion_outward_dot: float = 0.25,
        peek_max_inside_sec: float = 1.5,
        uturn_max_door_sec: float = 3.0,
        min_event_confidence: float = 0.70,
        purge_after_sec: float = 10.0,
        min_motion_speed: float = 0.6,
    ) -> None:
        if probe not in ("foot", "centroid"):
            raise ValueError(f"probe must be 'foot' or 'centroid', got {probe!r}")
        self.zones = {k: [list(p) for p in v] for k, v in zones.items()}
        self.camera_id = camera_id
        self.probe = probe
        self.min_track_frames = int(min_track_frames)
        self.min_dwell_door_sec = float(min_dwell_door_sec)
        self.min_inside_frames = int(min_inside_frames)
        self.min_outside_frames = int(min_outside_frames)
        self.lock_after_event = bool(lock_after_event)
        self.motion_toward_inside_dot = float(motion_toward_inside_dot)
        self.motion_outward_dot = float(motion_outward_dot)
        self.peek_max_inside_sec = float(peek_max_inside_sec)
        self.uturn_max_door_sec = float(uturn_max_door_sec)
        self.min_event_confidence = float(min_event_confidence)
        self.purge_after_sec = float(purge_after_sec)
        self.min_motion_speed = float(min_motion_speed)

        self.motion = TrackMotion(min_speed=self.min_motion_speed)
        self._tracks: Dict[Hashable, PersonTrack] = {}
        self.counts: Dict[str, int] = {"entry": 0, "exit": 0, "present": 0}
        self._inward_norm: Optional[Tuple[float, float]] = inward_normal(self.zones)
        self._px_zones: Optional[Dict[str, List[List[float]]]] = None

    # ── public API ─────────────────────────────────────────────────────────────

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
                ft = PersonTrack(track_id=t.track_id)
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
                # Warm-up: learn the zone, never emit until enough hits.
                ft.state = TrackState.UNKNOWN
            else:
                event = self._step(ft, zone, direction, now)
                t.meta["fsm_state"] = ft.state.name
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

    # ── state helpers ──────────────────────────────────────────────────────────

    def _state_key(self, t: Track) -> Hashable:
        gid = t.meta.get("global_id") if t.meta else None
        if gid:
            return ("gid", gid)
        return ("tid", t.track_id)

    def _sync(self, ft: PersonTrack, t: Track, now: float) -> None:
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
        """Scale normalized (0–1) zone polygons to pixel coordinates."""
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

    def _append_zone_path(self, ft: PersonTrack, zone: str) -> None:
        if zone != ft.current_zone:
            if ft.current_zone:
                if not ft.region_path or ft.region_path[-1] != ft.current_zone:
                    ft.region_path.append(ft.current_zone)
            ft.previous_zone = ft.current_zone
            ft.current_zone = zone

    # ── FSM ────────────────────────────────────────────────────────────────────

    def _step(
        self,
        ft: PersonTrack,
        zone: str,
        direction: str,
        now: float,
    ) -> Optional[Event]:
        st = ft.state

        # Warm-up: learn the zone, never emit until the track has enough hits
        # (gated in update() via t.hits). First sighting in the door band is
        # treated as "outside" until motion resolves direction.
        if st == TrackState.UNKNOWN:
            ft.current_zone = zone
            ft.previous_zone = zone
            ft.state = TrackState.OUTSIDE if zone in ("OUTSIDE", "DOOR") else TrackState.INSIDE
            return None

        if st == TrackState.OUTSIDE:
            if zone == "DOOR":
                ft.door_enter_time = now
                # Only approach if moving inward; otherwise stay outside
                if direction == "inward":
                    ft.state = TrackState.APPROACHING_DOOR
                else:
                    # Moving outward or ambiguous - not approaching the door
                    ft.state = TrackState.OUTSIDE
            elif zone == "INSIDE":
                # Skipped the door band: entry only with inward motion + stability.
                ft.inside_frames += 1
                if direction == "inward" and ft.inside_frames >= self.min_inside_frames:
                    return self._fire(ft, "entry")
                ft.state = TrackState.CROSSING_IN if direction == "inward" else TrackState.INSIDE
            else:
                ft.inside_frames = 0
            return None

        if st == TrackState.APPROACHING_DOOR:
            if direction == "inward":
                ft.inward_frames += 1
            if zone == "DOOR":
                ft.dwell_time_in_door = now - ft.door_enter_time
                if ft.dwell_time_in_door > self.uturn_max_door_sec:
                    ft.state = TrackState.OUTSIDE  # loitered and never crossed
                    ft.door_enter_time = now
                elif ft.dwell_time_in_door > self.min_dwell_door_sec and direction == "inward":
                    # Only cross if moving inward
                    ft.state = TrackState.CROSSING_IN
                    ft.crossing_start_time = now
                elif ft.dwell_time_in_door > self.min_dwell_door_sec and direction == "outward":
                    # Moving outward, back to outside
                    ft.state = TrackState.OUTSIDE
                    ft.door_enter_time = now
            elif zone == "INSIDE":
                # Moving to inside - must be inward to cross
                if direction == "inward":
                    ft.state = TrackState.CROSSING_IN
                    ft.crossing_start_time = now
                else:
                    # Moving to inside but direction is outward - impossible, stay in door
                    ft.state = TrackState.APPROACHING_DOOR
            else:  # back outside — turn back, no event
                ft.state = TrackState.OUTSIDE
            return None

        if st == TrackState.CROSSING_IN:
            if direction == "inward":
                ft.inward_frames += 1
            if zone == "INSIDE":
                ft.inside_frames += 1
                # Only fire entry if moving inward or stationary (not moving back out)
                if ft.inside_frames >= self.min_inside_frames and direction != "outward":
                    return self._fire(ft, "entry")
            elif zone == "DOOR":
                ft.dwell_time_in_door = now - ft.door_enter_time
                if (
                    ft.dwell_time_in_door > self.uturn_max_door_sec
                    and direction == "outward"
                ):
                    ft.state = TrackState.OUTSIDE  # walked in, turned around
            else:
                ft.state = TrackState.OUTSIDE
                ft.inside_frames = 0
            return None

        if st == TrackState.INSIDE:
            if zone == "DOOR":
                ft.door_enter_time = now
                # Only approach exit if moving outward; otherwise stay inside
                if direction == "outward":
                    ft.state = TrackState.APPROACHING_EXIT
                else:
                    # Moving inward or ambiguous - not exiting
                    ft.state = TrackState.INSIDE
            elif zone == "OUTSIDE":
                # Person moved from INSIDE directly to OUTSIDE — clear exit.
                # Go directly to CROSSING_OUT without requiring direction
                # (Kalman velocity can be stale after direction change).
                ft.outside_frames += 1
                if ft.outside_frames >= self.min_outside_frames:
                    return self._fire(ft, "exit")
                ft.state = TrackState.CROSSING_OUT
            else:
                ft.inside_frames = 0
                ft.outside_frames = 0
            return None

        if st == TrackState.APPROACHING_EXIT:
            if direction == "outward":
                ft.outward_frames += 1
            if zone == "DOOR":
                ft.dwell_time_in_door = now - ft.door_enter_time
                if (
                    ft.dwell_time_in_door > self.peek_max_inside_sec
                    and direction == "inward"
                ):
                    ft.state = TrackState.INSIDE  # peeked out and stepped back
                elif ft.dwell_time_in_door > self.min_dwell_door_sec and direction == "outward":
                    # Only cross if moving outward
                    ft.state = TrackState.CROSSING_OUT
                    ft.crossing_start_time = now
                elif ft.dwell_time_in_door > self.min_dwell_door_sec and direction == "inward":
                    # Moving inward, back to inside
                    ft.state = TrackState.INSIDE
                    ft.door_enter_time = now
            elif zone == "OUTSIDE":
                # Moving to outside - must be outward to cross
                if direction == "outward":
                    ft.state = TrackState.CROSSING_OUT
                    ft.crossing_start_time = now
                else:
                    # Moving to outside but direction is inward - impossible, stay in door
                    ft.state = TrackState.APPROACHING_EXIT
            else:  # stepped back inside
                ft.state = TrackState.INSIDE
            return None

        if st == TrackState.CROSSING_OUT:
            if direction == "outward":
                ft.outward_frames += 1
            if zone == "OUTSIDE":
                ft.outside_frames += 1
                # Only fire exit if moving outward or stationary (not moving back in)
                if ft.outside_frames >= self.min_outside_frames and direction != "inward":
                    return self._fire(ft, "exit")
            elif zone == "DOOR":
                ft.dwell_time_in_door = now - ft.door_enter_time
                if (
                    ft.dwell_time_in_door > self.peek_max_inside_sec
                    and direction == "inward"
                ):
                    ft.state = TrackState.INSIDE
            else:
                ft.state = TrackState.INSIDE
                ft.outside_frames = 0
            return None

        if st == TrackState.LOCKED_AFTER_ENTRY:
            # ENTRY already fired. Allow the person to walk back out through
            # the normal exit path: DOOR → OUTSIDE.  We go directly to
            # CROSSING_OUT (skipping APPROACHING_EXIT's direction gate) since
            # we already know the person is leaving from the zone transition.
            if zone == "OUTSIDE":
                ft.outside_frames += 1
                if ft.outside_frames >= self.min_outside_frames:
                    return self._fire(ft, "exit")
                ft.state = TrackState.CROSSING_OUT
            elif zone == "DOOR":
                ft.outside_frames = 0
                ft.outward_frames = 1  # pre-set so CROSSING_OUT's gate passes
                ft.state = TrackState.CROSSING_OUT
                ft.crossing_start_time = now
                ft.door_enter_time = now
            else:
                # Still inside — keep waiting.
                pass
            return None

        if st == TrackState.LOCKED_AFTER_EXIT:
            # EXIT already fired. Allow re-entry through the normal path:
            # DOOR → INSIDE.  Go directly to CROSSING_IN (skip
            # APPROACHING_DOOR's direction gate).
            if zone == "INSIDE":
                ft.inside_frames += 1
                if ft.inside_frames >= self.min_inside_frames:
                    return self._fire(ft, "entry")
                ft.state = TrackState.CROSSING_IN
            elif zone == "DOOR":
                ft.inside_frames = 0
                ft.inward_frames = 1  # pre-set so CROSSING_IN's gate passes
                ft.state = TrackState.CROSSING_IN
                ft.crossing_start_time = now
                ft.door_enter_time = now
            else:
                # Still outside — keep waiting.
                pass
            return None

        return None

    # ── events ─────────────────────────────────────────────────────────────────

    def _fire(self, ft: PersonTrack, direction: str) -> Event:
        if self.lock_after_event:
            ft.is_event_locked = True
            ft.state = (
                TrackState.LOCKED_AFTER_ENTRY
                if direction == "entry"
                else TrackState.LOCKED_AFTER_EXIT
            )
        else:
            # No lock: land in the zone the person just reached so the FSM
            # is ready for the next natural crossing (e.g. INSIDE after entry,
            # OUTSIDE after exit) without immediately re-firing.
            ft.state = TrackState.INSIDE if direction == "entry" else TrackState.OUTSIDE
            ft.inside_frames = 0
            ft.outside_frames = 0
            ft.inward_frames = 0
            ft.outward_frames = 0
        date_s, time_s = EventsStore.now_parts()
        person = ft.face_id or ft.global_id or f"Unknown#{ft.track_id}"
        path = list(ft.region_path)
        if ft.current_zone and (not path or path[-1] != ft.current_zone):
            path.append(ft.current_zone)
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

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def _purge(self, now: float) -> List[Event]:
        """Remove stale tracks.  Fire EXIT for tracks purged in LOCKED_AFTER_ENTRY."""
        stale = [
            key
            for key, ft in self._tracks.items()
            if ft.last_seen and now - ft.last_seen > self.purge_after_sec
        ]
        events: List[Event] = []
        for key in stale:
            ft = self._tracks[key]
            if ft.state == TrackState.LOCKED_AFTER_ENTRY:
                events.append(self._fire(ft, "exit"))
            self.motion.forget(key)
            self._tracks.pop(key, None)
        return events

    def _forget_removed(self, alive: set) -> None:
        for key in [k for k in self._tracks if k not in alive]:
            if self._tracks[key].last_seen == 0.0:
                continue
            # Leave recently-active tracks in place so short tracker gaps do
            # not wipe FSM state; _purge handles the long timeout.
