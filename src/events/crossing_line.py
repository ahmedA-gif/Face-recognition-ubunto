"""Crossing Line Engine — simple line-based entry/exit detection.

Inspired by Ultralytics ObjectCounter: detects when a tracked person's path
crosses a configured line, determines IN/OUT direction from the crossing.

Each track is counted AT MOST once per direction (like ObjectCounter's
counted_ids). Once a track is counted for entry, it won't be counted again
until the track disappears and reappears as a new track.

Configuration (settings.yaml):
    crossing_line:
        enabled: true
        line: {x1: 0.1, y1: 0.83, x2: 0.9, y2: 0.83}
        entry_direction: downward
        min_track_frames: 3
        cooldown_sec: 2.0
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Dict, Hashable, List, Optional, Tuple

from shapely.geometry import LineString, Point

from src.events.store import Event, EventsStore
from src.tracking.bytetrack import Track
from src.utils.geometry import foot_point, signed_distance


@dataclass
class _TrackState:
    track_id: int
    prev_pos: Optional[Tuple[float, float]] = None
    counted_entry: bool = False
    counted_exit: bool = False
    last_event_time: float = 0.0
    zone: str = "OUTSIDE"
    fsm_state: str = "OUTSIDE"


class CrossingLineEngine:
    """Line-crossing entry/exit engine.

    Drop-in replacement for DoorIntelligenceEngine / EntryExitEngineV2.
    Uses the same ``update(tracks, frame_shape, store) → List[Event]`` interface.
    """

    def __init__(
        self,
        line_norm: Tuple[float, float, float, float] = (0.25, 0.75, 0.75, 0.75),
        entry_direction: str = "upward",
        camera_id: str = "cam_01",
        min_track_frames: int = 3,
        cooldown_sec: float = 2.0,
        min_crossing_gap_sec: float = 1.0,
        min_displacement: float = 0.01,
        dead_zone_px: float = 14.0,
    ) -> None:
        self.line_norm = line_norm
        self.entry_direction = entry_direction
        self.camera_id = camera_id
        self.min_track_frames = min_track_frames
        self.cooldown_sec = cooldown_sec
        self.min_crossing_gap_sec = min_crossing_gap_sec
        self.min_displacement = min_displacement
        self.dead_zone_px = dead_zone_px

        self.counts: Dict[str, int] = {"entry": 0, "exit": 0, "present": 0}
        self._tracks: Dict[Hashable, _TrackState] = {}
        self._line_pixel: Optional[LineString] = None
        self._frame_w: float = 1.0
        self._frame_h: float = 1.0

    def update(
        self,
        tracks: List[Track],
        frame_shape: Tuple[int, int, int],
        store: EventsStore,
    ) -> List[Event]:
        h, w = frame_shape[:2]
        self._frame_w = w
        self._frame_h = h
        self._line_pixel = self._pixel_line(w, h)

        now = time.time()
        produced: List[Event] = []
        alive: set = set()

        for t in tracks:
            # Identity fusion runs before this engine. Keep the state through a
            # ByteTrack ID switch when a stable global identity is available.
            key = t.meta.get("global_id") or t.track_id
            alive.add(key)
            st = self._tracks.get(key)
            if st is None:
                st = _TrackState(track_id=key)
                self._tracks[key] = st

            fp = foot_point(t.xyxy)
            current_pos = (float(fp[0]), float(fp[1]))

            if t.hits < self.min_track_frames:
                st.zone = self._classify_side(current_pos)
                st.fsm_state = "WARMUP"
                t.meta["fsm_state"] = "WARMUP"
                t.meta["zone"] = st.zone
                t.meta["direction"] = "-"
                st.prev_pos = current_pos
                continue

            prev_pos = st.prev_pos
            zone = self._classify_side(current_pos)

            # Dead zone: keep previous zone if on the line
            if zone == "ON_LINE":
                zone = st.zone
            else:
                st.zone = zone

            event = None
            direction_label = "-"

            if prev_pos is not None and self._line_pixel is not None:
                segment = LineString([prev_pos, current_pos])

                if segment.length > 0 and self._line_pixel.intersects(segment):
                    cross_dir = self._crossing_direction(prev_pos, current_pos)

                    if cross_dir == "inward" and not st.counted_entry:
                        direction_label = "inward"
                        if now - st.last_event_time > max(self.min_crossing_gap_sec, self.cooldown_sec):
                            event = self._fire(t, "entry", now, zone)
                            st.counted_entry = True
                            st.zone = "INSIDE"
                            st.last_event_time = now

                    elif cross_dir == "outward" and not st.counted_exit:
                        direction_label = "outward"
                        if now - st.last_event_time > max(self.min_crossing_gap_sec, self.cooldown_sec):
                            event = self._fire(t, "exit", now, zone)
                            st.counted_exit = True
                            st.zone = "OUTSIDE"
                            st.last_event_time = now

            if event is not None:
                event.id = store.insert(event)
                produced.append(event)
                self.counts[event.direction] = self.counts.get(event.direction, 0) + 1
                self.counts["present"] = self.counts.get("present", 0) + (
                    1 if event.direction == "entry" else -1
                )

            st.fsm_state = (
                f"CROSSING_{direction_label.upper()}"
                if direction_label != "-"
                else f"TRACKING_{st.zone}"
            )
            t.meta["fsm_state"] = st.fsm_state
            t.meta["zone"] = st.zone
            t.meta["direction"] = direction_label
            st.prev_pos = current_pos

        for key in list(self._tracks):
            if key not in alive:
                del self._tracks[key]

        return produced

    def set_line(self, line_norm: Tuple[float, float, float, float]) -> None:
        self.line_norm = line_norm

    def _zone_for(self, point) -> str:
        return self._classify_side(point)

    def _pixel_line(self, w: float, h: float) -> LineString:
        x1, y1, x2, y2 = self.line_norm
        return LineString([(x1 * w, y1 * h), (x2 * w, y2 * h)])

    def _classify_side(self, pos: Tuple[float, float]) -> str:
        """Classify a pixel foot point against a pixel-space signed line."""
        px1, py1 = self.line_norm[0] * self._frame_w, self.line_norm[1] * self._frame_h
        px2, py2 = self.line_norm[2] * self._frame_w, self.line_norm[3] * self._frame_h
        dist = signed_distance(pos, ((px1, py1), (px2, py2)))
        # A consistent pixel-space dead zone prevents diagonal-line jitter.
        if abs(dist) < self.dead_zone_px:
            return "ON_LINE"

        if self.entry_direction == "upward":
            return "OUTSIDE" if dist < 0 else "INSIDE"
        elif self.entry_direction == "downward":
            return "INSIDE" if dist > 0 else "OUTSIDE"
        elif self.entry_direction == "rightward":
            return "OUTSIDE" if dist > 0 else "INSIDE"
        elif self.entry_direction == "leftward":
            return "INSIDE" if dist > 0 else "OUTSIDE"
        return "OUTSIDE" if dist < 0 else "INSIDE"

    def _crossing_direction(
        self,
        prev_pos: Tuple[float, float],
        current_pos: Tuple[float, float],
    ) -> str:
        x1, y1, x2, y2 = self.line_norm
        dx_line = x2 - x1
        dy_line = y2 - y1

        if abs(dx_line) > abs(dy_line):
            dy = current_pos[1] - prev_pos[1]
            if self.entry_direction == "upward":
                return "inward" if dy < 0 else "outward"
            elif self.entry_direction == "downward":
                return "inward" if dy > 0 else "outward"
        else:
            dx = current_pos[0] - prev_pos[0]
            if self.entry_direction == "rightward":
                return "inward" if dx > 0 else "outward"
            elif self.entry_direction == "leftward":
                return "inward" if dx < 0 else "outward"

        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        cross = dx_line * dy - dy_line * dx
        if self.entry_direction in ("upward", "leftward"):
            return "inward" if cross < 0 else "outward"
        return "inward" if cross > 0 else "outward"

    def _fire(self, t: Track, direction: str, now: float, zone: str) -> Event:
        person = t.person_name or ""
        return Event(
            date=time.strftime("%Y-%m-%d"),
            time=time.strftime("%H:%M:%S"),
            person=person,
            direction=direction,
            track_id=t.track_id,
            camera_id=self.camera_id,
            confidence=float(t.face_score or t.conf),
            event_id=str(uuid.uuid4()),
            fsm_path=[zone],
        )
