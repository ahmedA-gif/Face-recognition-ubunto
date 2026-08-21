from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.recognition.face_engine import FaceHit
from src.tracking.bytetrack import Track

_GREEN   = (80, 230, 0)      # track / OK
_CYAN    = (255, 230, 0)     # known face
_GOLD    = (0, 200, 255)     # highlight
_AMBER   = (0, 160, 255)     # unknown face
_RED     = (60, 60, 255)     # exit
_WHITE   = (255, 255, 255)
_GREY80  = (40, 40, 40)      # panels

# Door Intelligence region colours (semi-transparent fills)
_ZONE_COLORS = {
    "OUTSIDE": (70, 60, 200),   # BGR-ish
    "DOOR":    (0, 200, 255),
    "INSIDE":  (80, 220, 40),
}

_FONTB = cv2.FONT_HERSHEY_SIMPLEX

_tt = time.strftime


def _alpha_rect(
    canvas: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float = 0.45,
    radius: int = 6,
) -> None:
    """Rounded semi-transparent filled rectangle (region-only, no full-frame copy)."""
    H, W = canvas.shape[:2]
    x1 = max(0, pt1[0]); y1 = max(0, pt1[1])
    x2 = min(W, pt2[0]); y2 = min(H, pt2[1])
    if x2 <= x1 or y2 <= y1:
        return

    region = canvas[y1:y2, x1:x2]
    overlay = region.copy()
    r = radius
    rw, rh = region.shape[1], region.shape[0]
    cv2.rectangle(overlay, (r, 0), (rw - r, rh), color, -1)
    cv2.rectangle(overlay, (0, r), (rw, rh - r), color, -1)
    for cx, cy in [(r, r), (rw - r, r), (r, rh - r), (rw - r, rh - r)]:
        cv2.circle(overlay, (cx, cy), r, color, -1)
    cv2.addWeighted(overlay, alpha, region, 1 - alpha, 0, region)


def _draw_corner_box(
    canvas: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    corner_len: int = 14,
) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    cl = corner_len
    t = thickness
    cv2.line(canvas, (x1, y1), (x1 + cl, y1), color, t)
    cv2.line(canvas, (x1, y1), (x1, y1 + cl), color, t)
    cv2.line(canvas, (x2, y1), (x2 - cl, y1), color, t)
    cv2.line(canvas, (x2, y1), (x2, y1 + cl), color, t)
    cv2.line(canvas, (x1, y2), (x1 + cl, y2), color, t)
    cv2.line(canvas, (x1, y2), (x1, y2 - cl), color, t)
    cv2.line(canvas, (x2, y2), (x2 - cl, y2), color, t)
    cv2.line(canvas, (x2, y2), (x2, y2 - cl), color, t)


def _neon_text(
    canvas: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float,
    color: Tuple[int, int, int],
    thickness: int = 1,
    glow: bool = True,
) -> None:
    if glow:
        dark = tuple(max(0, c // 3) for c in color)
        cv2.putText(canvas, text, org, _FONTB, scale + 0.04, dark, thickness + 3, cv2.LINE_AA)
    cv2.putText(canvas, text, org, _FONTB, scale, color, thickness, cv2.LINE_AA)


class OverlayRenderer:
    """Simple overlay: corner boxes, name labels, top HUD, pulse on new tracks."""

    def __init__(
        self,
        pulse_frames: int = 18,
        hud: bool = True,
        show_boundary: bool = False,
    ) -> None:
        self.pulse_frames = pulse_frames
        self.hud = hud
        self.show_boundary = show_boundary
        self.boundary_line: Dict[str, float] | None = None
        self.boundary_label: str = "BOUNDARY"
        self._pulses: Dict[int, int] = defaultdict(int)
        self._frame_count = 0
        self._boundary_pulse: int = 0
        self._boundary_pulse_frames: int = 20  # frames to show red pulse
        self._zones_px: Dict[str, np.ndarray] = {}  # region -> pixel polygon
        self._zone_pulses: Dict[str, int] = defaultdict(int)  # region -> remaining frames

    def set_boundary(self, line_norm, label: str = "BOUNDARY") -> None:
        if isinstance(line_norm, (list, tuple)) and len(line_norm) == 4:
            self.boundary_line = {"x1": line_norm[0], "y1": line_norm[1], "x2": line_norm[2], "y2": line_norm[3]}
        elif isinstance(line_norm, dict):
            self.boundary_line = dict(line_norm)
        else:
            self.boundary_line = line_norm
        self.boundary_label = label

    def set_zones(self, zones_px: Dict[str, List[Tuple[int, int]]]) -> None:
        """Provide Door Intelligence polygons in pixel coordinates for drawing."""
        self._zones_px = {
            name: np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            for name, poly in zones_px.items()
        }

    def note_new_track(self, track_id: int) -> None:
        self._pulses[track_id] = self.pulse_frames

    def pulse_boundary(self) -> None:
        """Trigger a red pulse on the boundary line (called on entry/exit event)."""
        self._boundary_pulse = self._boundary_pulse_frames

    def pulse_zone(self, zone: str, frames: int = 20) -> None:
        """Trigger a flash pulse on a specific zone polygon (OUTSIDE / DOOR / INSIDE)."""
        self._zone_pulses[zone] = frames

    def draw(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        faces: List[FaceHit],
        counts: Dict[str, int],
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        self._frame_count += 1

        if self._zones_px:
            self._draw_zones(out, w, h)

        for track in tracks:
            self._draw_track(out, track)

        for face in faces:
            self._draw_face(out, face)

        if self.hud:
            self._draw_hud(out, w, h, counts)

        if self.show_boundary and self.boundary_line is not None:
            self._draw_boundary(out, w, h)

        return out

    def _draw_track(self, canvas: np.ndarray, track: Track) -> None:
        x1, y1, x2, y2 = map(int, track.xyxy)
        color = _GREEN
        pulse_left = self._pulses.get(track.track_id, 0)

        if pulse_left > 0:
            _alpha_rect(canvas, (x1, y1), (x2, y2), color, alpha=0.06, radius=3)

        _draw_corner_box(canvas, (x1, y1), (x2, y2), color, thickness=2, corner_len=18)

        label = track.person_name or f"ID:{track.track_id:03d}"
        fsm_state = (track.meta.get("fsm_state") if track.meta else None) or ""
        if fsm_state:
            label = f"{label} · {fsm_state}"
        tw, th = cv2.getTextSize(label, _FONTB, 0.5, 1)[0]
        ly = max(24, y1 - 10)
        _alpha_rect(canvas, (x1 - 2, ly - th - 6), (x1 + tw + 8, ly + 4), _GREY80, alpha=0.8, radius=3)
        cv2.rectangle(canvas, (x1 - 2, ly - th - 6), (x1 + tw + 8, ly + 4), _GREEN, 1)
        _neon_text(canvas, label, (x1 + 3, ly - 1), 0.5, color, glow=True)

        if pulse_left > 0:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            max_r = int(0.55 * max(x2 - x1, y2 - y1))
            r = int(12 + (self.pulse_frames - pulse_left) / self.pulse_frames * max_r)
            alpha_ring = pulse_left / self.pulse_frames
            ring_overlay = canvas.copy()
            cv2.circle(ring_overlay, (cx, cy), r, _CYAN, 1, cv2.LINE_AA)
            cv2.addWeighted(ring_overlay, alpha_ring, canvas, 1 - alpha_ring, 0, canvas)
            self._pulses[track.track_id] = pulse_left - 1

    def _draw_face(self, canvas: np.ndarray, face: FaceHit) -> None:
        x1, y1, x2, y2 = map(int, face.xyxy)
        color = _CYAN if face.name != "Unknown" else _AMBER

        _alpha_rect(canvas, (x1, y1), (x2, y2), color, alpha=0.07, radius=3)
        _draw_corner_box(canvas, (x1, y1), (x2, y2), color, thickness=1, corner_len=10)

        if face.name != "Unknown":
            tag = f"{face.name}  {face.match_score:.2f}"
        else:
            tag = f"Unknown  {face.match_score:.2f}"
        _neon_text(canvas, tag, (x1, max(18, y1 - 6)), 0.44, color, glow=True)

    def _draw_hud(self, canvas: np.ndarray, w: int, h: int, counts: Dict[str, int]) -> None:
        hud_h = 26
        _alpha_rect(canvas, (0, 0), (w, hud_h), _GREY80, alpha=0.75, radius=0)
        cv2.rectangle(canvas, (0, 0), (w, 1), _GREEN, 1)

        _neon_text(canvas, "Person · Face · Events", (12, 18), 0.5, _WHITE, glow=False)
        _neon_text(canvas, _tt("%H:%M:%S"), (w - 92, 18), 0.5, _WHITE, glow=False)

    def _draw_zones(self, canvas: np.ndarray, w: int, h: int) -> None:
        """Draw OUTSIDE / DOOR / INSIDE polygons as translucent filled regions."""
        for name, poly in self._zones_px.items():
            color = _ZONE_COLORS.get(name, _GREY80)
            pulse_left = self._zone_pulses.get(name, 0)
            alpha = 0.22 + (0.35 * pulse_left / 20) if pulse_left > 0 else 0.22
            if pulse_left > 0:
                self._zone_pulses[name] = pulse_left - 1
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [poly], color)
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
            cv2.polylines(canvas, [poly], True, color, 2, cv2.LINE_AA)
            cx = int(poly[:, 0, 0].mean())
            cy = int(poly[:, 0, 1].mean())
            _neon_text(canvas, name, (cx - 10, cy + 4), 0.5, color, glow=True)

    def _draw_boundary(self, canvas: np.ndarray, w: int, h: int) -> None:
        line = self.boundary_line
        if line is None:
            return
        p1 = (int(line["x1"] * w), int(line["y1"] * h))
        p2 = (int(line["x2"] * w), int(line["y2"] * h))
        
        # Red pulse on crossing event
        if self._boundary_pulse > 0:
            color = _RED
            self._boundary_pulse -= 1
            label = "CROSSING!"
        else:
            color = _GREEN
            label = self.boundary_label
        
        cv2.line(canvas, p1, p2, color, 3, cv2.LINE_AA)
        cv2.line(canvas, p1, p2, _CYAN, 1, cv2.LINE_AA)
        # direction arrow at centre
        cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        cv2.arrowedLine(canvas, (cx, cy), (cx + 26, cy), _CYAN, 1, cv2.LINE_AA, tipLength=0.5)
        _neon_text(canvas, label, (cx + 30, cy + 5), 0.4, color, glow=True)
