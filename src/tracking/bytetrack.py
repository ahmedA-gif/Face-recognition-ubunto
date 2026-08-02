from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.detection.person_yolo import Detection


@dataclass
class Track:
    track_id: int
    xyxy: np.ndarray
    conf: float
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    person_name: str = ""
    face_score: float = 0.0
    prev_side: Optional[str] = None
    last_event_at: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def centroid(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (float((x1 + x2) / 2), float((y1 + y2) / 2))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


class ByteTracker:
    """
    Lightweight ByteTrack-style tracker for CPU (IoU association).
    Good enough for door entry/exit without heavy BoT-SORT deps.
    For full ByteTrack, see notebooks/02_download_bytetrack.ipynb.
    """

    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_thresh: float = 0.3,
        track_buffer: int = 30,
    ) -> None:
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, detections: List[Detection]) -> List[Track]:
        high = [d for d in detections if d.conf >= self.track_high_thresh]
        low = [d for d in detections if self.track_low_thresh <= d.conf < self.track_high_thresh]

        for t in self.tracks:
            t.age += 1
            t.time_since_update += 1

        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_dets = list(range(len(high)))
        matches: List[tuple[int, int]] = []

        # Greedy IoU match (high conf)
        pairs = []
        for ti in unmatched_tracks:
            for di in unmatched_dets:
                pairs.append((_iou(self.tracks[ti].xyxy, high[di].xyxy), ti, di))
        pairs.sort(reverse=True, key=lambda x: x[0])
        used_t, used_d = set(), set()
        for score, ti, di in pairs:
            if score < self.match_thresh:
                break
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            matches.append((ti, di))

        for ti, di in matches:
            t = self.tracks[ti]
            d = high[di]
            t.xyxy = d.xyxy
            t.conf = d.conf
            t.hits += 1
            t.time_since_update = 0

        unmatched_tracks = [i for i in unmatched_tracks if i not in used_t]
        unmatched_dets = [i for i in unmatched_dets if i not in used_d]

        # Second association with low-conf dets
        pairs = []
        for ti in unmatched_tracks:
            for di, d in enumerate(low):
                pairs.append((_iou(self.tracks[ti].xyxy, d.xyxy), ti, di))
        pairs.sort(reverse=True, key=lambda x: x[0])
        used_t2, used_d2 = set(), set()
        for score, ti, di in pairs:
            if score < self.match_thresh:
                break
            if ti in used_t2 or di in used_d2:
                continue
            used_t2.add(ti)
            used_d2.add(di)
            t = self.tracks[ti]
            d = low[di]
            t.xyxy = d.xyxy
            t.conf = d.conf
            t.hits += 1
            t.time_since_update = 0

        unmatched_tracks = [i for i in unmatched_tracks if i not in used_t2]

        # New tracks from remaining high dets
        for di in unmatched_dets:
            d = high[di]
            if d.conf < self.new_track_thresh:
                continue
            self.tracks.append(
                Track(
                    track_id=self._next_id,
                    xyxy=d.xyxy.copy(),
                    conf=d.conf,
                )
            )
            self._next_id += 1

        # Remove stale
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.track_buffer]
        return [t for t in self.tracks if t.time_since_update == 0]
