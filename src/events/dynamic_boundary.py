from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.geometry import which_side


class DynamicBoundaryEngine:
    """Auto-learns the optimal entry/exit boundary from trajectory flow.

    - Collects per-identity centroid paths and direction vectors.
    - K-means (k=2) clusters the movement vectors to find the dominant
      walking corridor.
    - Synthesizes the boundary as the perpendicular bisector of the dominant
      flow vector through the flow centroid.
    - Entry direction is derived self-consistently: the majority flow
      direction becomes the "entry" direction.
    - EMA smoothing keeps the line stable once learned.

    Falls back to the seed (manually configured) line until enough tracks have
    been observed (``min_tracks_for_learning``).
    """

    def __init__(
        self,
        min_tracks_for_learning: int = 150,
        adaptation_rate: float = 0.05,
        smoothing: bool = True,
        seed_line: Optional[Dict[str, float]] = None,
        min_speed_norm: float = 0.01,
        window: int = 20,
        recompute_every: int = 15,
    ) -> None:
        self.min_tracks = max(5, int(min_tracks_for_learning))
        self.alpha = float(adaptation_rate)
        self.smoothing = bool(smoothing)
        self.min_speed_norm = float(min_speed_norm)
        self.window = int(window)
        self.recompute_every = max(1, int(recompute_every))

        self.seed_line = dict(seed_line or {
            "x1": 0.45, "y1": 0.10, "x2": 0.45, "y2": 0.90,
        })
        self.line_norm: Dict[str, float] = dict(self.seed_line)
        self.entry_direction: str = "A_to_B"   # set once learned
        self.learned: bool = False
        self.confidence: float = 0.0

        self._hist: Dict[str, deque] = {}                 # global_id -> [(x, y)]
        self._vectors: List[Tuple[float, float]] = []     # unit flow vectors
        self._points: List[Tuple[float, float]] = []      # trajectory centroids
        self._pending = 0
        self._calibration_logged = 0

    # ── public API ─────────────────────────────────────────────────────────────

    def feed(self, global_id: str, x: float, y: float) -> None:
        """Feed one centroid sample for an identity.

        Coordinates must be NORMALISED (0–1 relative to frame width/height).
        """
        h = self._hist.setdefault(global_id, deque(maxlen=self.window))
        h.append((x, y))
        if len(h) < 3:
            return
        first = h[0]
        dx, dy = x - first[0], y - first[1]
        speed = float((dx * dx + dy * dy) ** 0.5)
        if speed < self.min_speed_norm:
            return
        self._vectors.append((dx / speed, dy / speed))
        self._points.append((x, y))
        self._pending += 1
        if self._pending >= self.recompute_every and (self.learned or len(self._vectors) >= self.min_tracks):
            self._recompute()

    def progress(self) -> Tuple[int, int]:
        return len(self._vectors), self.min_tracks

    def forget(self, global_id: str) -> None:
        self._hist.pop(global_id, None)

    # ── internals ──────────────────────────────────────────────────────────────

    def _recompute(self) -> None:
        self._pending = 0
        if len(self._vectors) < self.min_tracks:
            self.confidence = len(self._vectors) / self.min_tracks
            return

        v_major, count_major = self._kmeans2(self._vectors)
        if v_major is None:
            return

        pts = np.asarray(self._points, dtype=np.float64)
        p_center = pts.mean(axis=0)

        norm = float(np.linalg.norm(v_major))
        if norm < 1e-9:
            return
        v = v_major / norm
        perp = np.array([-v[1], v[0]])

        half = 0.55
        p1 = p_center - perp * half
        p2 = p_center + perp * half
        p1 = np.clip(p1, 0.02, 0.98)
        p2 = np.clip(p2, 0.02, 0.98)

        line = {"x1": float(p1[0]), "y1": float(p1[1]), "x2": float(p2[0]), "y2": float(p2[1])}

        # Majority flow direction => entry direction (self-consistent)
        a = p_center - v * 0.15
        b = p_center + v * 0.15
        s_a = which_side((float(a[0]), float(a[1])), ((line["x1"], line["y1"]), (line["x2"], line["y2"])))
        s_b = which_side((float(b[0]), float(b[1])), ((line["x1"], line["y1"]), (line["x2"], line["y2"])))
        transition = f"{s_a}_to_{s_b}"
        if s_a == s_b:
            transition = "A_to_B"

        conf = min(1.0, len(self._vectors) / self.min_tracks)

        # EMA smoothing once learned
        if self.smoothing and self.learned:
            for k in ("x1", "y1", "x2", "y2"):
                self.line_norm[k] = (1 - self.alpha) * self.line_norm[k] + self.alpha * line[k]
        else:
            self.line_norm = line

        self.entry_direction = transition
        self.confidence = conf
        self.learned = True

    def _kmeans2(
        self,
        vectors: List[Tuple[float, float]],
    ) -> Tuple[Optional[np.ndarray], int]:
        """Tiny k-means (k=2) on unit vectors; returns (majority centroid, count)."""
        v = np.asarray(vectors, dtype=np.float64)
        pos = v[:, 0] > 0
        if pos.any():
            c1 = v[pos].mean(axis=0)
        else:
            c1 = v.mean(axis=0)
        if (~pos).any():
            c2 = v[~pos].mean(axis=0)
        else:
            c2 = v.mean(axis=0)

        for _ in range(12):
            d1 = v @ c1
            d2 = v @ c2
            m1 = d1 >= d2
            m2 = ~m1
            if m1.any():
                c1 = v[m1].mean(axis=0)
            if m2.any():
                c2 = v[m2].mean(axis=0)

        n1 = int(np.sum(d1 >= d2))
        n2 = len(v) - n1
        if n1 == n2 == 0:
            return None, 0
        if n1 >= n2:
            return c1, n1
        return c2, n2
