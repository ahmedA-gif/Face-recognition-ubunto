"""Trajectory smoothing: constant-velocity Kalman filter + history hooks.

Velocity is taken from the filter state (not raw frame deltas) so detector
jitter is removed before the Door Intelligence Engine computes motion
direction. Coordinates are pixel-space foot points.

Units: when ``t`` timestamps are supplied, velocity is px/second (adaptive
dt); when the caller omits timestamps (tests / synthetic feeds) dt=1.0 and
velocity is px/frame.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Hashable, List, Optional, Tuple

import numpy as np

from filterpy.kalman import KalmanFilter

Velocity = Tuple[float, float]
HistoryPoint = Tuple[float, float, float]  # (t, x, y)


class TrackMotion:
    """Per-track Kalman smoother + position/zone history buffers."""

    def __init__(
        self,
        process_noise: float = 0.05,
        measure_noise: float = 4.0,
        history_len: int = 90,
        min_speed: float = 0.6,
        max_dt: float = 5.0,
    ) -> None:
        self.process_noise = float(process_noise)
        self.measure_noise = float(measure_noise)
        self.history_len = int(history_len)
        self.min_speed = float(min_speed)
        self.max_dt = float(max_dt)

        self._kf: dict[Hashable, KalmanFilter] = {}
        self._hist: dict[Hashable, Deque[HistoryPoint]] = {}
        self._last_t: dict[Hashable, float] = {}

    def _make_filter(self, x: float, y: float) -> KalmanFilter:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0
        kf.F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
        )
        kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        kf.P = np.eye(4) * 20.0
        kf.R = np.eye(2) * self.measure_noise
        kf.Q = np.eye(4) * self.process_noise
        kf.Q[2, 2] *= 10.0
        kf.Q[3, 3] *= 10.0
        kf.x = np.array([[x], [y], [0.0], [0.0]], dtype=np.float64)
        return kf

    def update(self, key: Hashable, x: float, y: float, t: Optional[float] = None) -> Velocity:
        """Predict+update the filter for ``key``; return smoothed velocity (vx, vy)."""
        kf = self._kf.get(key)
        if kf is None:
            kf = self._make_filter(x, y)
            self._kf[key] = kf

        prev_t = self._last_t.get(key)
        dt = 1.0
        if t is not None:
            if prev_t is not None and t > prev_t:
                dt = float(np.clip(t - prev_t, 0.01, self.max_dt))
            self._last_t[key] = t

        kf.F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
        )
        kf.predict()
        kf.update(np.array([[x], [y]], dtype=np.float64))

        self._hist.setdefault(key, deque(maxlen=self.history_len)).append(
            (t if t is not None else 0.0, float(x), float(y))
        )
        return float(kf.x[2, 0]), float(kf.x[3, 0])

    def velocity(self, key: Hashable) -> Velocity:
        kf = self._kf.get(key)
        if kf is None:
            return 0.0, 0.0
        return float(kf.x[2, 0]), float(kf.x[3, 0])

    def speed(self, key: Hashable) -> float:
        vx, vy = self.velocity(key)
        return float((vx * vx + vy * vy) ** 0.5)

    def history(self, key: Hashable) -> List[HistoryPoint]:
        return list(self._hist.get(key, deque()))

    def forget(self, key: Hashable) -> None:
        self._kf.pop(key, None)
        self._hist.pop(key, None)
        self._last_t.pop(key, None)

    def reset(self) -> None:
        self._kf.clear()
        self._hist.clear()
        self._last_t.clear()
