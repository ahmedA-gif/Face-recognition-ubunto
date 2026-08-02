from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.tracking.bytetrack import Track


@dataclass
class _ExpiredTrack:
    track_id: int
    identity: str
    last_centroid: Tuple[float, float]
    exp_time: float
    points: List[Tuple[float, float, float]] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None


class IdentityFusionEngine:
    """Resolves fragmented ByteTrack ``track_id``s into persistent ``global_person_id``s.

    Fix 1 — Face-embedding Re-ID: a new track whose face embedding matches an
            existing identity's embedding pool inherits that identity (and all
            its state, history and attendance status).
    Fix 2 — Spatial-temporal stitching: a new track appearing within
            ``max_stitch_dist_px`` px and ``max_stitch_time_sec`` s of where an
            older track expired inherits the expired track's identity.
    Fix 3 — Identity-keyed state: every downstream consumer (entry/exit, events,
            attendance) is keyed on ``global_person_id`` — never raw track_id —
            so 3 fragmented ByteTrack IDs for one person produce exactly ONE
            attendance record.
    """

    def __init__(
        self,
        max_stitch_dist_px: float = 60.0,
        max_stitch_time_sec: float = 2.0,
        embedding_match_threshold: float = 0.42,
        max_pool_embeddings: int = 8,
        max_expired: int = 300,
    ) -> None:
        self.max_stitch_dist_px = max_stitch_dist_px
        self.max_stitch_time_sec = max_stitch_time_sec
        self.embedding_match_threshold = embedding_match_threshold
        self.max_pool_embeddings = max_pool_embeddings
        self.max_expired = max_expired

        self.track_identity: Dict[int, str] = {}            # track_id -> global_person_id
        self.identity_name: Dict[str, str] = {}             # global_id -> display name
        self.pools: Dict[str, List[np.ndarray]] = {}        # global_id -> [embeddings]
        self._history: Dict[int, deque] = {}                # track_id -> [(x, y, t)]
        self._expired: List[_ExpiredTrack] = []
        self._guest_counter = 0

        self.stitch_count = 0
        self.face_merge_count = 0

    # ── public API ─────────────────────────────────────────────────────────────

    def update(self, tracks: List[Track]) -> None:
        """Record points, archive expired tracks, then resolve identities.

        Expiry happens BEFORE resolution so a new track_id appearing in the
        same frame as the old one's disappearance can be stitched to it.
        """
        now = time.time()
        alive = set()
        for t in tracks:
            alive.add(t.track_id)
            self._record_point(t, now)
        self._expire(alive, now)
        for t in tracks:
            self._resolve(t)

    def global_id_for(self, track_id: int) -> Optional[str]:
        return self.track_identity.get(track_id)

    def stats(self) -> Dict[str, int]:
        return {
            "stitches": self.stitch_count,
            "face_merges": self.face_merge_count,
            "identities": len(self.pools),
            "guests": self._guest_counter,
        }

    # ── internals ──────────────────────────────────────────────────────────────

    def _new_guest(self) -> str:
        self._guest_counter += 1
        gid = f"Guest#{self._guest_counter:03d}"
        self.identity_name[gid] = gid
        return gid

    def _record_point(self, t: Track, now: float) -> None:
        self._history.setdefault(t.track_id, deque(maxlen=120)).append(
            (t.centroid[0], t.centroid[1], now)
        )

    def _resolve(self, t: Track) -> None:
        tid = t.track_id
        emb = t.meta.get("embedding")
        name = t.person_name if t.person_name and t.person_name != "Unknown" else None

        identity = self.track_identity.get(tid)
        if identity is None:
            identity = self._identity_for_new_track(t, emb, name)
        elif emb is not None:
            # Late face evidence corrects provisional identities (e.g. stitches)
            best_id, best_sim = self._best_pool_match(emb)
            if best_id is not None and best_sim >= self.embedding_match_threshold:
                identity = best_id
            elif name is not None and name != self.identity_name.get(identity):
                identity = name

        if emb is not None:
            self._update_pool(identity, emb)

        self.track_identity[tid] = identity
        t.meta["global_id"] = identity
        t.person_name = self.identity_name.get(identity, identity)

    def _identity_for_new_track(
        self,
        t: Track,
        emb: Optional[np.ndarray],
        name: Optional[str],
    ) -> str:
        # Fix 1 — face embedding Re-ID
        if emb is not None:
            best_id, best_sim = self._best_pool_match(emb)
            if best_id is not None and best_sim >= self.embedding_match_threshold:
                self.face_merge_count += 1
                return best_id

        # Gallery-known name is strong evidence
        if name is not None:
            self.identity_name.setdefault(name, name)
            return name

        # Fix 2 — spatial-temporal stitching
        stitched = self._try_stitch(t)
        if stitched is not None:
            return stitched

        # Fix 3 — brand-new identity (guest until face is recognised)
        return self._new_guest()

    def _best_pool_match(self, emb: np.ndarray) -> Tuple[Optional[str], float]:
        best_id, best_sim = None, 0.0
        for gid, pool in self.pools.items():
            if not pool:
                continue
            s = float((np.asarray(pool) @ emb).max())  # cosine (all L2-normalised)
            if s > best_sim:
                best_sim, best_id = s, gid
        return best_id, best_sim

    def _update_pool(self, gid: str, emb: np.ndarray) -> None:
        pool = self.pools.setdefault(gid, [])
        if pool:
            sims = [float(e @ emb) for e in pool]
            if max(sims) > 0.85:  # same view already captured
                return
        pool.append(emb.copy())
        if len(pool) > self.max_pool_embeddings:
            pool.pop(0)

    def _try_stitch(self, t: Track) -> Optional[str]:
        """Stitch to the best-matching expired track.

        Scoring: smallest time gap wins (a track that just vanished is the
        most likely continuation), then smallest distance as tie-break.
        """
        cx, cy = t.centroid
        now = time.time()
        best: Optional[_ExpiredTrack] = None
        best_gap = float("inf")
        best_dist = float("inf")
        for exp in self._expired:
            gap = now - exp.exp_time
            if gap > self.max_stitch_time_sec:
                continue
            ex, ey = exp.last_centroid
            dist = float(((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5)
            if dist > self.max_stitch_dist_px:
                continue
            if gap < best_gap or (gap == best_gap and dist < best_dist):
                best, best_gap, best_dist = exp, gap, dist
        if best is None:
            return None
        self.stitch_count += 1
        return best.identity

    def _expire(self, alive_ids: set, now: float) -> None:
        for tid, pts in list(self._history.items()):
            if tid in alive_ids:
                continue
            gid = self.track_identity.get(tid) or self._new_guest()
            if pts:
                last = pts[-1]
                self._expired.append(
                    _ExpiredTrack(
                        track_id=tid,
                        identity=gid,
                        last_centroid=(last[0], last[1]),
                        exp_time=now,
                        points=list(pts),
                        embedding=self.pools.get(gid, [None])[0] if self.pools.get(gid) else None,
                    )
                )
            self._history.pop(tid, None)
            self.track_identity.pop(tid, None)
        if len(self._expired) > self.max_expired:
            self._expired[:] = self._expired[-self.max_expired:]
