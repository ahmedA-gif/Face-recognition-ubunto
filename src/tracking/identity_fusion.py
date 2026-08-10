from __future__ import annotations

import json
import os
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
    appearance: Optional[np.ndarray] = None


class IdentityFusionEngine:
    """Resolves fragmented ByteTrack ``track_id``s into persistent ``global_person_id``s.

    Fix 1 — Face-embedding Re-ID: a new track whose face embedding matches an
            existing identity's embedding pool inherits that identity (and all
            its state, history and attendance status).
    Fix 2 — Appearance Re-ID: when no face is visible, a normalized HSV body
            histogram is matched against each identity's appearance pool, so a
            returning person (back turned, too far, mask) keeps the SAME Guest
            id instead of getting a brand-new one.
    Fix 3 — Spatial-temporal stitching: a new track appearing within
            ``max_stitch_dist_px`` px and ``max_stitch_time_sec`` s of where an
            older track expired inherits the expired track's identity.
    Fix 4 — Identity-keyed state: every downstream consumer (entry/exit, events,
            attendance) is keyed on ``global_person_id`` — never raw track_id —
            so 3 fragmented ByteTrack IDs for one person produce exactly ONE
            attendance record.
    Fix 5 — Persistence: guest counter, name map and embedding/appearance pools
            are saved to ``state_path`` so identities survive process restarts
            (Guest#001 is never re-assigned to a different person).
    """

    def __init__(
        self,
        max_stitch_dist_px: float = 60.0,
        max_stitch_time_sec: float = 2.0,
        embedding_match_threshold: float = 0.42,
        appearance_match_threshold: float = 0.85,
        max_pool_embeddings: int = 8,
        max_expired: int = 300,
        state_path: Optional[str] = None,
    ) -> None:
        self.max_stitch_dist_px = max_stitch_dist_px
        self.max_stitch_time_sec = max_stitch_time_sec
        self.embedding_match_threshold = embedding_match_threshold
        self.appearance_match_threshold = appearance_match_threshold
        self.max_pool_embeddings = max_pool_embeddings
        self.max_expired = max_expired
        self.state_path = state_path

        self.track_identity: Dict[int, str] = {}            # track_id -> global_person_id
        self.identity_name: Dict[str, str] = {}             # global_id -> display name
        self.pools: Dict[str, List[np.ndarray]] = {}        # global_id -> [embeddings]
        self.appearance_pools: Dict[str, List[np.ndarray]] = {}  # global_id -> [appearance sigs]
        self._history: Dict[int, deque] = {}                # track_id -> [(x, y, t)]
        self._expired: List[_ExpiredTrack] = []
        self._guest_counter = 0

        self.stitch_count = 0
        self.face_merge_count = 0
        self.appearance_merge_count = 0

        self._load()

    # ── persistence ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.state_path or not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._guest_counter = int(data.get("guest_counter", 0))
            self.identity_name = {k: str(v) for k, v in (data.get("identity_name") or {}).items()}
            self.pools = {
                str(gid): [np.asarray(e, dtype=np.float32) for e in embs]
                for gid, embs in (data.get("pools") or {}).items()
            }
            self.appearance_pools = {
                str(gid): [np.asarray(a, dtype=np.float32) for a in sigs]
                for gid, sigs in (data.get("appearance_pools") or {}).items()
            }
            print(f"[IdentityFusion] restored {len(self.pools)} identity(s) "
                  f"from {self.state_path} (guest_counter={self._guest_counter})")
        except Exception as exc:  # noqa: BLE001
            print(f"[IdentityFusion] WARN could not load state {self.state_path}: {exc}")

    def save(self) -> None:
        if not self.state_path:
            return
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            data = {
                "guest_counter": self._guest_counter,
                "identity_name": self.identity_name,
                "pools": {
                    gid: [e.tolist() for e in embs]
                    for gid, embs in self.pools.items()
                },
                "appearance_pools": {
                    gid: [a.tolist() for a in sigs]
                    for gid, sigs in self.appearance_pools.items()
                },
            }
            tmp = self.state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self.state_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[IdentityFusion] WARN could not save state: {exc}")

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
            "appearance_merges": self.appearance_merge_count,
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
        app = t.meta.get("appearance")
        name = t.person_name if t.person_name and t.person_name != "Unknown" else None

        identity = self.track_identity.get(tid)
        if identity is None:
            identity = self._identity_for_new_track(t, emb, app, name)
        else:
            # A gallery-recognised name is the strongest signal: it always wins
            # over any provisional Guest / pool identity. Otherwise a person's
            # own embedding pooled under Guest# would re-capture them forever.
            if name is not None and name != self.identity_name.get(identity, identity):
                if identity.startswith("Guest#"):
                    self._merge_pool(identity, name)
                    self._merge_appearance_pool(identity, name)
                self.identity_name.setdefault(name, name)
                identity = name
            elif emb is not None and name is None:
                # No gallery name: fall back to embedding Re-ID so fragmented
                # tracks of the same unknown person still stitch together.
                best_id, best_sim = self._best_pool_match(emb)
                if best_id is not None and best_sim >= self.embedding_match_threshold:
                    identity = best_id
            elif app is not None and name is None:
                best_id, best_sim = self._best_appearance_match(app)
                if best_id is not None and best_sim >= self.appearance_match_threshold:
                    identity = best_id

        if emb is not None:
            self._update_pool(identity, emb)
        if app is not None:
            self._update_appearance_pool(identity, app)

        self.track_identity[tid] = identity
        t.meta["global_id"] = identity
        t.person_name = self.identity_name.get(identity, identity)

    def _identity_for_new_track(
        self,
        t: Track,
        emb: Optional[np.ndarray],
        app: Optional[np.ndarray],
        name: Optional[str],
    ) -> str:
        # Gallery-known name is the strongest evidence.
        if name is not None:
            self.identity_name.setdefault(name, name)
            return name

        # Face embedding Re-ID for unknown persons (stitch fragmented tracks).
        if emb is not None:
            best_id, best_sim = self._best_pool_match(emb)
            if best_id is not None and best_sim >= self.embedding_match_threshold:
                self.face_merge_count += 1
                return best_id

        # Appearance Re-ID when the face is hidden (back turned / far away).
        if app is not None:
            best_id, best_sim = self._best_appearance_match(app)
            if best_id is not None and best_sim >= self.appearance_match_threshold:
                self.appearance_merge_count += 1
                return best_id

        # Spatial-temporal stitching
        stitched = self._try_stitch(t)
        if stitched is not None:
            return stitched

        # Fix 4 — brand-new identity (guest until face is recognised)
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

    def _best_appearance_match(self, sig: np.ndarray) -> Tuple[Optional[str], float]:
        best_id, best_sim = None, 0.0
        for gid, pool in self.appearance_pools.items():
            if not pool:
                continue
            s = float((np.asarray(pool) @ sig).max())  # cosine (L2-normalised)
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

    def _update_appearance_pool(self, gid: str, sig: np.ndarray) -> None:
        pool = self.appearance_pools.setdefault(gid, [])
        if pool:
            sims = [float(a @ sig) for a in pool]
            if max(sims) > 0.90:  # near-identical view already captured
                return
        pool.append(sig.copy())
        if len(pool) > self.max_pool_embeddings:
            pool.pop(0)

    def _merge_pool(self, src: str, dst: str) -> None:
        """Re-parent a Guest identity's embeddings onto a gallery name.

        After a Guest is upgraded to a real name, its embedding pool is moved
        to the name's pool so future Re-ID returns the real name (not the old
        Guest). The ``src → dst`` mapping is kept so legacy Guest references
        still resolve to the corrected identity.
        """
        src_pool = self.pools.pop(src, None)
        self.identity_name[src] = dst
        if not src_pool:
            return
        dst_pool = self.pools.setdefault(dst, [])
        for e in src_pool:
            if dst_pool:
                sims = [float(x @ e) for x in dst_pool]
                if max(sims) > 0.85:  # already captured
                    continue
            dst_pool.append(e)
        if len(dst_pool) > self.max_pool_embeddings:
            self.pools[dst] = dst_pool[-self.max_pool_embeddings:]

    def _merge_appearance_pool(self, src: str, dst: str) -> None:
        src_pool = self.appearance_pools.pop(src, None)
        if not src_pool:
            return
        dst_pool = self.appearance_pools.setdefault(dst, [])
        for a in src_pool:
            if dst_pool:
                sims = [float(x @ a) for x in dst_pool]
                if max(sims) > 0.90:
                    continue
            dst_pool.append(a)
        if len(dst_pool) > self.max_pool_embeddings:
            self.appearance_pools[dst] = dst_pool[-self.max_pool_embeddings:]

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
            gid = self.track_identity.get(tid)
            if gid is None:
                # Track vanished before identity resolution — do NOT burn a
                # fresh guest id on a fleeting false positive.
                self._history.pop(tid, None)
                self.track_identity.pop(tid, None)
                continue
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
                        appearance=self.appearance_pools.get(gid, [None])[0] if self.appearance_pools.get(gid) else None,
                    )
                )
            self._history.pop(tid, None)
            self.track_identity.pop(tid, None)
        if len(self._expired) > self.max_expired:
            self._expired[:] = self._expired[-self.max_expired:]

    def close(self) -> None:
        """Persist identity state (guest counter, name map, pools) to disk."""
        self.save()
