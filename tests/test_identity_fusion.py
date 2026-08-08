"""IdentityFusion must let a gallery-recognised name override Guest identities.

Regression test for the bug where a person's own embedding pooled under a
Guest# identity re-captured them forever, so every event was logged as Guest
even after face recognition matched the gallery.
"""

from __future__ import annotations

import numpy as np

from src.tracking.bytetrack import Track
from src.tracking.identity_fusion import IdentityFusionEngine


def _track(tid: int, emb: np.ndarray, name: str = "", hits: int = 5) -> Track:
    t = Track(track_id=tid, xyxy=np.array([10, 10, 30, 60], dtype=np.float64), conf=0.8, hits=hits)
    if emb is not None:
        t.meta["embedding"] = emb
    t.person_name = name
    return t


def _unit(dim: int = 8) -> np.ndarray:
    v = np.random.RandomState(dim).randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def test_gallery_name_upgrades_guest_identity():
    fusion = IdentityFusionEngine(embedding_match_threshold=0.42)
    e1 = _unit(1)

    # 1) Person enters; gallery fails (Unknown) -> Guest identity, embedding pooled.
    fusion.update([_track(1, e1)])
    t1 = _track(1, e1)
    assert fusion.global_id_for(1) == "Guest#001"

    # 2) Same track, same face, but now gallery recognises the name.
    fusion.update([_track(1, e1, name="Haseeb")])
    assert fusion.global_id_for(1) == "Haseeb"
    assert fusion.identity_name.get("Haseeb") == "Haseeb"


def test_new_track_with_known_name_never_becomes_guest():
    fusion = IdentityFusionEngine(embedding_match_threshold=0.42)
    e1 = _unit(2)
    fusion.update([_track(1, e1, name="Haseeb")])
    assert fusion.global_id_for(1) == "Haseeb"
    # A second fragmented track of the same person with the same face name:
    # pool Re-ID returns the name, not a Guest.
    e2 = _unit(2)  # same view -> near-identical embedding
    fusion.update([_track(2, e2, name="Haseeb")])
    assert fusion.global_id_for(2) == "Haseeb"
    assert fusion.face_merge_count >= 1 or fusion.track_identity[2] == "Haseeb"


def test_stitch_unknown_tracks_together_without_name():
    fusion = IdentityFusionEngine(embedding_match_threshold=0.42, max_stitch_dist_px=500, max_stitch_time_sec=60)
    e1 = _unit(3)
    fusion.update([_track(1, e1)])  # Guest#001
    # Track 1 expires; track 2 (same spot, same embedding) stitches to it.
    fusion.update([_track(2, e1)])
    assert fusion.global_id_for(2) == "Guest#001"
