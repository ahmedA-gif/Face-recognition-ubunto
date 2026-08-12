"""IdentityFusion must let a gallery-recognised name override Guest identities.

Regression test for the bug where a person's own embedding pooled under a
Guest# identity re-captured them forever, so every event was logged as Guest
even after face recognition matched the gallery.
"""

from __future__ import annotations

import numpy as np

from src.tracking.bytetrack import Track
from src.tracking.identity_fusion import IdentityFusionEngine


def _track(
    tid: int,
    emb: np.ndarray = None,
    name: str = "",
    hits: int = 5,
    reid: np.ndarray = None,
    appearance: np.ndarray = None,
) -> Track:
    t = Track(track_id=tid, xyxy=np.array([10, 10, 30, 60], dtype=np.float64), conf=0.8, hits=hits)
    if emb is not None:
        t.meta["embedding"] = emb
    if reid is not None:
        t.meta["person_reid"] = reid
    if appearance is not None:
        t.meta["appearance"] = appearance
    t.person_name = name
    return t


def _unit(dim: int = 8, seed: int = 0) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
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


# ── person-ReID (clothing-invariant) fallback ────────────────────────────────


def test_reid_merges_tracks_when_face_hidden():
    fusion = IdentityFusionEngine(reid_match_threshold=0.82)
    r1 = _unit(32, seed=7)
    # Face hidden: only the person-ReID embedding is available.
    fusion.update([_track(1, reid=r1)])
    assert fusion.global_id_for(1) == "Guest#001"
    assert len(fusion.reid_pools["Guest#001"]) == 1

    # Fragmented track of the SAME person (no face, different clothing) must
    # keep Guest#001 via the clothing-invariant reid embedding.
    fusion.update([_track(2, reid=_unit(32, seed=7))])  # same reid vector
    assert fusion.global_id_for(2) == "Guest#001"
    assert fusion.reid_merge_count >= 1


def test_reid_preferred_over_hsv_appearance_clothing_change():
    # Two tracks of one person wearing DIFFERENT clothes (HSV signatures far
    # apart) but with the same clothing-invariant reid embedding. Without reid
    # this would spawn Guest#002; with reid it must stay Guest#001.
    fusion = IdentityFusionEngine(reid_match_threshold=0.82, appearance_match_threshold=0.85)
    r1 = _unit(32, seed=7)
    a_blue = _unit(16, seed=1)
    a_red = _unit(16, seed=2)  # orthogonal to a_blue -> would fail HSV match

    fusion.update([_track(1, reid=r1, appearance=a_blue)])
    assert fusion.global_id_for(1) == "Guest#001"

    fusion.update([_track(2, reid=_unit(32, seed=7), appearance=a_red)])
    assert fusion.global_id_for(2) == "Guest#001", "reid must beat the HSV mismatch"
    assert fusion.appearance_merge_count == 0, "HSV never merged (correctly rejected)"


def test_face_embedding_still_wins_over_reid():
    fusion = IdentityFusionEngine(embedding_match_threshold=0.42, reid_match_threshold=0.82)
    e1 = _unit(32, seed=3)
    r1 = _unit(16, seed=7)
    r2 = _unit(16, seed=9)  # does NOT match r1

    # 1) Track with face + reid -> Guest#001.
    fusion.update([_track(1, e1, reid=r1)])
    assert fusion.global_id_for(1) == "Guest#001"

    # 2) Same face but a reid vector matching NOTHING -> face Re-ID must win.
    fusion.update([_track(2, e1, reid=r2)])
    assert fusion.global_id_for(2) == "Guest#001"
    assert fusion.face_merge_count >= 1


def test_reid_pools_persist_across_restart(tmp_path):
    state = str(tmp_path / "identity_state.json")
    r1 = _unit(32, seed=7)

    f1 = IdentityFusionEngine(reid_match_threshold=0.82, state_path=state)
    f1.update([_track(1, reid=r1)])
    f1.close()

    f2 = IdentityFusionEngine(reid_match_threshold=0.82, state_path=state)
    assert "Guest#001" in f2.reid_pools
    # Restored pool must still merge the same person after a restart.
    f2.update([_track(2, reid=_unit(32, seed=7))])
    assert f2.global_id_for(2) == "Guest#001"
    assert f2.reid_merge_count >= 1


def test_hsv_appearance_fallback_still_works_without_reid():
    fusion = IdentityFusionEngine(appearance_match_threshold=0.85)
    a1 = _unit(16, seed=1)
    fusion.update([_track(1, appearance=a1)])
    assert fusion.global_id_for(1) == "Guest#001"
    fusion.update([_track(2, appearance=_unit(16, seed=1))])
    assert fusion.global_id_for(2) == "Guest#001"
    assert fusion.appearance_merge_count >= 1


def test_reid_below_threshold_does_not_merge():
    # Spatial stitching disabled so only reid can merge these tracks.
    fusion = IdentityFusionEngine(
        reid_match_threshold=0.99, max_stitch_dist_px=0, max_stitch_time_sec=0
    )
    r1 = _unit(32, seed=7)
    fusion.update([_track(1, reid=r1)])
    assert fusion.global_id_for(1) == "Guest#001"
    # A different person (orthogonal reid) must get their own guest id.
    fusion.update([_track(2, reid=_unit(32, seed=42))])
    assert fusion.global_id_for(2) == "Guest#002"
    assert fusion.reid_merge_count == 0
