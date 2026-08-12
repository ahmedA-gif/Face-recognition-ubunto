"""Unit tests for the optional person-ReID (torchscript) embedding engine.

The numpy→torch bridge is the only torch-touching code; these tests exercise
the full ``extract()`` path with an injected fake model so they run even where
torch is absent or its numpy interop is broken.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.recognition.person_reid import PersonReIDEngine


class _FakeModel:
    """Mimics a scripted OSNet: maps a (1,3,H,W) blob to a (1,3) feature."""

    def __call__(self, x):
        return x.mean(axis=(2, 3))


def _frame(h: int = 100, w: int = 80, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _engine(**kwargs) -> PersonReIDEngine:
    kwargs.setdefault("model", _FakeModel())
    kwargs.setdefault("input_size", (64, 32))
    return PersonReIDEngine(**kwargs)


def test_constructor_requires_weights_or_model():
    with pytest.raises(ValueError):
        PersonReIDEngine()


def test_missing_weights_file_raises_cleanly():
    pytest.importorskip("torch")
    with pytest.raises((FileNotFoundError, RuntimeError, OSError, ValueError)):
        PersonReIDEngine(weights="/nonexistent/reid.pt")


def test_extract_returns_l2_normalized_embedding():
    engine = _engine()
    emb = engine.extract(_frame(), np.array([10, 15, 60, 75], dtype=float))
    assert emb is not None
    assert emb.dtype == np.float32
    assert emb.ndim == 1
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5


def test_extract_none_on_invalid_crop():
    engine = _engine(min_crop_px=16)
    frame = _frame()
    assert engine.extract(frame, None) is None
    assert engine.extract(None, np.array([0, 0, 10, 10], dtype=float)) is None
    # bbox smaller than min_crop_px
    assert engine.extract(frame, np.array([0, 0, 5, 5], dtype=float)) is None
    # inverted bbox
    assert engine.extract(frame, np.array([50, 50, 10, 10], dtype=float)) is None


def test_extract_none_on_model_error():
    class _Boom:
        def __call__(self, x):
            raise RuntimeError("model exploded")

    engine = _engine(model=_Boom())
    assert engine.extract(_frame(), np.array([10, 15, 60, 75], dtype=float)) is None


def test_extract_none_on_nonfinite_embedding():
    class _NaN:
        def __call__(self, x):
            return np.full((1, 3), np.nan, dtype=np.float32)

    engine = _engine(model=_NaN())
    assert engine.extract(_frame(), np.array([10, 15, 60, 75], dtype=float)) is None


def test_preprocess_letterbox_shape_and_norm():
    engine = _engine()
    crop = _frame(h=90, w=30, seed=1)  # very tall crop, aspect must be preserved
    blob = engine._preprocess(crop)
    assert blob.shape == (3, 64, 32)
    assert blob.dtype == np.float32
    # ImageNet-normalized: pixels live in roughly [-2.5, 2.5], not [0, 1]
    assert float(np.abs(blob).max()) > 1.0


def test_same_crop_same_embedding_different_crop_different():
    engine = _engine()
    f = _frame()
    box = np.array([10, 15, 60, 75], dtype=float)
    e1 = engine.extract(f, box)
    e2 = engine.extract(f, box)
    assert e1 is not None and e2 is not None
    assert float(np.dot(e1, e2)) > 0.999
    e3 = engine.extract(np.full_like(f, 40, dtype=np.uint8), box)
    assert e3 is not None and float(np.dot(e1, e3)) < 1.0


def test_int_input_size_expands_to_square():
    engine = PersonReIDEngine(model=_FakeModel(), input_size=64)
    assert engine.input_size == (64, 64)
