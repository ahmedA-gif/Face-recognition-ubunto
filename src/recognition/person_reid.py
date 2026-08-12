from __future__ import annotations

from contextlib import nullcontext
from typing import Any, List, Optional, Tuple, Union

import numpy as np


class PersonReIDEngine:
    """Optional person-Re-ID embedding extractor via a scripted PyTorch model.

    Wraps a ``torch.jit.load``-able model (``torch.jit.script`` / ``.trace``
    export of an OSNet-style re-ID network). The embedding is clothing-invariant,
    so the identity fusion engine prefers it over the HSV body-histogram when no
    face is visible — a person changing clothes between frames still keeps the
    same ``global_person_id``.

    The engine is fully optional: if no weights are configured, or torch is not
    usable, the pipeline simply keeps the HSV + strict-threshold fallback.

    ``extract(frame, xyxy)`` returns an L2-normalized 1-D ``np.float32``
    embedding, or ``None`` on any failure (invalid crop, model error, etc.) so
    callers can silently fall back.

    For testability the model and the numpy→tensor bridge are injectable; the
    only torch-touching step is ``_to_tensor`` / ``_forward``.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        weights: Optional[str] = None,
        *,
        model: Any = None,
        device: str = "cpu",
        input_size: Union[int, Tuple[int, int]] = (256, 128),
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
        min_crop_px: int = 16,
    ) -> None:
        if model is None:
            if not weights:
                raise ValueError("person-ReID requires either 'weights' or a 'model'")
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - torch optional
                raise RuntimeError(
                    "person-ReID needs torch; install it or remove person_reid_weights"
                ) from exc
            self._torch = torch
            self.model = torch.jit.load(weights, map_location=device)
            self.model.eval()
        else:
            self._torch = None
            self.model = model

        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = tuple(input_size)  # (H, W) like torchreid convention
        self.mean = mean or self.IMAGENET_MEAN
        self.std = std or self.IMAGENET_STD
        self.min_crop_px = min_crop_px

    # ── public API ─────────────────────────────────────────────────────────────

    def extract(self, frame: np.ndarray, xyxy) -> Optional[np.ndarray]:
        """Return an L2-normalized person embedding for the bbox, or None."""
        try:
            crop = self._crop(frame, xyxy)
            if crop is None:
                return None
            blob = self._preprocess(crop)
            feat = self._forward(self._to_tensor(blob))
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            if hasattr(feat, "detach"):  # torch.Tensor
                feat = feat.detach().cpu().numpy()
            v = np.asarray(feat, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(v))
            if not np.isfinite(norm) or norm < 1e-6:
                return None
            return (v / norm).astype(np.float32)
        except Exception:  # noqa: BLE001 — never break the pipeline on a bad crop
            return None

    # ── preprocessing ───────────────────────────────────────────────────────────

    def _crop(self, frame: np.ndarray, xyxy) -> Optional[np.ndarray]:
        if frame is None or xyxy is None:
            return None
        h, w = frame.shape[:2]
        x1 = max(0, int(round(xyxy[0]))); y1 = max(0, int(round(xyxy[1])))
        x2 = min(w, int(round(xyxy[2]))); y2 = min(h, int(round(xyxy[3])))
        if x2 - x1 < self.min_crop_px or y2 - y1 < self.min_crop_px:
            return None
        return frame[y1:y2, x1:x2]

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Letterbox to ``(H, W)`` preserving aspect, then ImageNet-normalize.

        Returns a float32 CHW blob ready for the model.
        """
        import cv2

        th, tw = self.input_size
        h, w = crop.shape[:2]
        scale = min(tw / w, th / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)  # letterbox padding
        dx, dy = (tw - nw) // 2, (th - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        for c in range(3):
            rgb[..., c] = (rgb[..., c] - self.mean[c]) / self.std[c]
        return rgb.transpose(2, 0, 1)

    def _to_tensor(self, blob: np.ndarray):
        if self._torch is None:  # injected fake model path (tests)
            return blob[None]
        return self._torch.from_numpy(blob)[None]

    def _forward(self, x):
        ctx = self._torch.no_grad() if self._torch is not None else nullcontext()
        with ctx:
            return self.model(x)
