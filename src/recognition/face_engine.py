from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class FaceHit:
    xyxy: np.ndarray        # [x1, y1, x2, y2]
    embedding: np.ndarray   # 512-dim unit vector
    det_score: float
    name: str = "Unknown"
    match_score: float = 0.0


class FaceEngine:
    """InsightFace SCRFD + ArcFace-R50 on CPU.

    Default pack:  ``buffalo_l``  (higher accuracy, ~330 MB)
    Fast pack:     ``buffalo_s``  (small, ~4 MB, less accurate)

    The models are read from  ``<root>/models/<pack>/``.  Point *root* at
    ``models/face/`` (project-relative) and make sure you have already
    downloaded + unzipped the pack there.

    buffalo_l files needed (models/face/models/buffalo_l/):
        det_10g.onnx   – SCRFD-10G face detector
        w600k_r50.onnx – ArcFace R50 recognition
        genderage.onnx – optional gender/age
        2d106det.onnx  – 2D landmark (106 pts)
        1k3d68.onnx    – 3D landmark (optional)
    """

    def __init__(
        self,
        root: str,
        pack: str = "buffalo_l",
        det_size: Tuple[int, int] = (640, 640),
        providers: Optional[List[str]] = None,
    ) -> None:
        from insightface.app import FaceAnalysis

        # Always force CPU — never try CUDA on a CPU-only box
        _providers = providers or ["CPUExecutionProvider"]

        self.app = FaceAnalysis(
            name=pack,
            root=root,
            providers=_providers,
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=-1, det_size=det_size)
        self._pack = pack
        print(f"[FaceEngine] loaded pack='{pack}'  providers={_providers}")

    def detect_and_embed(
        self,
        frame_bgr: np.ndarray,
        min_face_px: int = 40,
    ) -> List[FaceHit]:
        """Detect faces and return normalised 512-dim ArcFace embeddings."""
        faces = self.app.get(frame_bgr)
        hits: List[FaceHit] = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(float)
            w, h = x2 - x1, y2 - y1
            if min(w, h) < min_face_px:
                continue
            # normed_embedding is already L2-normalised by InsightFace
            emb = f.normed_embedding.astype(np.float32)
            hits.append(
                FaceHit(
                    xyxy=np.array([x1, y1, x2, y2], dtype=float),
                    embedding=emb,
                    det_score=float(f.det_score),
                )
            )
        return hits
