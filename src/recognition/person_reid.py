from __future__ import annotations

import os
from typing import Optional

import numpy as np


class PersonReID:
    """Optional person-ReID wrapper. Loads a scripted PyTorch model if available.

    The model is expected to accept a preprocessed RGB tensor of shape
    (1,3,H,W) and return a 1-D embedding tensor. If no model / torch is
    available this class is a noop and extract() returns None.
    """

    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu") -> None:
        self.enabled = False
        self.device = device
        self.model = None
        if weights_path is None:
            return
        try:
            import torch
        except Exception:
            return
        if not os.path.exists(weights_path):
            return
        try:
            # Prefer a scripted model for simple loading
            self.model = torch.jit.load(weights_path, map_location=device)
            self.model.to(device)
            self.model.eval()
            self.torch = torch
            self.enabled = True
        except Exception:
            # Could not load as scripted; try regular state_dict-based load
            try:
                # Provide a minimal fallback: assume the file is a state_dict for a model
                # The caller is responsible for providing a compatible scripted model.
                return
            except Exception:
                return

    def extract(self, frame: np.ndarray, xyxy) -> Optional[np.ndarray]:
        if not self.enabled or frame is None or xyxy is None:
            return None
        x1, y1, x2, y2 = map(int, xyxy)
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = frame[y1:y2, x1:x2]
        try:
            # Preprocess to 128x256 (W x H) common for person-ReID
            import cv2
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            inp = cv2.resize(crop_rgb, (128, 256), interpolation=cv2.INTER_LINEAR)
        except Exception:
            return None
        try:
            t = self.torch
            arr = t.from_numpy(inp.astype('float32') / 255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            # Normalize with ImageNet stats (reasonable default)
            mean = t.tensor([0.485, 0.456, 0.406], device=self.device).view(1, -1, 1, 1)
            std = t.tensor([0.229, 0.224, 0.225], device=self.device).view(1, -1, 1, 1)
            arr = (arr - mean) / std
            with t.no_grad():
                out = self.model(arr)
            if isinstance(out, tuple) or isinstance(out, list):
                out = out[0]
            emb = out.squeeze().cpu().numpy().astype('float32')
            norm = np.linalg.norm(emb)
            if norm < 1e-6:
                return None
            return (emb / norm).astype('float32')
        except Exception:
            return None
