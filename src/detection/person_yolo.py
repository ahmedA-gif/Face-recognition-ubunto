from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    xyxy: np.ndarray  # (4,) x1,y1,x2,y2
    conf: float
    cls_id: int = 0


def _letterbox(
    img: np.ndarray,
    target: Tuple[int, int],
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, float, float]:
    h, w = img.shape[:2]
    th, tw = target
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dx = (tw - nw) // 2
    dy = (th - nh) // 2
    canvas = np.full((th, tw, 3), color, dtype=np.uint8)
    canvas[dy : dy + nh, dx : dx + nw] = resized
    return canvas, scale, dx, dy


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float,
) -> List[int]:
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-9)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return keep


class PersonDetector:
    """YOLO nano person-only detector (CPU via ONNX Runtime)."""

    def __init__(
        self,
        weights: str,
        conf: float = 0.45,
        iou: float = 0.5,
        imgsz: int = 416,
        device: str = "cpu",
        person_class_id: int = 0,
    ) -> None:
        import onnxruntime as ort

        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.person_class_id = person_class_id
        self.session = ort.InferenceSession(weights, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta()
        inp = self.session.get_inputs()[0]
        self._inp_name = inp.name
        self._inp_shape = inp.shape  # e.g. [1, 3, H, W] or [1, 3, -1, -1]
        self._num_classes = 80 if "v8" in meta.description.lower() or "yolo" in meta.description.lower() else 80

    def detect(self, frame: np.ndarray) -> List[Detection]:
        letterboxed, scale, dx, dy = _letterbox(frame, (self.imgsz, self.imgsz))
        blob = letterboxed.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, :, :, :]

        outputs = self.session.run(None, {self._inp_name: blob})
        pred = outputs[0][0]

        num_classes = pred.shape[0] - 4
        boxes = pred[:4, :]
        scores = pred[4:, :]

        cls_ids = scores.argmax(axis=0)
        confs = scores[cls_ids, np.arange(scores.shape[1])]

        mask = (cls_ids == self.person_class_id) & (confs >= self.conf)
        if not mask.any():
            return []

        boxes = boxes[:, mask]
        confs = confs[mask]
        cls_ids = cls_ids[mask]

        cx, cy, w, h = boxes
        x1 = (cx - w / 2 - dx) / scale
        y1 = (cy - h / 2 - dy) / scale
        x2 = (cx + w / 2 - dx) / scale
        y2 = (cy + h / 2 - dy) / scale

        stacked = np.stack([x1, y1, x2, y2], axis=1)
        keep = _nms(stacked, confs, self.iou)

        out: List[Detection] = []
        for i in keep:
            out.append(
                Detection(
                    xyxy=stacked[i].astype(float),
                    conf=float(confs[i]),
                    cls_id=int(cls_ids[i]),
                )
            )
        return out
