from __future__ import annotations

from typing import List

import numpy as np

from src.recognition.face_engine import FaceHit
from src.tracking.bytetrack import Track


def _center(xyxy: np.ndarray) -> tuple[float, float]:
    return float((xyxy[0] + xyxy[2]) / 2), float((xyxy[1] + xyxy[3]) / 2)


def face_inside_person(face: np.ndarray, person: np.ndarray, margin: float = 0.05) -> bool:
    fx, fy = _center(face)
    x1, y1, x2, y2 = person
    w, h = x2 - x1, y2 - y1
    return (x1 - margin * w) <= fx <= (x2 + margin * w) and (y1 - margin * h) <= fy <= (y2 + margin * h)


def face_to_head_distance(face_box: np.ndarray, person_box: np.ndarray) -> float:
    """Distance from a face centre to the expected upper-body/head centre."""
    fx, fy = _center(face_box)
    x1, y1, x2, y2 = map(float, person_box)
    head_x = (x1 + x2) / 2.0
    head_y = y1 + (y2 - y1) * 0.20
    return float(((fx - head_x) ** 2 + (fy - head_y) ** 2) ** 0.5)


def appearance_signature(frame: np.ndarray, xyxy, n_bins: int = 16) -> np.ndarray | None:
    """Compact body-colour signature for appearance-based re-identification.

    Crops the person bbox from ``frame``, builds a normalized HSV histogram
    (H+S jointly), L2-normalised. Used as a FALLBACK when no face embedding is
    available so a returning person (e.g. back turned) still keeps Guest#001.
    Returns None on invalid crop so callers skip matching.
    """
    try:
        import cv2
    except ImportError:
        return None
    if frame is None or xyxy is None:
        return None
    h, w = frame.shape[:2]
    x1 = max(0, int(xyxy[0])); y1 = max(0, int(xyxy[1]))
    x2 = min(w, int(xyxy[2])); y2 = min(h, int(xyxy[3]))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = frame[y1:y2, x1:x2]
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    except cv2.error:
        return None
    hist = cv2.calcHist([hsv], [0, 1], None, [n_bins, n_bins], [0, 180, 0, 256])
    hist = hist.ravel().astype(np.float32)
    norm = float(np.linalg.norm(hist))
    if norm < 1e-6:
        return None
    return (hist / norm).astype(np.float32)


def attach_faces_to_tracks(
    tracks: List[Track],
    faces: List[FaceHit],
    gallery_match_fn,
) -> None:
    """Match faces to person tracks and set person_name / face_score.

    The best overlapping face's embedding is always stored on the track
    (``track.meta["embedding"]``) — even for Unknown faces — so the identity
    fusion engine can build embedding pools for re-identification.
    """
    for face in faces:
        name, score = gallery_match_fn(face.embedding)
        face.name = name
        face.match_score = score
        # attach to best overlapping track
        best: Track | None = None
        best_distance = float("inf")
        for t in tracks:
            if not face_inside_person(face.xyxy, t.xyxy):
                continue
            distance = face_to_head_distance(face.xyxy, t.xyxy)
            if distance < best_distance:
                best_distance = distance
                best = t
        if best is not None:
            best.meta["embedding"] = face.embedding
            if name != "Unknown" and score >= best.face_score:
                best.person_name = name
                best.face_score = score
