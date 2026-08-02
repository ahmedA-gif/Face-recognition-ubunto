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
        best_area = 0.0
        for t in tracks:
            if not face_inside_person(face.xyxy, t.xyxy):
                continue
            area = max(0.0, t.xyxy[2] - t.xyxy[0]) * max(0.0, t.xyxy[3] - t.xyxy[1])
            if area > best_area:
                best_area = area
                best = t
        if best is not None:
            best.meta["embedding"] = face.embedding
            if name != "Unknown" and score >= best.face_score:
                best.person_name = name
                best.face_score = score
