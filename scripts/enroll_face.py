#!/usr/bin/env python3
"""Enroll faces from data/faces_gallery/<Name>/*.jpg into SQLite gallery."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recognition.face_engine import FaceEngine
from src.recognition.gallery import FaceGallery
from src.utils.config import load_settings


def main() -> None:
    cfg = load_settings()
    m = cfg["models"]
    ev = cfg["events"]
    gal = cfg["gallery"]

    engine = FaceEngine(root=m["face_root"], pack=m["face_pack"], det_size=tuple(m["face_det_size"]))
    gallery = FaceGallery(db_path=ev["faces_db_path"], match_threshold=m["face_match_threshold"])
    counts = gallery.enroll_folder(gal["images_dir"], engine)
    if not counts:
        print(f"No images found under {gal['images_dir']}")
        print("Create folders like: data/faces_gallery/Ahmed/photo1.jpg")
        return
    for name, n in counts.items():
        print(f"Enrolled {n} face(s) for '{name}'")
    print(f"Gallery DB: {ev['faces_db_path']}")


if __name__ == "__main__":
    main()
