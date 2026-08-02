#!/usr/bin/env python3
"""
enroll_faces.py — Enroll face photos into the SQLite gallery (FAISS-CPU indexed).

Folder layout expected:
    data/faces_gallery/
        Ahmed/
            photo1.jpg
            photo2.png
        Sara/
            headshot.jpg

Usage:
    python scripts/enroll_faces.py
    python scripts/enroll_faces.py --clear          # wipe DB first, then re-enroll
    python scripts/enroll_faces.py --list           # just list enrolled people
    python scripts/enroll_faces.py --name Ahmed --image path/to/photo.jpg  # single enroll
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recognition.face_engine import FaceEngine
from src.recognition.gallery import FaceGallery
from src.utils.config import load_settings

import cv2

# ── colour helpers ─────────────────────────────────────────────────────────────
_G = "\033[92m"; _R = "\033[91m"; _Y = "\033[93m"
_C = "\033[96m"; _W = "\033[97m"; _B = "\033[94m"
_RESET = "\033[0m"; _BOLD = "\033[1m"; _DIM = "\033[2m"

BANNER = f"""
{_C}{_BOLD}╔══════════════════════════════════════════════════╗
║      Person · Face · Events  — Face Enroller     ║
╚══════════════════════════════════════════════════╝{_RESET}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enroll face photos into the face gallery DB.")
    p.add_argument("--clear",  action="store_true", help="Wipe the gallery DB before enrolling.")
    p.add_argument("--list",   action="store_true", help="List enrolled people and exit.")
    p.add_argument("--name",   default=None, help="Name for single-image enrollment.")
    p.add_argument("--image",  default=None, help="Path to a single image to enroll with --name.")
    p.add_argument("--config", default=None, help="Alternate settings.yaml path.")
    return p.parse_args()


def enroll_single(engine: FaceEngine, gallery: FaceGallery, name: str, image_path: str) -> None:
    img = cv2.imread(image_path)
    if img is None:
        print(f"  {_R}✘  Cannot read image: {image_path}{_RESET}")
        return
    hits = engine.detect_and_embed(img, min_face_px=20)
    if not hits:
        print(f"  {_Y}⚠  No face detected in {image_path}{_RESET}")
        return
    # largest face
    hit = max(hits, key=lambda h: (h.xyxy[2] - h.xyxy[0]) * (h.xyxy[3] - h.xyxy[1]))
    gallery.add(name, hit.embedding)
    print(f"  {_G}✔{_RESET}  Enrolled '{_W}{name}{_RESET}' from {Path(image_path).name}")


def main() -> None:
    args = parse_args()
    print(BANNER)

    cfg = load_settings(args.config)
    m   = cfg["models"]
    ev  = cfg["events"]
    gal = cfg["gallery"]

    gallery = FaceGallery(
        db_path=ev["faces_db_path"],
        match_threshold=m["face_match_threshold"],
        backend="faiss",
    )

    # ── list mode ────────────────────────────────────────────────────────────
    if args.list:
        people = gallery.list_people()
        total  = gallery.count()
        print(f"  {_C}Gallery: {gallery.status()}{_RESET}\n")
        if not people:
            print(f"  {_Y}⚠  Gallery is empty.{_RESET}")
        else:
            print(f"  {_W}Enrolled people ({len(people)}):{_RESET}")
            for name in people:
                print(f"    {_G}·{_RESET}  {name}")
            print(f"\n  Total embeddings: {_W}{total}{_RESET}")
        return

    # ── clear ────────────────────────────────────────────────────────────────
    if args.clear:
        db = Path(ev["faces_db_path"])
        faiss_f = db.with_suffix(".faiss")
        if db.exists():
            db.unlink()
            print(f"  {_Y}Wiped {db.name}{_RESET}")
        if faiss_f.exists():
            faiss_f.unlink()
            print(f"  {_Y}Wiped {faiss_f.name}{_RESET}")
        # Recreate fresh gallery
        gallery = FaceGallery(
            db_path=ev["faces_db_path"],
            match_threshold=m["face_match_threshold"],
            backend="faiss",
        )

    # ── load FaceEngine ───────────────────────────────────────────────────────
    print(f"  {_DIM}Loading InsightFace '{m['face_pack']}' …{_RESET}")
    engine = FaceEngine(
        root=m["face_root"],
        pack=m["face_pack"],
        det_size=tuple(m["face_det_size"]),
    )

    # ── single image ──────────────────────────────────────────────────────────
    if args.name and args.image:
        enroll_single(engine, gallery, args.name, args.image)
        print(f"\n  {_C}{gallery.status()}{_RESET}\n")
        return

    # ── bulk folder enroll ────────────────────────────────────────────────────
    gallery_dir = Path(gal["images_dir"])
    print(f"\n  {_W}Scanning {gallery_dir} …{_RESET}\n")
    if not gallery_dir.exists():
        print(f"  {_R}✘  Directory not found: {gallery_dir}{_RESET}")
        print(f"  Create it and add sub-folders per person:\n")
        print(f"  {_DIM}data/faces_gallery/")
        print(f"      Ahmed/   photo1.jpg")
        print(f"      Sara/    headshot.jpg{_RESET}\n")
        sys.exit(1)

    person_dirs = sorted(p for p in gallery_dir.iterdir() if p.is_dir())
    if not person_dirs:
        print(f"  {_Y}⚠  No person sub-folders found in {gallery_dir}{_RESET}")
        print(f"  {_DIM}Create:  data/faces_gallery/<YourName>/photo.jpg{_RESET}\n")
        sys.exit(0)

    total_enrolled = 0
    for person_dir in person_dirs:
        name = person_dir.name
        images = [
            p for p in person_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
        if not images:
            print(f"  {_Y}⚠  No images in {person_dir.name}/{_RESET}")
            continue

        n = 0
        failed = 0
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                failed += 1
                continue
            hits = engine.detect_and_embed(img, min_face_px=20)
            if not hits:
                print(f"    {_Y}·{_RESET}  {img_path.name}  {_DIM}(no face detected){_RESET}")
                failed += 1
                continue
            hit = max(hits, key=lambda h: (h.xyxy[2] - h.xyxy[0]) * (h.xyxy[3] - h.xyxy[1]))
            gallery.add(name, hit.embedding)
            n += 1
            print(f"    {_G}✔{_RESET}  {img_path.name}")

        status = f"{_G}{n} enrolled{_RESET}"
        if failed:
            status += f"  {_Y}{failed} skipped{_RESET}"
        print(f"  {_W}{name}{_RESET}: {status}")
        total_enrolled += n

    # Final status
    print(f"\n{'─'*52}")
    print(f"  Total embeddings enrolled : {_W}{total_enrolled}{_RESET}")
    print(f"  {_C}{gallery.status()}{_RESET}")
    print(f"\n  {_G}Ready! Run the pipeline:{_RESET}")
    print(f"  {_W}python scripts/run_video.py --source path/to/video.mp4{_RESET}")
    print()


if __name__ == "__main__":
    main()
