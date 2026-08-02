#!/usr/bin/env python3
"""
check_models.py — Verify that all required model files are present.

Run first before anything else:
    python scripts/check_models.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── colour helpers ─────────────────────────────────────────────────────────────
_G  = "\033[92m"   # green
_R  = "\033[91m"   # red
_Y  = "\033[93m"   # yellow
_B  = "\033[94m"   # blue
_C  = "\033[96m"   # cyan
_W  = "\033[97m"   # white
_DIM = "\033[2m"
_RESET = "\033[0m"
_BOLD  = "\033[1m"

def ok(msg):  print(f"  {_G}✔{_RESET}  {msg}")
def err(msg): print(f"  {_R}✘{_RESET}  {_R}{msg}{_RESET}")
def warn(msg):print(f"  {_Y}⚠{_RESET}  {_Y}{msg}{_RESET}")
def info(msg):print(f"  {_C}ℹ{_RESET}  {msg}")

BANNER = f"""
{_C}{_BOLD}╔══════════════════════════════════════════════════╗
║      Person · Face · Events — Model Checker      ║
╚══════════════════════════════════════════════════╝{_RESET}
"""

# ── expected files ──────────────────────────────────────────────────────────────
YOLO_FILES = [
    ROOT / "models/yolo/yolo11n.pt",
    ROOT / "models/yolo/yolo11n.onnx",
]

BUFFALO_L_FILES = [
    ROOT / "models/face/models/buffalo_l/det_10g.onnx",
    ROOT / "models/face/models/buffalo_l/w600k_r50.onnx",
    ROOT / "models/face/models/buffalo_l/genderage.onnx",
    ROOT / "models/face/models/buffalo_l/2d106det.onnx",
    ROOT / "models/face/models/buffalo_l/1k3d68.onnx",
]

BUFFALO_L_ZIP = ROOT / "models/face/models/buffalo_l.zip"

TRACKER_FILES = [
    ROOT / "models/tracker/bytetrack.yaml",
]


def check_section(title: str, files: list[Path]) -> bool:
    print(f"\n{_B}{_BOLD}[ {title} ]{_RESET}")
    all_ok = True
    for p in files:
        rel = p.relative_to(ROOT)
        if p.exists():
            size_mb = p.stat().st_size / 1_048_576
            ok(f"{rel}  {_DIM}({size_mb:.1f} MB){_RESET}")
        else:
            err(f"MISSING: {rel}")
            all_ok = False
    return all_ok


def check_buffalo_l_zip():
    """Auto-extract buffalo_l.zip if the folder is incomplete."""
    needed = BUFFALO_L_FILES
    missing = [f for f in needed if not f.exists()]
    if not missing:
        return True

    print(f"\n  {_Y}buffalo_l folder incomplete — checking for zip…{_RESET}")
    if BUFFALO_L_ZIP.exists():
        warn(f"Found {BUFFALO_L_ZIP.name} ({BUFFALO_L_ZIP.stat().st_size/1_048_576:.0f} MB). Extracting…")
        dest = BUFFALO_L_ZIP.parent
        try:
            with zipfile.ZipFile(BUFFALO_L_ZIP, "r") as zf:
                zf.extractall(dest)
            info(f"Extracted to {dest}")
            still_missing = [f for f in needed if not f.exists()]
            if not still_missing:
                ok("All buffalo_l files extracted successfully.")
                return True
            else:
                for f in still_missing:
                    err(f"Still missing after extract: {f.relative_to(ROOT)}")
                return False
        except Exception as exc:
            err(f"Extraction failed: {exc}")
            return False
    else:
        err("buffalo_l.zip not found either.")
        info("Download it with InsightFace:")
        info("  python -c \"import insightface; insightface.model_zoo.get_model('buffalo_l')\"")
        return False


def check_python_deps() -> bool:
    print(f"\n{_B}{_BOLD}[ Python dependencies ]{_RESET}")
    all_ok = True
    deps = {
        "numpy":        "numpy",
        "cv2":          "opencv-python",
        "yaml":         "PyYAML",
        "ultralytics":  "ultralytics",
        "onnxruntime":  "onnxruntime",
        "insightface":  "insightface",
        "scipy":        "scipy",
        "filterpy":     "filterpy",
    }
    for mod, pkg in deps.items():
        try:
            __import__(mod)
            ok(f"{pkg}")
        except ImportError:
            err(f"{pkg}  ← NOT installed  →  pip install {pkg}")
            all_ok = False

    # FAISS special check
    try:
        import faiss  # type: ignore  # noqa: F401
        ok("faiss-cpu")
    except ImportError:
        warn("faiss-cpu  ← not installed  →  pip install faiss-cpu")
        info("Without faiss-cpu the gallery will fall back to NumPy (slower for large galleries).")
    except Exception as exc:
        warn(f"faiss import error: {exc}")
        info("Try: pip uninstall faiss faiss-gpu faiss-cpu && pip install faiss-cpu")

    return all_ok


def check_data_dirs() -> None:
    print(f"\n{_B}{_BOLD}[ Data directories ]{_RESET}")
    dirs = [
        ROOT / "data/db",
        ROOT / "data/snapshots",
        ROOT / "data/faces_gallery",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        ok(f"{d.relative_to(ROOT)}  (ready)")


def main() -> None:
    print(BANNER)
    all_ok = True

    all_ok &= check_section("YOLO models", YOLO_FILES)

    print(f"\n{_B}{_BOLD}[ InsightFace buffalo_l (ArcFace R50) ]{_RESET}")
    missing_bf = [f for f in BUFFALO_L_FILES if not f.exists()]
    if missing_bf:
        ok_bf = check_buffalo_l_zip()
    else:
        ok_bf = True

    for p in BUFFALO_L_FILES:
        rel = p.relative_to(ROOT)
        if p.exists():
            size_mb = p.stat().st_size / 1_048_576
            ok(f"{rel}  {_DIM}({size_mb:.1f} MB){_RESET}")
        else:
            err(f"MISSING: {rel}")
            ok_bf = False
    all_ok &= ok_bf

    all_ok &= check_section("Tracker config", TRACKER_FILES)
    check_data_dirs()
    check_python_deps()

    # Gallery status
    gallery_dir = ROOT / "data/faces_gallery"
    people = [p.name for p in gallery_dir.iterdir() if p.is_dir()] if gallery_dir.exists() else []
    print(f"\n{_B}{_BOLD}[ Face gallery ]{_RESET}")
    if people:
        ok(f"Enrolled people: {', '.join(people)}")
        info("Run  python scripts/enroll_faces.py  to re-enroll after adding new photos.")
    else:
        warn("No person folders found in data/faces_gallery/")
        info("Create:  data/faces_gallery/<YourName>/photo1.jpg  and run enroll_faces.py")

    # Summary
    print(f"\n{'─'*52}")
    if all_ok:
        print(f"{_G}{_BOLD}  ✔  All critical models present. Ready to run!{_RESET}")
        print(f"\n  Next steps:")
        print(f"  {_W}1.{_RESET} python scripts/enroll_faces.py     ← enroll your face gallery")
        print(f"  {_W}2.{_RESET} python scripts/run_video.py --source path/to/video.mp4")
        print(f"  {_W}3.{_RESET} python main.py                      ← live CCTV (RTSP)")
    else:
        print(f"{_R}{_BOLD}  ✘  Some models are missing — fix the errors above first.{_RESET}")
    print()


if __name__ == "__main__":
    main()
