#!/usr/bin/env python3
"""
run_video.py — Run the full pipeline on an uploaded video file (or webcam).

Usage examples
--------------
# Test on a video file (shows live window + saves output):
    python scripts/run_video.py --source data/test_video.mp4

# Headless (no display), save annotated output:
    python scripts/run_video.py --source data/test_video.mp4 --no-display --output data/output.mp4

# Process only first 300 frames (quick sanity check):
    python scripts/run_video.py --source data/test_video.mp4 --max-frames 300

# Webcam:
    python scripts/run_video.py --source 0

# Live CCTV (override settings.yaml):
    python scripts/run_video.py --source "rtsp://admin:admin1234@192.168.2.112:554/cam/realmonitor?channel=1&subtype=0"

Controls
--------
  Q / ESC  →  quit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_settings
from src.pipeline.runner import run_pipeline

# ── colour helpers ─────────────────────────────────────────────────────────────
_G = "\033[92m"; _R = "\033[91m"; _Y = "\033[93m"
_C = "\033[96m"; _W = "\033[97m"; _B = "\033[94m"
_RESET = "\033[0m"; _BOLD = "\033[1m"; _DIM = "\033[2m"

BANNER = f"""
{_C}{_BOLD}╔══════════════════════════════════════════════════╗
║   Person · Face · Events  — Video / Live Test    ║
╚══════════════════════════════════════════════════╝{_RESET}
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the person/face pipeline on a video file or live CCTV stream.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--source", "-s",
        default=None,
        help=(
            "Video file path  |  webcam index (0,1,…)  |  RTSP URL.\n"
            "If omitted, uses camera.source from config/settings.yaml."
        ),
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Save annotated video to this path (e.g. data/output.mp4). Optional.",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the live OpenCV window (useful for headless/SSH runs).",
    )
    p.add_argument(
        "--max-frames", "-n",
        type=int,
        default=None,
        help="Stop after N frames (quick sanity check).",
    )
    p.add_argument(
        "--skip-frames",
        type=int,
        default=None,
        help="Run YOLO detection every N frames (overrides settings.yaml).",
    )
    p.add_argument(
        "--face-every-n",
        type=int,
        default=None,
        help="Run face recognition every N frames (overrides settings.yaml).",
    )
    p.add_argument(
        "--face-pack",
        choices=["buffalo_l", "buffalo_s"],
        default=None,
        help="InsightFace model pack. Default: buffalo_l (better accuracy).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Face match cosine-similarity threshold (0.0–1.0, default 0.42).",
    )
    p.add_argument(
        "--config", "-c",
        default=None,
        help="Alternate settings.yaml path.",
    )
    p.add_argument(
        "--enroll-first",
        action="store_true",
        help="Run face enrollment from data/faces_gallery/ before starting pipeline.",
    )
    return p.parse_args()


def maybe_enroll(cfg: dict) -> None:
    """Run gallery enrollment inline if --enroll-first is passed."""
    from src.recognition.face_engine import FaceEngine
    from src.recognition.gallery import FaceGallery

    m = cfg["models"]; ev = cfg["events"]; gal = cfg["gallery"]
    print(f"\n{_Y}► Enrolling faces from {gal['images_dir']} …{_RESET}")
    engine  = FaceEngine(root=m["face_root"], pack=m["face_pack"], det_size=tuple(m["face_det_size"]))
    gallery = FaceGallery(db_path=ev["faces_db_path"], match_threshold=m["face_match_threshold"])
    counts  = gallery.enroll_folder(gal["images_dir"], engine)
    if not counts:
        print(f"  {_Y}⚠  No images found — add photos to data/faces_gallery/<Name>/{_RESET}")
    else:
        for name, n in counts.items():
            print(f"  {_G}✔{_RESET}  Enrolled {n} embedding(s) for '{_W}{name}{_RESET}'")
    print()


def main() -> None:
    args = parse_args()
    print(BANNER)

    cfg = load_settings(args.config)

    # Apply CLI overrides to config
    if args.face_pack:
        cfg["models"]["face_pack"] = args.face_pack
    if args.threshold:
        cfg["models"]["face_match_threshold"] = args.threshold

    # Resolve source
    source = args.source
    if source is None:
        source = cfg["camera"]["source"]
        print(f"  {_DIM}No --source given, using config: {source}{_RESET}")

    # Show what we are about to do
    is_file = isinstance(source, str) and Path(source).is_file()
    is_rtsp  = isinstance(source, str) and source.lower().startswith("rtsp")
    is_cam   = str(source).isdigit() if isinstance(source, str) else isinstance(source, int)

    src_type = (
        f"{_G}Video file{_RESET}  {_DIM}{source}{_RESET}"   if is_file else
        f"{_C}Live RTSP{_RESET}   {_DIM}{source}{_RESET}"   if is_rtsp else
        f"{_B}Webcam{_RESET}      index={source}"
    )
    print(f"  Source  : {src_type}")
    print(f"  Pack    : {_W}{cfg['models']['face_pack']}{_RESET}")
    print(f"  Threshold: {cfg['models']['face_match_threshold']}")
    if args.output:
        print(f"  Output  : {_W}{args.output}{_RESET}")
    if args.max_frames:
        print(f"  Max frames: {args.max_frames}")
    print()

    if args.enroll_first:
        maybe_enroll(cfg)

    # Gallery status
    from src.recognition.gallery import FaceGallery
    ev = cfg["events"]
    m  = cfg["models"]
    gallery = FaceGallery(db_path=ev["faces_db_path"], match_threshold=m["face_match_threshold"])
    print(f"  {_C}Gallery: {gallery.status()}{_RESET}")
    if gallery.count() == 0:
        print(f"  {_Y}⚠  Gallery is empty — faces will show as 'Unknown'.{_RESET}")
        print(f"  {_DIM}  Add photos to data/faces_gallery/<Name>/ and run:{_RESET}")
        print(f"  {_DIM}  python scripts/enroll_faces.py{_RESET}")
    print()

    # Controls reminder
    if not args.no_display:
        print(f"  {_DIM}Controls: Q or ESC to quit the preview window.{_RESET}\n")

    # Live sources (RTSP/webcam) run forever with auto-restart; video files run once.
    is_live = is_rtsp or is_cam
    t0 = time.perf_counter()
    while True:
        try:
            result = run_pipeline(
                cfg,
                source_override=source,
                output_path=args.output,
                display=not args.no_display,
                max_frames=args.max_frames,
                skip_frames=args.skip_frames,
                face_every_n=args.face_every_n,
            )
            elapsed = time.perf_counter() - t0
            fps_avg = result.frames_processed / elapsed if elapsed > 0 else 0

            print(f"\n{'─'*52}")
            print(f"{_G}{_BOLD}  Pipeline finished{_RESET}")
            print(f"  Frames processed : {_W}{result.frames_processed}{_RESET}")
            print(f"  Elapsed time     : {_W}{elapsed:.1f}s{_RESET}  ({_W}{fps_avg:.1f} fps{_RESET})")
            print(f"  Events triggered : {_W}{result.events_count}{_RESET}")
            if result.output_path:
                print(f"  Saved output     : {_G}{result.output_path}{_RESET}")
            print()
            if not is_live or args.max_frames is not None:
                break
            print(f"{_Y}  Stream ended — reconnecting in 5s (24/7 mode)…{_RESET}\n")
            time.sleep(5)
            t0 = time.perf_counter()
        except KeyboardInterrupt:
            print(f"\n{_Y}  Interrupted by user.{_RESET}")
            sys.exit(0)


if __name__ == "__main__":
    main()
