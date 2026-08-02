#!/usr/bin/env python3
"""Main CPU pipeline: capture/video → person → track → face → entry/exit events → overlay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_settings
from src.pipeline.runner import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the person/face events pipeline.")
    parser.add_argument(
        "--source",
        default=None,
        help="Video file path, webcam index, or RTSP/go2rtc URL. Defaults to config/settings.yaml.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the annotated output video.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the live OpenCV window. Useful for notebooks and headless runs.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after processing this many frames.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=None,
        help="Override config/settings.yaml pipeline.skip_frames.",
    )
    parser.add_argument(
        "--face-every-n",
        type=int,
        default=None,
        help="Override config/settings.yaml pipeline.face_every_n.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional alternate settings.yaml path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_settings(args.config)
    result = run_pipeline(
        cfg,
        source_override=args.source,
        output_path=args.output,
        display=not args.no_display,
        max_frames=args.max_frames,
        skip_frames=args.skip_frames,
        face_every_n=args.face_every_n,
    )
    print(
        f"Done. frames={result.frames_processed}, events={result.events_count}, "
        f"source={result.source}, output={result.output_path or 'n/a'}"
    )


if __name__ == "__main__":
    main()
