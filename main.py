#!/usr/bin/env python3
"""Main CPU pipeline: capture/video → person → track → face → entry/exit events → overlay."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SITE = Path(__file__).resolve().parent / ".venv" / "Lib" / "site-packages"
_ORT_CAPI = _SITE / "onnxruntime" / "capi"
for _sub in ("nvidia\\cudnn\\bin", "nvidia\\cublas\\bin", "nvidia\\cuda_runtime\\bin", "nvidia\\cufft\\bin"):
    _d = _SITE / _sub
    if _d.is_dir():
        os.add_dll_directory(str(_d))
        os.environ["PATH"] = str(_d) + ";" + os.environ.get("PATH", "")
if _ORT_CAPI.is_dir():
    os.add_dll_directory(str(_ORT_CAPI))
    os.environ["PATH"] = str(_ORT_CAPI) + ";" + os.environ.get("PATH", "")

from src.utils.config import load_settings
from src.pipeline.runner import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the person/face events pipeline.")
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the Flask web dashboard (http://localhost:5000).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Web dashboard port (default: 5000).",
    )
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

    if args.web:
        from web.app import app, socketio
        print(f"[Web] Starting VisionAttend AI dashboard at http://localhost:{args.port}")
        socketio.run(app, host="0.0.0.0", port=args.port, debug=False, allow_unsafe_werkzeug=True)
        return

    cfg = load_settings(args.config)
    while True:
        try:
            result = run_pipeline(
                cfg,
                source_override=args.source,
                output_path=args.output,
                display=not args.no_display,
                max_frames=args.max_frames,
                skip_frames=args.skip_frames,
                face_every_n=args.face_every_n,
            )
        except KeyboardInterrupt:
            break
        print(
            f"Done. frames={result.frames_processed}, events={result.events_count}, "
            f"source={result.source}, output={result.output_path or 'n/a'}"
        )
        if args.max_frames is not None:
            break
        print("[Main] stream ended — restarting in 5s (24/7 mode) ...")
        time.sleep(5)


if __name__ == "__main__":
    main()
