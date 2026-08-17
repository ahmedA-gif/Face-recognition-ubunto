"""Calibrate the entry/exit boundary: capture a live frame with the current
line drawn so you can see where it sits relative to the real door.

Usage:
    python scripts/calibrate_boundary.py [y1] [y2] [x1] [x2]
    python scripts/calibrate_boundary.py --source rtsp://127.0.0.1:8554/cam_01_sub --y1 0.55 --y2 0.55 --x1 0.2 --x2 0.8
    python scripts/calibrate_boundary.py 0.55 0.55 0.2 0.8

Prints the on-screen door position so you can dial in the exact coordinates.
Supports both positional arguments and named flags.
"""
import argparse
import sys
import os
from pathlib import Path

import cv2

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_settings


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate entry/exit boundary line and visualize on camera frame"
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Video source (default: from config/settings.yaml)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to settings.yaml (default: config/settings.yaml)"
    )
    parser.add_argument(
        "--y1",
        type=float,
        default=None,
        help="Normalized y1 coordinate (0-1)"
    )
    parser.add_argument(
        "--y2",
        type=float,
        default=None,
        help="Normalized y2 coordinate (0-1)"
    )
    parser.add_argument(
        "--x1",
        type=float,
        default=None,
        help="Normalized x1 coordinate (0-1)"
    )
    parser.add_argument(
        "--x2",
        type=float,
        default=None,
        help="Normalized x2 coordinate (0-1)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path (default: data/calibration_line.png)"
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=15,
        help="Frame index to capture (default: 15)"
    )
    parser.add_argument(
        "--buffer-threshold",
        type=float,
        default=10.0,
        help="Buffer zone threshold in pixels for visualization"
    )
    
    # Support positional arguments for backward compatibility
    args, unknown = parser.parse_known_args()
    
    # Handle positional arguments
    positional = []
    for arg in unknown:
        if arg.startswith('-'):
            break
        try:
            positional.append(float(arg))
        except ValueError:
            pass
    
    if len(positional) >= 2:
        if args.y1 is None:
            args.y1 = positional[0]
        if args.y2 is None:
            args.y2 = positional[1]
    if len(positional) >= 4:
        if args.x1 is None:
            args.x1 = positional[2]
        if args.x2 is None:
            args.x2 = positional[3]
    
    # Load configuration
    cfg = load_settings(args.config)
    source = args.source or cfg["camera"]["source"]
    line = dict(cfg["entry_exit"]["line"])
    
    # Override line coordinates if provided
    if args.y1 is not None:
        line["y1"] = args.y1
    if args.y2 is not None:
        line["y2"] = args.y2
    if args.x1 is not None:
        line["x1"] = args.x1
    if args.x2 is not None:
        line["x2"] = args.x2
    
    # Capture frame
    cap = cv2.VideoCapture(str(source) if not str(source).isdigit() else int(source))
    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {source}")
        sys.exit(1)
    
    for _ in range(args.frame_index + 1):
        ok, frame = cap.read()
        if not ok:
            print(f"ERROR: Could not read frame {args.frame_index} from {source}")
            cap.release()
            sys.exit(1)
    cap.release()
    
    H, W = frame.shape[:2]
    
    # Convert normalized coordinates to pixel coordinates
    x1, y1 = int(line["x1"] * W), int(line["y1"] * H)
    x2, y2 = int(line["x2"] * W), int(line["y2"] * H)
    
    # Draw the boundary line
    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    # Draw buffer zone visualization
    buffer_px = args.buffer_threshold
    if buffer_px > 0:
        # Calculate perpendicular direction
        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2)**0.5
        if length > 0:
            perp_x = -dy / length * buffer_px
            perp_y = dx / length * buffer_px
            
            # Draw buffer zone lines
            cv2.line(frame, 
                     (int(x1 + perp_x), int(y1 + perp_y)), 
                     (int(x2 + perp_x), int(y2 + perp_y)), 
                     (0, 255, 0), 1)
            cv2.line(frame, 
                     (int(x1 - perp_x), int(y1 - perp_y)), 
                     (int(x2 - perp_x), int(y2 - perp_y)), 
                     (0, 255, 0), 1)
    
    # Add text information
    info_text = f"y={line['y1']:.3f}  x={line['x1']:.3f}-{line['x2']:.3f}"
    cv2.putText(frame, info_text, (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add help text
    help_text = "OUTSIDE (above) | DOOR (on line) | INSIDE (below)"
    cv2.putText(frame, help_text, (10, H - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Add coordinate help
    coord_help = f"Frame: {W}x{H} | Buffer: {buffer_px}px"
    cv2.putText(frame, coord_help, (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = ROOT / "data" / "calibration"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "calibration_line.png"
    
    # Save the calibration image
    cv2.imwrite(str(output_path), frame)
    
    # Print results
    print(f"Saved: {output_path}")
    print(f"Frame size: {W}x{H}")
    print(f"Line coordinates (normalized): x1={line['x1']:.4f}, y1={line['y1']:.4f}, x2={line['x2']:.4f}, y2={line['y2']:.4f}")
    print(f"Line coordinates (pixels): ({x1}, {y1}) -> ({x2}, {y2})")
    print(f"Buffer threshold: {buffer_px}px")
    print("\nUsage in settings.yaml:")
    print(f"  line:")
    print(f"    x1: {line['x1']}")
    print(f"    y1: {line['y1']}")
    print(f"    x2: {line['x2']}")
    print(f"    y2: {line['y2']}")
    print("\nTechnique: Line-based with buffer zone (Layer 1: Signed Distance)")


if __name__ == "__main__":
    main()
