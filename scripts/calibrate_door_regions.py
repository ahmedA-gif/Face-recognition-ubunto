#!/usr/bin/env python3
"""Calibrate the Door Intelligence regions: click to draw OUTSIDE / DOOR / INSIDE.

Captures a live frame (or a video frame) and lets you draw the three polygons
by clicking vertices. Press:

    o  → start OUTSIDE polygon      i  → start INSIDE polygon
    d  → start DOOR corridor        u  → undo last vertex
    r  → clear current polygon      w  → write config/zones.yaml and exit
    q  → quit without saving
    b  → toggle buffer zone visualization
    s  → save current configuration

Left-click adds a vertex; vertices are connected in click order. Use normalized
(0-1) coordinates so the zones survive resolution changes.

This implements the 3-zone Door Intelligence technique:
- OUTSIDE: Area outside the door (where people approach from)
- DOOR: Threshold corridor/band (where crossing is detected)
- INSIDE: Area inside the room (where people enter to)

Usage:
    python scripts/calibrate_door_regions.py
    python scripts/calibrate_door_regions.py --source data/test_video.mp4 --frame 120
    python scripts/calibrate_door_regions.py --source rtsp://127.0.0.1:8554/cam_01_sub --camera-id office_entrance
"""
from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_settings
from src.utils.geometry import clean_polygon, polygon_centroid, inward_normal

REGIONS = ("OUTSIDE", "DOOR", "INSIDE")
COLORS = {
    "OUTSIDE": (0, 255, 255),      # Yellow/Cyan
    "DOOR": (0, 165, 255),        # Orange
    "INSIDE": (0, 255, 0),        # Green
}

# Priority order for region detection (DOOR has highest priority)
REGION_PRIORITY = ("DOOR", "INSIDE", "OUTSIDE")


def _draw_state(canvas, polys, active, cursor, show_buffer=False, frame_shape=None):
    out = canvas.copy()
    h, w = frame_shape[:2] if frame_shape else (canvas.shape[0], canvas.shape[1])
    
    for name, poly in polys.items():
        if len(poly) < 2:
            continue
        color = COLORS[name]
        pts = np.asarray(poly, np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, color, 2, cv2.LINE_AA)
        for (x, y) in poly:
            cv2.circle(out, (x, y), 4, color, -1)
        if len(poly) >= 3:
            cv2.fillPoly(out, [pts], color)
            mask = np.zeros(out.shape[:2], np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            region = out.copy()
            region[mask == 0] = canvas[mask == 0]
            out = cv2.addWeighted(region, 0.25, out, 0.75, 0)
        cx = int(np.mean([p[0] for p in poly]))
        cy = int(np.mean([p[1] for p in poly]))
        cv2.putText(out, name, (cx - 10, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if active and cursor is not None:
        color = COLORS[active]
        cv2.line(out, cursor, (cursor[0] + 1, cursor[1] + 1), color, 1)

    # Draw inward normal indicator for DOOR region
    if show_buffer and "DOOR" in polys and len(polys["DOOR"]) >= 3:
        door_poly = polys["DOOR"]
        if len(door_poly) >= 3:
            regions_norm = {
                "OUTSIDE": [[p[0]/w, p[1]/h] for p in polys.get("OUTSIDE", [])],
                "INSIDE": [[p[0]/w, p[1]/h] for p in polys.get("INSIDE", [])],
                "DOOR": [[p[0]/w, p[1]/h] for p in polys.get("DOOR", [])]
            }
            normal = inward_normal(regions_norm)
            if normal:
                center_x = int(np.mean([p[0] for p in door_poly]))
                center_y = int(np.mean([p[1] for p in door_poly]))
                end_x = int(center_x + normal[0] * 50)
                end_y = int(center_y + normal[1] * 50)
                cv2.arrowedLine(out, (center_x, center_y), (end_x, end_y), (255, 0, 0), 2)
                cv2.putText(out, "INSIDE", (end_x + 5, end_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    help_text = "o:OUTSIDE d:DOOR i:INSIDE  u:undo  r:clear  b:buffer  s:save  w:write&quit  q:quit"
    cv2.rectangle(out, (0, 0), (len(out[0]), 26), (30, 30, 30), -1)
    cv2.putText(out, help_text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    status = " | ".join(f"{n}:{len(p)}" for n, p in polys.items())
    cv2.putText(out, status, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Add technique info
    technique_text = "Technique: Polygon FSM (OUTSIDE -> DOOR -> INSIDE)"
    cv2.putText(out, technique_text, (w - 350, h - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Draw OUTSIDE/DOOR/INSIDE polygons for door intelligence calibration."
    )
    ap.add_argument(
        "--source", 
        default=None, 
        help="Video file or RTSP URL (default: camera.source in settings)."
    )
    ap.add_argument(
        "--frame", 
        type=int, 
        default=0, 
        help="Frame index to grab from a video file (default: 0)."
    )
    ap.add_argument(
        "--config", 
        default=None, 
        help="Alternate settings.yaml path (default: config/settings.yaml)."
    )
    ap.add_argument(
        "--camera-id",
        default="camera_1",
        help="Camera identifier for zones.yaml (default: camera_1)."
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="Clean polygons by removing duplicates and simplifying."
    )
    ap.add_argument(
        "--validate",
        action="store_true",
        help="Validate regions and show inward normal direction."
    )
    args = ap.parse_args()

    cfg = load_settings(args.config)
    source = args.source or cfg["camera"]["source"]

    cap = cv2.VideoCapture(str(source) if not str(source).isdigit() else int(source))
    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {source}")
        sys.exit(1)
    
    for _ in range(args.frame + 1):
        ok, frame = cap.read()
        if not ok:
            print(f"ERROR: Could not read frame {args.frame} from {source}")
            cap.release()
            sys.exit(1)
    cap.release()
    h, w = frame.shape[:2]

    polys = {name: [] for name in REGIONS}
    active = "OUTSIDE"
    cursor = None
    show_buffer = False

    def on_mouse(event, x, y, *_):
        nonlocal cursor, active
        if event == cv2.EVENT_MOUSEMOVE:
            cursor = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            polys[active].append((int(x), int(y)))

    window_name = "Door Regions - Polygon FSM Calibration"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    
    print("\n" + "="*70)
    print("DOOR INTELLIGENCE CALIBRATION")
    print("="*70)
    print("\nInstructions:")
    print("  o  - Start drawing OUTSIDE polygon (area outside the door)")
    print("  d  - Start drawing DOOR polygon (threshold corridor/band)")
    print("  i  - Start drawing INSIDE polygon (area inside the room)")
    print("  u  - Undo last vertex")
    print("  r  - Clear current polygon")
    print("  b  - Toggle buffer/inward normal visualization")
    print("  s  - Save current configuration")
    print("  w  - Write to config/zones.yaml and exit")
    print("  q  - Quit without saving")
    print("\nTechnique: 3-Zone Polygon FSM")
    print("  - OUTSIDE -> DOOR -> INSIDE = ENTRY")
    print("  - INSIDE -> DOOR -> OUTSIDE = EXIT")
    print("  - DOOR region is the decision band with highest priority")
    print("="*70 + "\n")

    while True:
        vis = _draw_state(frame, polys, active, cursor, show_buffer, frame.shape)
        cv2.imshow(window_name, vis)
        key = cv2.waitKey(1) & 0xFF
        ch = chr(key).lower() if 32 <= key < 127 else ""
        
        if ch == "o":
            active = "OUTSIDE"
            print("[Mode] Drawing OUTSIDE region")
        elif ch == "d":
            active = "DOOR"
            print("[Mode] Drawing DOOR corridor (threshold band)")
        elif ch == "i":
            active = "INSIDE"
            print("[Mode] Drawing INSIDE region")
        elif ch == "u":
            if polys[active]:
                polys[active].pop()
                print(f"[Undo] Removed last vertex from {active}")
        elif ch == "r":
            polys[active].clear()
            print(f"[Clear] Cleared {active} polygon")
        elif ch == "b":
            show_buffer = not show_buffer
            print(f"[Buffer] Visualization {'ON' if show_buffer else 'OFF'}")
        elif ch == "s":
            # Save temporary snapshot
            save_snapshot(polys, frame.shape, args.camera_id)
            print("[Save] Temporary configuration saved")
        elif ch == "w":
            break
        elif key in (27, ord("q")):
            cv2.destroyAllWindows()
            print("[Exit] Calibration cancelled")
            return

    cv2.destroyAllWindows()

    # Clean polygons if requested
    if args.clean:
        for name in REGIONS:
            if len(polys[name]) >= 3:
                cleaned = clean_polygon(polys[name])
                if cleaned:
                    polys[name] = cleaned
                    print(f"[Clean] {name} polygon cleaned ({len(polys[name])} vertices)")

    # Normalize coordinates
    normalized = {
        name: [[round(x / w, 4), round(y / h, 4)] for (x, y) in poly]
        for name, poly in polys.items()
    }
    
    # Validate regions
    empty = [name for name in REGIONS if len(normalized[name]) < 3]
    if empty:
        print(f"WARNING: regions with < 3 points will be skipped: {empty}")
    
    # Validate inward normal if we have all three regions
    if not empty:
        regions_norm = {
            "OUTSIDE": normalized["OUTSIDE"],
            "DOOR": normalized["DOOR"],
            "INSIDE": normalized["INSIDE"]
        }
        normal = inward_normal(regions_norm)
        if normal:
            print(f"[Validation] Inward normal: ({normal[0]:.3f}, {normal[1]:.3f})")
            print("  This direction points from OUTSIDE toward INSIDE")
        else:
            print("[Validation] WARNING: Could not compute inward normal")
            print("  Check that OUTSIDE and INSIDE regions are properly defined")

    # Save to zones.yaml
    zones_path = ROOT / "config" / "zones.yaml"
    lines = [
        "# Door Intelligence regions (NORMALIZED 0-1 coords) — generated by calibrate_door_regions.py\n",
        f"# Camera: {args.camera_id}\n",
        "# Region priority: DOOR (decision band) > INSIDE > OUTSIDE\n",
        "# Technique: Polygon FSM with 3 zones\n",
        f"{args.camera_id}:\n",
        "  zones:\n",
    ]
    key_map = {"OUTSIDE": "outside", "DOOR": "door_corridor", "INSIDE": "inside"}
    for name in REGIONS:
        poly = normalized[name]
        if len(poly) < 3:
            continue
        lines.append(f"    {key_map[name]}:\n")
        for x, y in poly:
            lines.append(f"      - [{x}, {y}]\n")
    zones_path.write_text("".join(lines), encoding="utf-8")
    
    print(f"\n[Success] Saved {zones_path}")
    print(f"[Setup] Set door_intelligence.enabled: true in config/settings.yaml")
    print(f"[Setup] Set door_intelligence.camera_id: {args.camera_id}")
    print(f"[Setup] Set door_intelligence.zones_path: config/zones.yaml")
    
    # Print summary
    print("\n" + "="*70)
    print("CALIBRATION SUMMARY")
    print("="*70)
    print(f"Camera ID: {args.camera_id}")
    print(f"Frame size: {w}x{h}")
    for name in REGIONS:
        poly = normalized[name]
        if len(poly) >= 3:
            print(f"  {name}: {len(poly)} vertices")
        else:
            print(f"  {name}: NOT DEFINED")
    print(f"\nInward normal: Points from OUTSIDE to INSIDE")
    print(f"Technique: Polygon FSM (OUTSIDE -> DOOR -> INSIDE)")
    print("="*70)


def save_snapshot(polys, frame_shape, camera_id):
    """Save a temporary snapshot of current calibration state."""
    h, w = frame_shape[:2]
    normalized = {
        name: [[round(x / w, 4), round(y / h, 4)] for (x, y) in poly]
        for name, poly in polys.items()
    }
    
    snapshot_dir = ROOT / "data" / "calibration"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"{camera_id}_zones_snapshot.json"
    
    snapshot = {
        "camera_id": camera_id,
        "frame_size": {"width": w, "height": h},
        "zones": normalized,
        "timestamp": __import__("datetime").datetime.now().isoformat()
    }
    
    with open(snapshot_path, 'w') as f:
        json.dump(snapshot, f, indent=2)
    
    return snapshot_path


if __name__ == "__main__":
    main()
