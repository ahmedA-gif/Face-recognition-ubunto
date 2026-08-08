#!/usr/bin/env python3
"""Calibrate the Door Intelligence regions: click to draw OUTSIDE / DOOR / INSIDE.

Captures a live frame (or a video frame) and lets you draw the three polygons
by clicking vertices. Press:

    o  → start OUTSIDE polygon      i  → start INSIDE polygon
    d  → start DOOR corridor        u  → undo last vertex
    r  → clear current polygon      w  → write config/zones.yaml and exit
    q  → quit without saving

Left-click adds a vertex; vertices are connected in click order. Use normalized
(0-1) coordinates so the zones survive resolution changes.

Usage:
    .venv/bin/python3 scripts/calibrate_door_regions.py
    .venv/bin/python3 scripts/calibrate_door_regions.py --source data/test_video.mp4 --frame 120
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_settings

REGIONS = ("OUTSIDE", "DOOR", "INSIDE")
COLORS = {
    "OUTSIDE": (0, 255, 255),
    "DOOR": (0, 165, 255),
    "INSIDE": (0, 255, 0),
}


def _draw_state(canvas, polys, active, cursor):
    out = canvas.copy()
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

    help_text = "o:OUTSIDE d:DOOR i:INSIDE  u:undo  r:clear  w:write&quit  q:quit"
    cv2.rectangle(out, (0, 0), (len(out[0]), 26), (30, 30, 30), -1)
    cv2.putText(out, help_text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    status = " | ".join(f"{n}:{len(p)}" for n, p in polys.items())
    cv2.putText(out, status, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Draw OUTSIDE/DOOR/INSIDE polygons for door intelligence.")
    ap.add_argument("--source", default=None, help="Video file or RTSP URL (default: camera.source in settings).")
    ap.add_argument("--frame", type=int, default=0, help="Frame index to grab from a video file.")
    ap.add_argument("--config", default=None, help="Alternate settings.yaml path.")
    args = ap.parse_args()

    cfg = load_settings(args.config)
    source = args.source or cfg["camera"]["source"]

    cap = cv2.VideoCapture(str(source) if not str(source).isdigit() else int(source))
    if not cap.isOpened():
        sys.exit(f"Cannot open source: {source}")
    for _ in range(args.frame + 1):
        ok, frame = cap.read()
        if not ok:
            sys.exit(f"Could not read frame {args.frame} from {source}")
    cap.release()
    h, w = frame.shape[:2]

    polys = {name: [] for name in REGIONS}
    active = "OUTSIDE"
    cursor = None

    def on_mouse(event, x, y, *_):
        nonlocal cursor, active
        if event == cv2.EVENT_MOUSEMOVE:
            cursor = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            polys[active].append((int(x), int(y)))

    cv2.namedWindow("Door Regions")
    cv2.setMouseCallback("Door Regions", on_mouse)

    while True:
        vis = _draw_state(frame, polys, active, cursor)
        cv2.imshow("Door Regions", vis)
        key = cv2.waitKey(1) & 0xFF
        ch = chr(key).lower() if 32 <= key < 127 else ""
        if ch == "o":
            active = "OUTSIDE"
        elif ch == "d":
            active = "DOOR"
        elif ch == "i":
            active = "INSIDE"
        elif ch == "u":
            if polys[active]:
                polys[active].pop()
        elif ch == "r":
            polys[active].clear()
        elif ch == "w":
            break
        elif key in (27, ord("q")):
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    normalized = {
        name: [[round(x / w, 4), round(y / h, 4)] for (x, y) in poly]
        for name, poly in polys.items()
    }
    empty = [name for name in REGIONS if len(normalized[name]) < 3]
    if empty:
        print(f"WARNING: regions with < 3 points will be skipped: {empty}")

    zones_path = ROOT / "config" / "zones.yaml"
    lines = [
        "# Door Intelligence regions (NORMALIZED 0-1 coords) — generated by calibrate_door_regions.py\n",
        "# Region priority: DOOR (decision band) > INSIDE > OUTSIDE\n",
        "camera_1:\n",
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
    print(f"Saved {zones_path}")
    print("Then set door_intelligence.enabled: true in config/settings.yaml")


if __name__ == "__main__":
    main()
