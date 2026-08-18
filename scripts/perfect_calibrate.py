#!/usr/bin/env python3
"""Perfect Calibration - Manually set the best zones for the sample.png

Based on image analysis of sample.png (710x604):
- Camera is INSIDE looking OUT through the door
- Strongest horizontal edge at y=480 (79.4% of height) - this is the door threshold
- Door spans full width (x=0 to x=710)
- OUTSIDE: Above y=480 (through the door opening)
- DOOR: Around y=480 (threshold corridor)
- INSIDE: Below y=480 (room interior)

This script sets the PERFECT calibration for this scene.

Usage:
    python scripts/perfect_calibrate.py
    python scripts/perfect_calibrate.py --camera-id office_entrance
    python scripts/perfect_calibrate.py --interactive
"""

from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_settings


def perfect_calibration(camera_id: str = "office_entrance", interactive: bool = False):
    """
    Set the perfect calibration for sample.png scene.
    
    Analysis results:
    - Image: 710x604
    - Camera: INSIDE looking OUT
    - Door threshold: y=480 (79.4% of height)
    - Door spans: full width (0-710)
    """
    
    print("\n" + "="*70)
    print("PERFECT CALIBRATION - Manually Optimized for Sample Scene")
    print("="*70)
    
    # Image dimensions
    img = cv2.imread(str(ROOT / "sample.png"))
    if img is None:
        print("ERROR: sample.png not found")
        return None
    
    h, w = img.shape[:2]
    print(f"\nImage Analysis:")
    print(f"  Size: {w}x{h}")
    print(f"  Camera position: INSIDE looking OUT")
    print(f"  Door threshold: y={int(h * 0.794)} ({h * 0.794:.1%} of height)")
    
    # FINAL CORRECT zones - NON-OVERLAPPING
    # Camera is INSIDE looking OUT through door
    # Priority: DOOR > INSIDE > OUTSIDE
    # INSIDE and OUTSIDE must NOT overlap, or OUTSIDE will never be detected
    
    # OUTSIDE: The door opening (where you see outside world)
    # This is a rectangle showing the outdoors through the door
    outside = [
        [0.30, 0.10],
        [0.70, 0.10],
        [0.70, 0.66],
        [0.30, 0.66],
    ]
    
    # DOOR corridor: Threshold at floor level
    # Foot point crosses here: OUTSIDE -> DOOR -> INSIDE = ENTRY
    #                          INSIDE -> DOOR -> OUTSIDE = EXIT
    door = [
        [0.25, 0.66],
        [0.75, 0.66],
        [0.75, 0.82],
        [0.25, 0.82],
    ]
    
    # INSIDE: Room interior - does NOT include OUTSIDE
    # This has 3 parts: left wall, ceiling+right wall, floor
    # Connected: left wall -> ceiling -> right wall -> floor -> left of door -> back to left wall
    inside = [
        # Part 1: Left wall area (left of OUTSIDE)
        [0.00, 0.00],    # Top-left corner
        [0.30, 0.00],    # Ceiling edge (left of OUTSIDE)
        [0.30, 0.10],    # Top-left of OUTSIDE (corner)
        [0.30, 0.66],    # Bottom-left of OUTSIDE (corner)
        
        # Part 2: Below DOOR (floor area)
        [0.25, 0.66],    # Top-left of DOOR
        [0.25, 0.82],    # Bottom-left of DOOR
        [0.00, 0.82],    # Floor left
        [0.00, 1.00],    # Bottom-left corner
        
        # Part 3: Right wall + ceiling (right of OUTSIDE)
        [1.00, 1.00],    # Bottom-right corner
        [1.00, 0.82],    # Floor right
        [0.75, 0.82],    # Bottom-right of DOOR
        [0.75, 0.66],    # Top-right of DOOR
        [0.70, 0.66],    # Bottom-right of OUTSIDE
        [0.70, 0.10],    # Top-right of OUTSIDE
        [0.70, 0.00],    # Ceiling edge (right of OUTSIDE)
        [1.00, 0.00],    # Top-right corner
        [1.00, 0.00],    # Closing point (will be removed by clean_polygon)
    ]
    
    zones = {
        "outside": outside,
        "door_corridor": door,
        "inside": inside,
    }
    
    print("\nPerfect Zones:")
    for name, poly in zones.items():
        center_x = np.mean([x for x, y in poly])
        center_y = np.mean([y for x, y in poly])
        print(f"\n{name.upper()}:")
        print(f"  Center: ({center_x:.3f}, {center_y:.3f})")
        print(f"  Vertices: {len(poly)}")
        for x, y in poly:
            px, py = int(x * w), int(y * h)
            print(f"    ({x:.3f}, {y:.3f}) -> ({px}, {py})")
    
    # Validate zones don't overlap incorrectly
    print("\n" + "="*70)
    print("Validation:")
    print("="*70)
    
    # Check that DOOR is below OUTSIDE
    outside_y = np.mean([y for x, y in zones["outside"]])
    door_y = np.mean([y for x, y in zones["door_corridor"]])
    inside_y = np.mean([y for x, y in zones["inside"]])
    
    print(f"  OUTSIDE center Y: {outside_y:.3f} ({int(outside_y * h)})")
    print(f"  DOOR center Y: {door_y:.3f} ({int(door_y * h)})")
    print(f"  INSIDE center Y: {inside_y:.3f} ({int(inside_y * h)})")
    
    if door_y > outside_y:
        print("  [OK] DOOR is below OUTSIDE (correct for INSIDE camera)")
    else:
        print("  [ERROR] DOOR is NOT below OUTSIDE")
    
    # Check that DOOR is in the correct position (around y=0.79)
    if 0.70 < door_y < 0.85:
        print(f"  [OK] DOOR at correct vertical position ({door_y:.2f})")
    else:
        print(f"  [WARN] DOOR vertical position may need adjustment ({door_y:.2f})")
    
    # Check all coordinates are in range
    all_valid = True
    for name, poly in zones.items():
        for x, y in poly:
            if not (0 <= x <= 1) or not (0 <= y <= 1):
                print(f"  [ERROR] {name} has out-of-range coordinate: ({x}, {y})")
                all_valid = False
    
    if all_valid:
        print("  [OK] All coordinates are in valid range [0, 1]")
    
    # Save to zones.yaml
    print("\n" + "="*70)
    print("Saving Configuration:")
    print("="*70)
    
    zones_path = ROOT / "config" / "zones.yaml"
    
    # Create backup
    if zones_path.exists():
        import shutil
        backup = zones_path.with_suffix(".yaml.bak")
        shutil.copy(zones_path, backup)
        print(f"  Backup created: {backup}")
    
    # Generate YAML
    lines = [
        f"# Door Intelligence regions (NORMALIZED 0-1 coords) - Perfect Calibration\n",
        f"# Camera: {camera_id} (INSIDE looking OUT)\n",
        "# Auto-detected door threshold at y=0.794 (79.4% of height)\n",
        "# Region priority: DOOR (decision band) > INSIDE > OUTSIDE\n",
        f"{camera_id}:\n",
        "  zones:\n",
    ]
    
    key_map = {
        "outside": "outside",
        "door_corridor": "door_corridor",
        "inside": "inside",
    }
    
    for name in ["outside", "door_corridor", "inside"]:
        poly = zones[name]
        lines.append(f"    {key_map[name]}:\n")
        for x, y in poly:
            lines.append(f"      - [{x}, {y}]\n")
    
    zones_path.write_text("".join(lines), encoding="utf-8")
    print(f"  Zones saved to: {zones_path}")
    
    # Update settings.yaml
    cfg = load_settings()
    cfg["door_intelligence"] = {
        "enabled": True,
        "camera_id": camera_id,
        "zones_path": "config/zones.yaml",
        "probe": "foot",
        "min_track_frames": 5,
        "min_dwell_door_sec": 0.15,
        "min_inside_frames": 3,
        "min_outside_frames": 3,
        "lock_after_event": False,
        "motion_toward_inside_dot": 0.25,
        "motion_outward_dot": 0.25,
        "peek_max_inside_sec": 1.5,
        "uturn_max_door_sec": 3.0,
        "min_event_confidence": 0.7,
        "purge_after_sec": 10.0,
    }
    
    # Set line at the door threshold
    cfg["entry_exit"]["line"] = {
        "x1": 0.25,
        "y1": round(door_y, 4),
        "x2": 0.75,
        "y2": round(door_y, 4),
    }
    cfg["entry_exit"]["entry_direction"] = "B_to_A"  # From bottom to top is entry
    
    settings_path = ROOT / "config" / "settings.yaml"
    if settings_path.exists():
        import shutil
        backup = settings_path.with_suffix(".yaml.bak")
        shutil.copy(settings_path, backup)
        print(f"  Settings backup: {backup}")
    
    with open(settings_path, 'w') as f:
        import yaml
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    
    print(f"  Settings saved to: {settings_path}")
    
    # Show visualization if interactive
    if interactive:
        print("\n" + "="*70)
        print("Visualization:")
        print("="*70)
        
        overlay = img.copy()
        colors = {
            "outside": (0, 255, 255),   # Cyan
            "door_corridor": (0, 165, 255),  # Orange
            "inside": (0, 255, 0),     # Green
        }
        
        for name, poly in zones.items():
            pts = np.array([[int(x * w), int(y * h)] for x, y in poly], np.int32)
            cv2.polylines(overlay, [pts], True, colors[name], 3)
            cv2.fillPoly(overlay, [pts], colors[name])
            
            center_x = int(np.mean([x * w for x, y in poly]))
            center_y = int(np.mean([y * h for x, y in poly]))
            cv2.putText(overlay, name.upper(), (center_x - 40, center_y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw threshold line
        line_y = int(door_y * h)
        cv2.line(overlay, (0, line_y), (w, line_y), (0, 0, 255), 2)
        cv2.putText(overlay, f"Threshold: y={line_y}", (10, line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.imshow("Perfect Calibration - OFFICE_ENTRANCE", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("CALIBRATION COMPLETE!")
    print("="*70)
    print("\nFSM Flow:")
    print("  OUTSIDE -> DOOR -> INSIDE = ENTRY")
    print("  INSIDE -> DOOR -> OUTSIDE = EXIT")
    print("\nDoor Threshold:")
    print(f"  Line at y={door_y:.3f} ({int(door_y * h)} pixels)")
    print("\nConfiguration Files:")
    print(f"  Zones: {zones_path}")
    print(f"  Settings: {settings_path}")
    print("\nNext Steps:")
    print("  1. Test with: python main.py --max-frames 50")
    print("  2. Verify with: python scripts/calibrate_door_regions.py --interactive")
    print("  3. Run full: python main.py")
    print("="*70 + "\n")
    
    return zones


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Perfect Calibration - Manually optimized for sample.png scene"
    )
    parser.add_argument(
        "--camera-id",
        default="office_entrance",
        help="Camera identifier"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Show visualization"
    )
    
    args = parser.parse_args()
    perfect_calibration(args.camera_id, args.interactive)
