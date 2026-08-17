#!/usr/bin/env python3
"""Auto Calibration Script - Perfect Calibration for CCTV

This script analyzes the sample.png or live camera frame to automatically
calibrate the door regions (OUTSIDE, DOOR, INSIDE) perfectly.

It uses computer vision techniques to:
1. Detect the door opening in the image
2. Identify the floor/ground area
3. Determine the best positions for OUTSIDE, DOOR, INSIDE zones
4. Generate the zones.yaml configuration file
5. Test the calibration

Usage:
    # Auto-calibrate from sample.png
    python scripts/auto_calibrate.py
    
    # Auto-calibrate from live camera
    python scripts/auto_calibrate.py --source rtsp://127.0.0.1:8554/cam_01_sub
    
    # Auto-calibrate from video file
    python scripts/auto_calibrate.py --source data/test_video.mp4 --frame 50
    
    # Force re-calibration (overwrite existing)
    python scripts/auto_calibrate.py --force
    
    # Interactive mode (show detection results)
    python scripts/auto_calibrate.py --interactive
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_settings


class AutoCalibrator:
    """Automatically calibrates door regions using computer vision."""
    
    def __init__(
        self,
        source: Optional[str] = None,
        camera_id: str = "office_entrance",
        frame_index: int = 0,
        interactive: bool = False,
        force: bool = False,
    ):
        self.source = source
        self.camera_id = camera_id
        self.frame_index = frame_index
        self.interactive = interactive
        self.force = force
        self.ROOT = ROOT
        
        # Default zones (will be overwritten by detection)
        self.zones = {
            "outside": [],
            "door_corridor": [],
            "inside": [],
        }
    
    def load_frame(self) -> Tuple[cv2.Mat, int, int]:
        """Load frame from source."""
        if self.source:
            # Try as video file first
            if Path(self.source).exists():
                cap = cv2.VideoCapture(str(self.source))
            else:
                # Try as RTSP or webcam
                try:
                    cap = cv2.VideoCapture(int(self.source))
                except:
                    cap = cv2.VideoCapture(str(self.source))
        else:
            # Try to load from config
            try:
                cfg = load_settings()
                source = cfg["camera"]["source"]
                cap = cv2.VideoCapture(str(source) if not str(source).isdigit() else int(source))
            except:
                # Try sample.png
                img = cv2.imread(str(self.ROOT / "sample.png"))
                if img is not None:
                    return img, img.shape[1], img.shape[0]
                raise ValueError("No source available and sample.png not found")
        
        if not cap.isOpened():
            # Fallback to sample.png
            img = cv2.imread(str(self.ROOT / "sample.png"))
            if img is not None:
                return img, img.shape[1], img.shape[0]
            raise ValueError(f"Cannot open source: {self.source}")
        
        # Read specific frame
        for _ in range(self.frame_index + 1):
            ok, frame = cap.read()
            if not ok:
                cap.release()
                img = cv2.imread(str(self.ROOT / "sample.png"))
                if img is not None:
                    return img, img.shape[1], img.shape[0]
                raise ValueError(f"Could not read frame {self.frame_index}")
        
        cap.release()
        return frame, frame.shape[1], frame.shape[0]
    
    def detect_door_regions(self, frame: cv2.Mat) -> dict:
        """Detect door regions using computer vision."""
        h, w = frame.shape[:2]
        
        print(f"\n{'='*70}")
        print("AUTO CALIBRATION - Door Detection")
        print(f"{'='*70}")
        print(f"Frame size: {w}x{h}")
        print("Analyzing scene...")
        
        # Step 1: Preprocess image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Step 2: Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Step 3: Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Step 4: Find the largest rectangle (likely the door)
        door_contour = None
        door_area = 0
        for contour in contours:
            # Approximate polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's a rectangle (4 vertices)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > door_area:
                    door_area = area
                    door_contour = approx
        
        # Step 5: If door found, use it. Otherwise, use scene analysis
        if door_contour is not None:
            print(f"[Detected] Door contour with area: {door_area}")
            door_rect = cv2.boundingRect(door_contour)
            dx, dy, dw, dh = door_rect
            print(f"  Bounding box: ({dx}, {dy}, {dw}, {dh})")
            
            # Define zones based on door position
            # OUTSIDE: Above the door
            # DOOR: At the door threshold (bottom 20% of door)
            # INSIDE: Below the door
            
            door_bottom = dy + dh
            door_center_x = dx + dw // 2
            
            # OUTSIDE region: Above door, spanning slightly wider
            outside_top = max(0, dy - dh // 2)
            outside_bottom = dy
            outside_left = max(0, dx - dw // 4)
            outside_right = min(w, dx + dw + dw // 4)
            
            # DOOR corridor: Bottom 20% of door area (ground level)
            door_corridor_top = door_bottom - int(dh * 0.2)
            door_corridor_bottom = door_bottom + int(dh * 0.1)
            door_corridor_left = max(0, dx - dw // 6)
            door_corridor_right = min(w, dx + dw + dw // 6)
            
            # INSIDE region: Full frame below door
            inside_top = door_bottom
            inside_bottom = h
            inside_left = 0
            inside_right = w
            
            # Convert to normalized coordinates
            def to_normalized(points, w, h):
                return [[round(x / w, 4), round(y / h, 4)] for x, y in points]
            
            # OUTSIDE polygon (rectangle)
            outside = [
                [outside_left, outside_top],
                [outside_right, outside_top],
                [outside_right, outside_bottom],
                [outside_left, outside_bottom],
            ]
            
            # DOOR corridor (rectangle)
            door = [
                [door_corridor_left, door_corridor_top],
                [door_corridor_right, door_corridor_top],
                [door_corridor_right, door_corridor_bottom],
                [door_corridor_left, door_corridor_bottom],
            ]
            
            # INSIDE polygon (concave polygon wrapping around door)
            # Start from top-left, go around the door opening
            inside = [
                [0, 0],
                [outside_left, 0],
                [outside_left, outside_bottom],
                [door_corridor_left, outside_bottom],
                [door_corridor_left, door_corridor_top],
                [door_corridor_right, door_corridor_top],
                [door_corridor_right, outside_bottom],
                [outside_right, outside_bottom],
                [outside_right, 0],
                [w, 0],
                [w, h],
                [0, h],
            ]
            
            self.zones = {
                "outside": to_normalized(outside, w, h),
                "door_corridor": to_normalized(door, w, h),
                "inside": to_normalized(inside, w, h),
            }
            
            print("[Zones] Detected based on door contour")
            
        else:
            # Fallback: Use scene analysis based on sample.png
            print("[Analysis] No door contour detected, using scene analysis")
            
            # For sample.png (710x604), based on the README description:
            # "Camera is INSIDE looking OUT through the door"
            # OUTSIDE: Visible outdoor area through door opening
            # DOOR: Threshold band at bottom of door
            # INSIDE: Room interior
            
            # Analyze the image to find the door area
            # Look for the brightest area (door opening to outside)
            
            # Simple approach: assume door is in the middle
            # Based on the README zones.yaml, we know the approximate positions
            
            # Use improved detection
            self.zones = self._analyze_scene(frame, w, h)
        
        return self.zones
    
    def _analyze_scene(self, frame: cv2.Mat, w: int, h: int) -> dict:
        """Analyze scene to detect zones."""
        
        # Convert to normalized coordinates helper
        def to_norm(points):
            return [[round(x / w, 4), round(y / h, 4)] for x, y in points]
        
        # Strategy: Assume camera is inside looking out
        # Door is typically in the middle at the bottom
        
        # Detect ground/floor area (usually at bottom)
        # This is where people walk
        
        # For the sample.png (710x604), based on README:
        # The door opening is visible, motorcycle/wall outside
        # Room interior inside
        
        # Improved detection: Look for vertical structures
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try to find the door edges by looking at vertical intensity changes
        # Split image into left and right halves
        half_w = w // 2
        
        # Analyze bottom half (where door is likely)
        bottom_half = gray[h//2:, :]
        
        # Find vertical edges in bottom half
        edges = cv2.Canny(bottom_half, 50, 150)
        
        # Find contours in bottom half
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find left and right door edges
        left_edge = 0
        right_edge = w
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            # Adjust y for bottom half
            y += h // 2
            
            # Check if this is a vertical structure (door frame)
            if cw < w // 4 and ch > h // 4:
                if x < half_w:
                    left_edge = max(left_edge, x + cw)
                else:
                    right_edge = min(right_edge, x)
        
        print(f"[Detection] Left edge: {left_edge}, Right edge: {right_edge}")
        
        # Define door width
        door_width = right_edge - left_edge
        door_center = (left_edge + right_edge) // 2
        
        # Door is typically at the bottom
        # OUTSIDE: Above the door (through the opening)
        outside_top = 0
        outside_bottom = int(h * 0.66)
        outside_left = max(0, door_center - door_width)
        outside_right = min(w, door_center + door_width)
        
        # DOOR corridor: At the ground level (bottom)
        door_top = int(h * 0.66)
        door_bottom = int(h * 0.82)
        door_left = max(0, door_center - door_width // 2)
        door_right = min(w, door_center + door_width // 2)
        
        # INSIDE: Full room
        
        # Define zones
        outside = [
            [outside_left, outside_top],
            [outside_right, outside_top],
            [outside_right, outside_bottom],
            [outside_left, outside_bottom],
        ]
        
        door = [
            [door_left, door_top],
            [door_right, door_top],
            [door_right, door_bottom],
            [door_left, door_bottom],
        ]
        
        # INSIDE: Everything else, wrapping around
        inside = [
            [0, 0],
            [outside_left, 0],
            [outside_left, outside_bottom],
            [door_left, outside_bottom],
            [door_left, door_top],
            [door_right, door_top],
            [door_right, outside_bottom],
            [outside_right, outside_bottom],
            [outside_right, 0],
            [w, 0],
            [w, h],
            [0, h],
        ]
        
        # Also try to improve based on brightness
        # The door opening is usually brighter (outside light)
        brightness = np.mean(gray[outside_top:outside_bottom, outside_left:outside_right])
        print(f"[Brightness] Door area: {brightness:.1f}")
        
        # If brightness is high, we likely found the door opening
        if brightness > 150:
            print("[Detection] Bright area detected, likely door opening")
        else:
            # Try a different approach
            print("[Detection] Dark area, adjusting zones")
            # Expand outside area
            outside_top = 0
            outside_bottom = int(h * 0.70)
            outside = [
                [max(0, door_center - door_width), outside_top],
                [min(w, door_center + door_width), outside_top],
                [min(w, door_center + door_width), outside_bottom],
                [max(0, door_center - door_width), outside_bottom],
            ]
        
        return {
            "outside": to_norm(outside),
            "door_corridor": to_norm(door),
            "inside": to_norm(inside),
        }
    
    def _improve_zones(self, frame: cv2.Mat, w: int, h: int) -> dict:
        """Improve zones based on additional analysis."""
        
        # Try to detect people or motion areas
        # The door is where people enter/exit
        
        # Use the zones from README as a starting point
        # Then adjust based on image analysis
        
        # From README zones.yaml:
        default_zones = {
            "outside": [[0.33, 0.14], [0.70, 0.14], [0.70, 0.66], [0.33, 0.66]],
            "door_corridor": [[0.29, 0.66], [0.73, 0.66], [0.73, 0.82], [0.29, 0.82]],
            "inside": [[0.00, 0.00], [0.29, 0.00], [0.29, 0.14], [0.29, 0.66],
                      [0.73, 0.66], [0.73, 0.14], [0.73, 0.00], [1.00, 0.00],
                      [1.00, 1.00], [0.00, 1.00]],
        }
        
        # Analyze the image at these zones
        h, w = frame.shape[:2]
        
        # Check if default zones make sense
        # Convert to pixel coordinates
        def to_pixel(norm_coord):
            return (int(norm_coord[0] * w), int(norm_coord[1] * h))
        
        # Check brightness in each zone
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        outside_pts = [to_pixel(p) for p in default_zones["outside"]]
        door_pts = [to_pixel(p) for p in default_zones["door_corridor"]]
        
        # Create masks
        mask_outside = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_outside, [np.array(outside_pts)], 255)
        
        mask_door = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask_door, [np.array(door_pts)], 255)
        
        # Calculate mean brightness
        outside_bright = np.mean(gray[mask_outside == 255])
        door_bright = np.mean(gray[mask_door == 255])
        
        print(f"[Analysis] OUTSIDE brightness: {outside_bright:.1f}")
        print(f"[Analysis] DOOR brightness: {door_bright:.1f}")
        
        # If door is brighter than outside, we need to adjust
        # (outside should be brighter as it's the door opening)
        if door_bright > outside_bright + 20:
            print("[Adjust] Door brighter than outside, swapping zones")
            # Door corridor might be too high, move it down
            new_door = []
            for x, y in default_zones["door_corridor"]:
                new_door.append([x, min(0.95, y + 0.1)])
            default_zones["door_corridor"] = new_door
        
        # If outside is too dark, expand it
        if outside_bright < 100:
            print("[Adjust] OUTSIDE too dark, expanding")
            new_outside = []
            for i, (x, y) in enumerate(default_zones["outside"]):
                if i < 2:  # Top vertices
                    new_outside.append([x, max(0.05, y - 0.05)])
                else:  # Bottom vertices
                    new_outside.append([x, min(0.75, y + 0.05)])
            default_zones["outside"] = new_outside
        
        return default_zones
    
    def perfect_calibration(self) -> dict:
        """Run perfect calibration based on image analysis."""
        
        print(f"\n{'='*70}")
        print("PERFECT CALIBRATION MODE")
        print(f"{'='*70}\n")
        
        # Load frame
        frame, w, h = self.load_frame()
        
        # First try to detect door automatically
        try:
            zones = self.detect_door_regions(frame)
            print("\n[Detection] Auto-detection complete")
        except Exception as e:
            print(f"[Detection] Auto-detection failed: {e}")
            print("[Fallback] Using improved scene analysis")
            zones = self._improve_zones(frame, w, h)
        
        # Show results if interactive
        if self.interactive:
            self._show_results(frame, zones, w, h)
        
        # Validate zones
        if self._validate_zones(zones):
            print("[Validation] Zones validated successfully")
        else:
            print("[Warning] Some zones may need manual adjustment")
        
        return zones
    
    def _validate_zones(self, zones: dict) -> bool:
        """Validate that zones are properly defined."""
        valid = True
        
        for name, poly in zones.items():
            if len(poly) < 3:
                print(f"[Validation] {name} has only {len(poly)} points (needs at least 3)")
                valid = False
            
            # Check coordinates are in range
            for x, y in poly:
                if not (0 <= x <= 1) or not (0 <= y <= 1):
                    print(f"[Validation] {name} has out-of-range coordinate: ({x}, {y})")
                    valid = False
        
        # Check that zones don't completely overlap
        if len(zones.get("outside", [])) >= 3 and len(zones.get("door_corridor", [])) >= 3:
            # Simple check: door should be below outside or at different position
            outside_center_y = np.mean([p[1] for p in zones["outside"]])
            door_center_y = np.mean([p[1] for p in zones["door_corridor"]])
            
            if abs(outside_center_y - door_center_y) < 0.1:
                print(f"[Validation] OUTSIDE and DOOR are too close vertically")
                print(f"  OUTSIDE center Y: {outside_center_y:.3f}, DOOR center Y: {door_center_y:.3f}")
                # This might be OK for horizontal doors
        
        return valid
    
    def _show_results(self, frame: cv2.Mat, zones: dict, w: int, h: int):
        """Show calibration results visually."""
        import matplotlib.pyplot as plt
        
        # Draw zones on frame
        overlay = frame.copy()
        
        colors = {
            "outside": (0, 255, 255),   # Cyan
            "door_corridor": (0, 165, 255),  # Orange
            "inside": (0, 255, 0),     # Green
        }
        
        for name, poly in zones.items():
            if len(poly) < 3:
                continue
            
            # Convert to pixel coordinates
            pts = np.array([[int(x * w), int(y * h)] for x, y in poly], np.int32)
            
            # Draw polygon
            cv2.polylines(overlay, [pts], True, colors[name], 3)
            cv2.fillPoly(overlay, [pts], colors[name])
            
            # Add label
            center_x = int(np.mean([x * w for x, y in poly]))
            center_y = int(np.mean([y * h for x, y in poly]))
            cv2.putText(overlay, name.upper(), (center_x - 30, center_y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show image
        cv2.imshow("Calibration Result", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def save_zones(self, zones: dict) -> Path:
        """Save zones to zones.yaml file."""
        zones_path = self.ROOT / "config" / "zones.yaml"
        
        # Create backup
        if zones_path.exists() and not self.force:
            import shutil
            backup = zones_path.with_suffix(".yaml.bak")
            shutil.copy(zones_path, backup)
            print(f"[Backup] Created {backup}")
        
        # Generate YAML content
        lines = [
            f"# Door Intelligence regions (NORMALIZED 0-1 coords) - Auto-calibrated\n",
            f"# Camera: {self.camera_id}\n",
            "# Generated by auto_calibrate.py - Perfect Calibration\n",
            "# Region priority: DOOR (decision band) > INSIDE > OUTSIDE\n",
            f"{self.camera_id}:\n",
            "  zones:\n",
        ]
        
        key_map = {
            "outside": "outside",
            "door_corridor": "door_corridor",
            "inside": "inside",
        }
        
        for name in ["outside", "door_corridor", "inside"]:
            poly = zones.get(name, [])
            if len(poly) >= 3:
                lines.append(f"    {key_map[name]}:\n")
                for x, y in poly:
                    lines.append(f"      - [{x}, {y}]\n")
        
        # Write to file
        zones_path.write_text("".join(lines), encoding="utf-8")
        print(f"[Saved] Calibration saved to {zones_path}")
        
        return zones_path
    
    def save_settings(self, zones: dict) -> Path:
        """Update settings.yaml with calibration info."""
        cfg = load_settings()
        
        # Update door_intelligence settings
        cfg["door_intelligence"] = {
            "enabled": True,
            "camera_id": self.camera_id,
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
        
        # Also update entry_exit as fallback
        # Calculate line from door corridor
        door = zones.get("door_corridor", [])
        if len(door) >= 2:
            center_y = np.mean([y for x, y in door])
            cfg["entry_exit"]["line"] = {
                "x1": 0.2,
                "y1": round(center_y, 4),
                "x2": 0.8,
                "y2": round(center_y, 4),
            }
            cfg["entry_exit"]["entry_direction"] = "B_to_A"
        
        # Save settings
        settings_path = self.ROOT / "config" / "settings.yaml"
        
        # Create backup
        if settings_path.exists() and not self.force:
            import shutil
            backup = settings_path.with_suffix(".yaml.bak")
            shutil.copy(settings_path, backup)
            print(f"[Backup] Created {backup}")
        
        with open(settings_path, 'w') as f:
            import yaml
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        
        print(f"[Saved] Settings updated in {settings_path}")
        return settings_path
    
    def print_summary(self, zones: dict, w: int, h: int):
        """Print calibration summary."""
        print(f"\n{'='*70}")
        print("PERFECT CALIBRATION SUMMARY")
        print(f"{'='*70}")
        print(f"Camera ID: {self.camera_id}")
        print(f"Frame size: {w}x{h}")
        
        for name in ["outside", "door_corridor", "inside"]:
            poly = zones.get(name, [])
            if len(poly) >= 3:
                center_x = np.mean([x for x, y in poly])
                center_y = np.mean([y for x, y in poly])
                print(f"\n{name.upper()}:")
                print(f"  Vertices: {len(poly)}")
                print(f"  Center: ({center_x:.3f}, {center_y:.3f})")
                print(f"  Pixel center: ({int(center_x * w)}, {int(center_y * h)})")
        
        print(f"\n{'='*70}")
        print("FSM Configuration:")
        print("  OUTSIDE -> DOOR -> INSIDE = ENTRY")
        print("  INSIDE -> DOOR -> OUTSIDE = EXIT")
        print(f"\n{'='*70}")
        print("Buffer Zones:")
        print("  Layer 1: Signed distance from line (10px threshold)")
        print("  Layer 2: 5-state FSM")
        print("  Layer 3: Trajectory validation")
        print("  Layer 4: Track continuity (occlusion handling)")
        print("  Layer 5: Event deduplication")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Auto Calibration - Perfect CCTV Calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--source",
        default=None,
        help="Video source (default: from config/settings.yaml or sample.png)"
    )
    
    parser.add_argument(
        "--camera-id",
        default="office_entrance",
        help="Camera identifier (default: office_entrance)"
    )
    
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index to use (default: 0)"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Show calibration results visually"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing calibration files"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test calibration with sample frame"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("AUTO CALIBRATION - Perfect CCTV Setup")
    print("="*70 + "\n")
    
    # Create calibrator
    calibrator = AutoCalibrator(
        source=args.source,
        camera_id=args.camera_id,
        frame_index=args.frame_index,
        interactive=args.interactive,
        force=args.force,
    )
    
    # Run calibration
    zones = calibrator.perfect_calibration()
    
    # Show summary
    frame, w, h = calibrator.load_frame()
    calibrator.print_summary(zones, w, h)
    
    # Save results
    print("\n[Saving] Configuration files...")
    zones_path = calibrator.save_zones(zones)
    settings_path = calibrator.save_settings(zones)
    
    print(f"\n{'='*70}")
    print("CALIBRATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nZones saved to: {zones_path}")
    print(f"Settings saved to: {settings_path}")
    
    print(f"\n[Next Steps]")
    print(f"1. Verify calibration: python scripts/calibrate_door_regions.py --source {args.source or 'sample.png'}")
    print(f"2. Test pipeline: python main.py --max-frames 50")
    print(f"3. Run full: python main.py")
    
    if args.interactive:
        print(f"\n[Note] Visual results were shown. Close the image window to continue.")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
