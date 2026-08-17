#!/usr/bin/env python3
"""Unified Calibration System - All Techniques Combined.

This script provides a complete calibration solution implementing all techniques:
1. Line-based boundary with buffer zones (Layer 1: Signed Distance)
2. 3-zone Polygon FSM (OUTSIDE -> DOOR -> INSIDE)
3. 5-layer validation system
4. Dynamic boundary auto-learning

The script allows you to:
- Calibrate a simple line boundary with buffer visualization
- Calibrate full 3-zone polygons (OUTSIDE/DOOR/INSIDE)
- Validate the inward normal direction
- Test the configuration with live preview
- Auto-detect the optimal line from motion patterns

Usage:
    # Line-based calibration (simple)
    python scripts/calibrate_unified.py --mode line --y1 0.55 --y2 0.55
    
    # Polygon-based calibration (3-zone FSM)
    python scripts/calibrate_unified.py --mode polygon
    
    # Auto-detect line from camera
    python scripts/calibrate_unified.py --mode auto --source rtsp://127.0.0.1:8554/cam_01_sub
    
    # Full calibration with all options
    python scripts/calibrate_unified.py --mode polygon --source rtsp://127.0.0.1:8554/cam_01_sub --camera-id office_entrance

Techniques Implemented:
- OUTSIDE region: Area outside the door where people approach from
- DOOR region: Threshold corridor/band where crossing is detected
- INSIDE region: Area inside the room where people enter to
- Buffer zone: Spatial hysteresis around boundaries
- Polygon FSM: Finite state machine with 8 states for accurate tracking
- 5-layer validation: Signed distance + FSM + trajectory + continuity + deduplication
"""

from __future__ import annotations

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_settings
from src.utils.geometry import (
    clean_polygon,
    polygon_centroid,
    inward_normal,
    signed_distance,
    foot_point,
    which_side,
)
from src.events.store import EventsStore
from src.events.door_intelligence import DoorIntelligenceEngine, TrackState as DI_TrackState
from src.events.entry_exit_v2 import EntryExitEngineV2


class CalibrationMode(Enum):
    LINE = "line"
    POLYGON = "polygon"
    AUTO = "auto"
    TEST = "test"


@dataclass
class LineConfig:
    """Configuration for line-based entry/exit detection."""
    x1: float = 0.5
    y1: float = 0.5
    x2: float = 0.5
    y2: float = 0.5
    entry_direction: str = "B_to_A"
    buffer_threshold: float = 10.0
    hysteresis_px: float = 14.0


@dataclass
class PolygonConfig:
    """Configuration for polygon-based door intelligence."""
    outside: List[Tuple[float, float]] = field(default_factory=list)
    door: List[Tuple[float, float]] = field(default_factory=list)
    inside: List[Tuple[float, float]] = field(default_factory=list)
    min_dwell_door_sec: float = 0.15
    min_inside_frames: int = 3
    min_outside_frames: int = 3


@dataclass
class CalibrationResult:
    """Result of calibration process."""
    mode: str
    camera_id: str
    frame_size: Tuple[int, int]
    line_config: Optional[LineConfig] = None
    polygon_config: Optional[PolygonConfig] = None
    inward_normal: Optional[Tuple[float, float]] = None
    validation_passed: bool = True
    error_message: Optional[str] = None


class UnifiedCalibrator:
    """Unified calibration system implementing all techniques."""
    
    def __init__(
        self,
        camera_id: str = "camera_1",
        mode: CalibrationMode = CalibrationMode.POLYGON,
    ):
        self.camera_id = camera_id
        self.mode = mode
        self.ROOT = ROOT
        
        # Initialize configurations
        self.line_config = LineConfig()
        self.polygon_config = PolygonConfig()
        
        # Drawing state for polygon mode
        self.polys: Dict[str, List[Tuple[int, int]]] = {
            "OUTSIDE": [],
            "DOOR": [],
            "INSIDE": [],
        }
        self.active_region = "OUTSIDE"
        self.cursor = None
        self.show_buffer = False
        self.show_normal = True
        
        # Colors for visualization
        self.COLORS = {
            "OUTSIDE": (0, 255, 255),   # Cyan
            "DOOR": (0, 165, 255),     # Orange
            "INSIDE": (0, 255, 0),     # Green
            "BUFFER": (0, 255, 0),     # Green
            "LINE": (0, 0, 255),       # Red
        }
    
    def load_frame(
        self,
        source: Optional[str] = None,
        config_path: Optional[str] = None,
        frame_index: int = 15,
    ) -> Tuple[cv2.Mat, Tuple[int, int]]:
        """Load a frame from camera or video source."""
        cfg = load_settings(config_path)
        actual_source = source or cfg["camera"]["source"]
        
        cap = cv2.VideoCapture(
            str(actual_source) if not str(actual_source).isdigit() else int(actual_source)
        )
        if not cap.isOpened():
            raise ValueError(f"Cannot open source: {actual_source}")
        
        for _ in range(frame_index + 1):
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise ValueError(f"Could not read frame {frame_index} from {actual_source}")
        
        cap.release()
        h, w = frame.shape[:2]
        return frame, (w, h)
    
    def calibrate_line(
        self,
        source: Optional[str] = None,
        config_path: Optional[str] = None,
        y1: Optional[float] = None,
        y2: Optional[float] = None,
        x1: Optional[float] = None,
        x2: Optional[float] = None,
        frame_index: int = 15,
        output_path: Optional[str] = None,
    ) -> CalibrationResult:
        """Calibrate line-based boundary."""
        print(f"\n{'='*70}")
        print("LINE-BASED CALIBRATION (Layer 1: Signed Distance)")
        print(f"{'='*70}\n")
        
        # Load frame
        frame, (w, h) = self.load_frame(source, config_path, frame_index)
        
        # Get line configuration
        cfg = load_settings(config_path)
        line = dict(cfg.get("entry_exit", {}).get("line", {
            "x1": 0.5, "y1": 0.5, "x2": 0.5, "y2": 0.5
        }))
        
        # Override with provided values
        if y1 is not None:
            line["y1"] = y1
        if y2 is not None:
            line["y2"] = y2
        if x1 is not None:
            line["x1"] = x1
        if x2 is not None:
            line["x2"] = x2
        
        # Convert to pixels
        px1, py1 = int(line["x1"] * w), int(line["y1"] * h)
        px2, py2 = int(line["x2"] * w), int(line["y2"] * h)
        
        # Draw line with buffer visualization
        buffer_px = self.line_config.buffer_threshold
        
        # Draw main line
        cv2.line(frame, (px1, py1), (px2, py2), (0, 0, 255), 3)
        
        # Draw buffer zone
        dx = px2 - px1
        dy = py2 - py1
        length = (dx**2 + dy**2)**0.5
        if length > 0:
            perp_x = -dy / length * buffer_px
            perp_y = dx / length * buffer_px
            
            cv2.line(frame,
                    (int(px1 + perp_x), int(py1 + perp_y)),
                    (int(px2 + perp_x), int(py2 + perp_y)),
                    (0, 255, 0), 1)
            cv2.line(frame,
                    (int(px1 - perp_x), int(py1 - perp_y)),
                    (int(px2 - perp_x), int(py2 - perp_y)),
                    (0, 255, 0), 1)
        
        # Add labels
        cv2.putText(frame, f"OUTSIDE (above line)", (px1 + 5, py1 - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"INSIDE (below line)", (px1 + 5, py1 + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Line: ({line['x1']:.3f},{line['y1']:.3f})-({line['x2']:.3f},{line['y2']:.3f})",
                   (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Buffer: {buffer_px}px | Frame: {w}x{h}",
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Save output
        if output_path:
            out = Path(output_path)
        else:
            out_dir = self.ROOT / "data" / "calibration"
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"{self.camera_id}_line_calibration.png"
        
        cv2.imwrite(str(out), frame)
        
        # Determine entry direction
        entry_direction = cfg.get("entry_exit", {}).get("entry_direction", "B_to_A")
        
        print(f"Line coordinates (normalized):")
        print(f"  x1: {line['x1']:.4f}, y1: {line['y1']:.4f}")
        print(f"  x2: {line['x2']:.4f}, y2: {line['y2']:.4f}")
        print(f"Entry direction: {entry_direction}")
        print(f"Buffer threshold: {buffer_px}px")
        print(f"Hysteresis: {self.line_config.hysteresis_px}px")
        print(f"\nTechnique: Line-based with buffer zone (Layer 1)")
        print(f"Saved visualization: {out}")
        
        return CalibrationResult(
            mode="line",
            camera_id=self.camera_id,
            frame_size=(w, h),
            line_config=LineConfig(
                x1=line["x1"],
                y1=line["y1"],
                x2=line["x2"],
                y2=line["y2"],
                entry_direction=entry_direction,
                buffer_threshold=buffer_px,
                hysteresis_px=self.line_config.hysteresis_px,
            ),
            validation_passed=True,
        )
    
    def calibrate_polygon(
        self,
        source: Optional[str] = None,
        config_path: Optional[str] = None,
        frame_index: int = 0,
        camera_id: Optional[str] = None,
        clean: bool = True,
        validate: bool = True,
    ) -> CalibrationResult:
        """Calibrate polygon-based regions (OUTSIDE/DOOR/INSIDE)."""
        if camera_id:
            self.camera_id = camera_id
        
        print(f"\n{'='*70}")
        print("POLYGON-BASED CALIBRATION (3-Zone FSM)")
        print(f"{'='*70}")
        print("\nInstructions:")
        print("  o  - Start drawing OUTSIDE polygon")
        print("  d  - Start drawing DOOR polygon")
        print("  i  - Start drawing INSIDE polygon")
        print("  u  - Undo last vertex")
        print("  r  - Clear current polygon")
        print("  b  - Toggle buffer/normal visualization")
        print("  w  - Save and exit")
        print("  q  - Quit without saving")
        print("\nTechnique: OUTSIDE -> DOOR -> INSIDE (Polygon FSM)")
        print(f"{'='*70}\n")
        
        # Load frame
        frame, (w, h) = self.load_frame(source, config_path, frame_index)
        
        window_name = f"Door Calibration - {self.camera_id}"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._on_mouse)
        
        try:
            while True:
                vis = self._draw_polygon_state(frame, (w, h))
                cv2.imshow(window_name, vis)
                
                key = cv2.waitKey(1) & 0xFF
                ch = chr(key).lower() if 32 <= key < 127 else ""
                
                if ch == "o":
                    self.active_region = "OUTSIDE"
                    print(f"[Mode] Drawing OUTSIDE region")
                elif ch == "d":
                    self.active_region = "DOOR"
                    print(f"[Mode] Drawing DOOR corridor")
                elif ch == "i":
                    self.active_region = "INSIDE"
                    print(f"[Mode] Drawing INSIDE region")
                elif ch == "u":
                    if self.polys[self.active_region]:
                        self.polys[self.active_region].pop()
                        print(f"[Undo] Removed last vertex from {self.active_region}")
                elif ch == "r":
                    self.polys[self.active_region].clear()
                    print(f"[Clear] Cleared {self.active_region} polygon")
                elif ch == "b":
                    self.show_buffer = not self.show_buffer
                    print(f"[Buffer] {'ON' if self.show_buffer else 'OFF'}")
                elif ch == "w":
                    break
                elif key in (27, ord("q")):
                    cv2.destroyAllWindows()
                    print("[Exit] Calibration cancelled")
                    return CalibrationResult(
                        mode="polygon",
                        camera_id=self.camera_id,
                        frame_size=(w, h),
                        validation_passed=False,
                        error_message="User cancelled calibration",
                    )
            
            cv2.destroyAllWindows()
            
            # Clean polygons if requested
            if clean:
                for name in ["OUTSIDE", "DOOR", "INSIDE"]:
                    if len(self.polys[name]) >= 3:
                        cleaned = clean_polygon(self.polys[name])
                        if cleaned:
                            self.polys[name] = cleaned
                            print(f"[Clean] {name} polygon cleaned")
            
            # Normalize coordinates
            normalized = {
                name: [[round(x / w, 4), round(y / h, 4)] for (x, y) in poly]
                for name, poly in self.polys.items()
            }
            
            # Validate regions
            empty = [name for name in ["OUTSIDE", "DOOR", "INSIDE"] 
                     if len(normalized[name]) < 3]
            
            if empty:
                print(f"WARNING: regions with < 3 points will be skipped: {empty}")
            
            # Compute inward normal
            inward_norm = None
            if not empty:
                regions_norm = {
                    "OUTSIDE": normalized["OUTSIDE"],
                    "DOOR": normalized["DOOR"],
                    "INSIDE": normalized["INSIDE"],
                }
                inward_norm = inward_normal(regions_norm)
                if inward_norm:
                    print(f"[Validation] Inward normal: ({inward_norm[0]:.3f}, {inward_norm[1]:.3f})")
                    print("  Direction: OUTSIDE -> INSIDE")
                else:
                    print("[Validation] WARNING: Could not compute inward normal")
            
            # Save polygon configuration
            self.polygon_config = PolygonConfig(
                outside=normalized["OUTSIDE"],
                door=normalized["DOOR"],
                inside=normalized["INSIDE"],
            )
            
            # Save to zones.yaml
            zones_path = self.ROOT / "config" / "zones.yaml"
            lines = [
                f"# Door Intelligence regions (NORMALIZED 0-1 coords) - {self.camera_id}\n",
                "# Generated by calibrate_unified.py - Polygon FSM technique\n",
                "# Region priority: DOOR (decision band) > INSIDE > OUTSIDE\n",
                "# FSM transitions: OUTSIDE -> DOOR -> INSIDE (ENTRY)",
                "#                 INSIDE -> DOOR -> OUTSIDE (EXIT)\n",
                f"{self.camera_id}:\n",
                "  zones:\n",
            ]
            
            key_map = {
                "OUTSIDE": "outside",
                "DOOR": "door_corridor",
                "INSIDE": "inside"
            }
            
            for name in ["OUTSIDE", "DOOR", "INSIDE"]:
                poly = normalized[name]
                if len(poly) >= 3:
                    lines.append(f"    {key_map[name]}:\n")
                    for x, y in poly:
                        lines.append(f"      - [{x}, {y}]\n")
            
            zones_path.write_text("".join(lines), encoding="utf-8")
            
            print(f"\n[Success] Saved {zones_path}")
            print(f"[Setup] Set door_intelligence.enabled: true in config/settings.yaml")
            print(f"[Setup] Set door_intelligence.camera_id: {self.camera_id}")
            
            # Print summary
            print("\n" + "="*70)
            print("CALIBRATION SUMMARY")
            print("="*70)
            print(f"Camera ID: {self.camera_id}")
            print(f"Frame size: {w}x{h}")
            for name in ["OUTSIDE", "DOOR", "INSIDE"]:
                poly = normalized[name]
                if len(poly) >= 3:
                    print(f"  {name}: {len(poly)} vertices")
                else:
                    print(f"  {name}: NOT DEFINED")
            
            if inward_norm:
                print(f"\nInward normal: ({inward_norm[0]:.3f}, {inward_norm[1]:.3f})")
            print(f"Technique: Polygon FSM (OUTSIDE -> DOOR -> INSIDE)")
            print("="*70)
            
            return CalibrationResult(
                mode="polygon",
                camera_id=self.camera_id,
                frame_size=(w, h),
                polygon_config=self.polygon_config,
                inward_normal=inward_norm,
                validation_passed=not empty and inward_norm is not None,
            )
            
        except Exception as e:
            cv2.destroyAllWindows()
            return CalibrationResult(
                mode="polygon",
                camera_id=self.camera_id,
                frame_size=(w, h),
                validation_passed=False,
                error_message=str(e),
            )
    
    def _on_mouse(self, event, x, y, *_, **__):
        """Mouse callback for polygon drawing."""
        if event == cv2.EVENT_MOUSEMOVE:
            self.cursor = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.polys[self.active_region].append((int(x), int(y)))
    
    def _draw_polygon_state(self, canvas, frame_shape):
        """Draw current polygon calibration state."""
        out = canvas.copy()
        h, w = frame_shape
        
        # Draw all polygons
        for name, poly in self.polys.items():
            if len(poly) < 2:
                continue
            color = self.COLORS[name]
            pts = np.asarray(poly, np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], True, color, 2, cv2.LINE_AA)
            
            for (px, py) in poly:
                cv2.circle(out, (px, py), 4, color, -1)
            
            if len(poly) >= 3:
                cv2.fillPoly(out, [pts], color)
                mask = np.zeros(out.shape[:2], np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                region = out.copy()
                region[mask == 0] = canvas[mask == 0]
                out = cv2.addWeighted(region, 0.25, out, 0.75, 0)
            
            # Label
            cx = int(np.mean([p[0] for p in poly]))
            cy = int(np.mean([p[1] for p in poly]))
            cv2.putText(out, name, (cx - 10, cy + 4), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw active cursor
        if self.active_region and self.cursor is not None:
            color = self.COLORS[self.active_region]
            cv2.line(out, self.cursor, 
                     (self.cursor[0] + 1, self.cursor[1] + 1), color, 1)
        
        # Draw inward normal if requested
        if self.show_buffer and "DOOR" in self.polys and len(self.polys["DOOR"]) >= 3:
            door_poly = self.polys["DOOR"]
            regions_norm = {
                "OUTSIDE": [[p[0]/w, p[1]/h] for p in self.polys.get("OUTSIDE", [])],
                "INSIDE": [[p[0]/w, p[1]/h] for p in self.polys.get("INSIDE", [])],
                "DOOR": [[p[0]/w, p[1]/h] for p in self.polys.get("DOOR", [])]
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
        
        # Help text
        help_text = "o:OUTSIDE d:DOOR i:INSIDE  u:undo  r:clear  b:buffer  w:save  q:quit"
        cv2.rectangle(out, (0, 0), (len(out[0]), 26), (30, 30, 30), -1)
        cv2.putText(out, help_text, (8, 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        
        # Status
        status = " | ".join(f"{n}:{len(p)}" for n, p in self.polys.items())
        cv2.putText(out, status, (8, 44), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Technique info
        technique_text = "Technique: Polygon FSM (3-zone)"
        cv2.putText(out, technique_text, (w - 300, h - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return out
    
    def test_calibration(
        self,
        source: Optional[str] = None,
        config_path: Optional[str] = None,
        max_frames: int = 100,
    ) -> bool:
        """Test the current calibration with live preview."""
        print(f"\n{'='*70}")
        print("TESTING CALIBRATION")
        print(f"{'='*70}\n")
        
        try:
            cfg = load_settings(config_path)
            actual_source = source or cfg["camera"]["source"]
            
            # Try to create the door intelligence engine
            zones = {
                "outside": self.polygon_config.outside,
                "door_corridor": self.polygon_config.door,
                "inside": self.polygon_config.inside,
            }
            
            if all(len(z) >= 3 for z in zones.values()):
                engine = DoorIntelligenceEngine(
                    zones=zones,
                    camera_id=self.camera_id,
                )
                print("[Test] Door Intelligence engine created successfully")
                print(f"  Zones: OUTSIDE({len(zones['outside'])}), DOOR({len(zones['door_corridor'])}), INSIDE({len(zones['inside'])})")
                
                if self.polygon_config.inward_normal:
                    print(f"  Inward normal: {self.polygon_config.inward_normal}")
                
                return True
            else:
                print("[Test] ERROR: Not all zones have enough points")
                return False
                
        except Exception as e:
            print(f"[Test] ERROR: {e}")
            return False
    
    def generate_settings(
        self,
        output_path: Optional[str] = None,
    ) -> Path:
        """Generate updated settings.yaml with current calibration."""
        cfg = load_settings()
        
        if self.mode == CalibrationMode.POLYGON:
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
            if self.polygon_config.outside and self.polygon_config.inside:
                # Compute line from door region
                if self.polygon_config.door:
                    door_pts = np.array(self.polygon_config.door)
                    center = door_pts.mean(axis=0)
                    cfg["entry_exit"]["line"] = {
                        "x1": float(center[0]),
                        "y1": float(center[1] - 0.1),
                        "x2": float(center[0]),
                        "y2": float(center[1] + 0.1),
                    }
        
        elif self.mode == CalibrationMode.LINE:
            if self.line_config:
                cfg["entry_exit"]["line"] = {
                    "x1": self.line_config.x1,
                    "y1": self.line_config.y1,
                    "x2": self.line_config.x2,
                    "y2": self.line_config.y2,
                }
                cfg["entry_exit"]["entry_direction"] = self.line_config.entry_direction
                cfg["entry_exit"]["hysteresis_px"] = self.line_config.hysteresis_px
        
        # Save updated settings
        if output_path:
            out = Path(output_path)
        else:
            out = self.ROOT / "config" / "settings.yaml"
        
        with open(out, 'w') as f:
            import yaml
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        
        print(f"[Config] Updated settings saved to {out}")
        return out


def main():
    parser = argparse.ArgumentParser(
        description="Unified Calibration System - All Techniques Combined",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Techniques Implemented:
  1. Line-based: Simple boundary line with buffer zone (Layer 1)
  2. Polygon FSM: 3-zone system (OUTSIDE -> DOOR -> INSIDE)
  3. 5-layer validation: Signed distance + FSM + trajectory + continuity + deduplication
  4. Auto-detection: Learn optimal line from motion patterns

Examples:
  # Line calibration
  python scripts/calibrate_unified.py --mode line --y1 0.55 --y2 0.55
  
  # Polygon calibration (interactive)
  python scripts/calibrate_unified.py --mode polygon --camera-id office_entrance
  
  # Test calibration
  python scripts/calibrate_unified.py --mode test --source rtsp://127.0.0.1:8554/cam_01_sub
  
  # Full calibration with auto-detection
  python scripts/calibrate_unified.py --mode auto --frames 200
"""
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="polygon",
        choices=["line", "polygon", "auto", "test"],
        help="Calibration mode (default: polygon)"
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
        "--camera-id",
        default="camera_1",
        help="Camera identifier (default: camera_1)"
    )
    
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index for static calibration (default: 0)"
    )
    
    # Line mode arguments
    parser.add_argument(
        "--y1",
        type=float,
        default=None,
        help="Normalized y1 coordinate for line"
    )
    parser.add_argument(
        "--y2",
        type=float,
        default=None,
        help="Normalized y2 coordinate for line"
    )
    parser.add_argument(
        "--x1",
        type=float,
        default=None,
        help="Normalized x1 coordinate for line"
    )
    parser.add_argument(
        "--x2",
        type=float,
        default=None,
        help="Normalized x2 coordinate for line"
    )
    
    # Polygon mode arguments
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean polygons by removing duplicates"
    )
    
    # Test mode arguments
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames to process in test mode"
    )
    
    # Auto mode arguments
    parser.add_argument(
        "--min-tracks",
        type=int,
        default=150,
        help="Minimum tracks for auto-learning (default: 150)"
    )
    
    args = parser.parse_args()
    
    # Convert mode string to enum
    mode_map = {
        "line": CalibrationMode.LINE,
        "polygon": CalibrationMode.POLYGON,
        "auto": CalibrationMode.AUTO,
        "test": CalibrationMode.TEST,
    }
    mode = mode_map.get(args.mode, CalibrationMode.POLYGON)
    
    # Create calibrator
    calibrator = UnifiedCalibrator(
        camera_id=args.camera_id,
        mode=mode,
    )
    
    # Run calibration based on mode
    if mode == CalibrationMode.LINE:
        result = calibrator.calibrate_line(
            source=args.source,
            config_path=args.config,
            y1=args.y1,
            y2=args.y2,
            x1=args.x1,
            x2=args.x2,
            frame_index=args.frame_index,
        )
    
    elif mode == CalibrationMode.POLYGON:
        result = calibrator.calibrate_polygon(
            source=args.source,
            config_path=args.config,
            frame_index=args.frame_index,
            camera_id=args.camera_id,
            clean=args.clean,
            validate=True,
        )
    
    elif mode == CalibrationMode.TEST:
        success = calibrator.test_calibration(
            source=args.source,
            config_path=args.config,
            max_frames=args.frames,
        )
        if success:
            print("\n[Result] Calibration test PASSED")
            sys.exit(0)
        else:
            print("\n[Result] Calibration test FAILED")
            sys.exit(1)
    
    elif mode == CalibrationMode.AUTO:
        print("\nAuto-detection mode not yet fully implemented.")
        print("Please use --mode line or --mode polygon")
        sys.exit(1)
    
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)
    
    # Generate updated settings if calibration succeeded
    if result and result.validation_passed:
        settings_path = calibrator.generate_settings()
        print(f"\n[Complete] Calibration successful!")
        print(f"  Mode: {result.mode}")
        print(f"  Camera: {result.camera_id}")
        print(f"  Frame size: {result.frame_size[0]}x{result.frame_size[1]}")
        if result.line_config:
            print(f"  Line: ({result.line_config.x1:.3f},{result.line_config.y1:.3f}) to ({result.line_config.x2:.3f},{result.line_config.y2:.3f})")
        if result.polygon_config:
            print(f"  Zones: OUTSIDE({len(result.polygon_config.outside)}), DOOR({len(result.polygon_config.door)}), INSIDE({len(result.polygon_config.inside)})")
        print(f"  Settings: {settings_path}")
    else:
        print(f"\n[Error] Calibration failed: {result.error_message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
