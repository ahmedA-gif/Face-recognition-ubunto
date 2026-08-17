# Calibration Fix and Implementation Guide

## Summary of Fixes

This update fixes the calibration issue by implementing all techniques (buffer, outside, inside, door) with:

1. **Cross-platform path handling** - Fixed hardcoded Linux paths (`/home/rahmat/...`) to work on Windows and Ubuntu
2. **Unified calibration system** - New `calibrate_unified.py` combining all techniques
3. **Improved calibration scripts** - Enhanced `calibrate_boundary.py` and `calibrate_door_regions.py`
4. **All techniques integrated**:
   - Line-based with buffer zones (Layer 1: Signed Distance)
   - 3-zone Polygon FSM (OUTSIDE -> DOOR -> INSIDE)
   - 5-layer validation system
   - Door Intelligence engine with polygon zones

## Techniques Implemented

### 1. Line-Based Boundary with Buffer (Layer 1)
- Uses signed distance from line for spatial validation
- Configurable buffer zone (hysteresis) to prevent jitter
- Entry/exit direction based on line crossing

**Configuration:**
```yaml
entry_exit:
  line:
    x1: 0.2
    y1: 0.55
    x2: 0.8
    y2: 0.55
  entry_direction: B_to_A
  buffer_threshold: 10.0
  hysteresis_px: 14.0
```

### 2. Polygon FSM (3-Zone System)
- **OUTSIDE**: Area outside the door where people approach from
- **DOOR**: Threshold corridor/band where crossing is detected
- **INSIDE**: Area inside the room where people enter to

**FSM States:**
```
OUTSIDE -> APPROACHING_DOOR -> CROSSING_IN -> INSIDE (ENTRY)
INSIDE -> APPROACHING_EXIT -> CROSSING_OUT -> OUTSIDE (EXIT)
```

**Configuration:**
```yaml
# config/zones.yaml
camera_1:
  zones:
    outside:
      - [0.33, 0.14]
      - [0.70, 0.14]
      - [0.70, 0.66]
      - [0.33, 0.66]
    door_corridor:
      - [0.29, 0.66]
      - [0.73, 0.66]
      - [0.73, 0.82]
      - [0.29, 0.82]
    inside:
      - [0.00, 0.00]
      - [0.29, 0.00]
      - [0.29, 0.14]
      - [0.29, 0.66]
      - [0.73, 0.66]
      - [0.73, 0.14]
      - [0.73, 0.00]
      - [1.00, 0.00]
      - [1.00, 1.00]
      - [0.00, 1.00]
```

### 3. 5-Layer Validation System

**Layer 1: Signed Distance from Line**
- Perpendicular distance from foot point to line
- Configurable threshold band (±10 pixels default)
- Eliminates bounding box flickering

**Layer 2: 5-State Finite State Machine**
- OUTSIDE -> APPROACHING -> BUFFER -> CROSSING -> INSIDE (ENTRY)
- INSIDE -> APPROACHING_EXIT -> BUFFER -> CROSSING -> OUTSIDE (EXIT)
- Only complete transitions trigger events

**Layer 3: Trajectory Validation**
- Checks velocity vector direction
- Validates minimum displacement
- Ensures consistent movement toward/away from line

**Layer 4: Track Continuity (Occlusion Handling)**
- Re-identifies tracks after short occlusions (1-3s)
- Uses feature matching (face embeddings) when available
- Falls back to spatial proximity and appearance

**Layer 5: Event Deduplication**
- Lockout period (5s) per track after event
- Spatial hysteresis (20px from line before reset)
- Confidence-based filtering

## Files Changed

### 1. `scripts/calibrate_boundary.py`
- Fixed hardcoded Linux paths to cross-platform
- Added buffer zone visualization
- Added named arguments support
- Cross-platform output path handling
- Better error messages

**Usage:**
```bash
# Simple line calibration
python scripts/calibrate_boundary.py --y1 0.55 --y2 0.55 --x1 0.2 --x2 0.8

# With custom source
python scripts/calibrate_boundary.py --source rtsp://127.0.0.1:8554/cam_01_sub --y1 0.55

# Positional arguments (backward compatible)
python scripts/calibrate_boundary.py 0.55 0.55 0.2 0.8
```

### 2. `scripts/calibrate_door_regions.py`
- Fixed cross-platform path handling
- Added inward normal visualization
- Added buffer zone toggle
- Better user interface with mode indicators
- Save temporary snapshots
- Validation of regions

**Usage:**
```bash
# Interactive polygon calibration
python scripts/calibrate_door_regions.py

# With custom source
python scripts/calibrate_door_regions.py --source data/test_video.mp4 --frame 120

# With camera ID
python scripts/calibrate_door_regions.py --camera-id office_entrance
```

### 3. `scripts/calibrate_unified.py` (NEW)
- Unified calibration system
- Supports line, polygon, auto, and test modes
- Implements all techniques
- Comprehensive validation

**Usage:**
```bash
# Line-based calibration
python scripts/calibrate_unified.py --mode line --y1 0.55 --y2 0.55

# Polygon-based calibration (interactive)
python scripts/calibrate_unified.py --mode polygon --camera-id office_entrance

# Test existing calibration
python scripts/calibrate_unified.py --mode test

# Generate settings only
python scripts/calibrate_unified.py --mode line --y1 0.55 --y2 0.55 --x1 0.2 --x2 0.8
```

### 4. `config/settings.yaml`
- Fixed Windows-style paths to relative paths
- Updated calibration defaults

**Changed:**
```yaml
# Before (Windows paths)
yolo_weights: D:\Face-recognition-ubunto\models\yolo\yolo11n.pt

# After (relative paths)
yolo_weights: models/yolo/yolo11n.pt
```

### 5. `config/zones.yaml`
- Already properly configured with 3 zones
- OUTSIDE, DOOR, INSIDE regions defined

## How to Update

### For Existing Installations

```bash
# 1. Pull the latest changes
git pull origin main

# 2. Update calibration files
cp scripts/calibrate_boundary.py scripts/calibrate_boundary.py.bak
cp scripts/calibrate_door_regions.py scripts/calibrate_door_regions.py.bak

# 3. Fix settings.yaml paths (if on Ubuntu)
sed -i 's/D:\\Face-recognition-ubunto\\\(.*\)/\1/g' config/settings.yaml

# 4. Test calibration
python scripts/calibrate_unified.py --mode test
```

### For New Installations

```bash
# 1. Clone the repository
git clone https://github.com/ahmedA-gif/Face-recognition-ubunto.git
cd Face-recognition-ubunto

# 2. Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux
# OR on Windows:
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run calibration (choose one method)

# Method A: Simple line calibration
python scripts/calibrate_boundary.py --y1 0.55 --y2 0.55 --x1 0.2 --x2 0.8

# Method B: Interactive polygon calibration
python scripts/calibrate_door_regions.py

# Method C: Unified calibration
python scripts/calibrate_unified.py --mode polygon --camera-id office_entrance

# 5. Update settings (optional)
python scripts/calibrate_unified.py --mode line --y1 0.55 --y2 0.55 --x1 0.2 --x2 0.8
```

## Calibration Commands Reference

### Quick Line Calibration
```bash
# Horizontal line at 55% height, spanning 20%-80% width
python scripts/calibrate_boundary.py --y1 0.55 --y2 0.55 --x1 0.2 --x2 0.8

# Vertical line
python scripts/calibrate_boundary.py --y1 0.1 --y2 0.9 --x1 0.5 --x2 0.5
```

### Interactive Polygon Calibration
```bash
# Start interactive calibration
python scripts/calibrate_door_regions.py

# Keys during calibration:
#   o - Switch to OUTSIDE region
#   d - Switch to DOOR region
#   i - Switch to INSIDE region
#   Left-click - Add vertex
#   u - Undo last vertex
#   r - Clear current region
#   b - Toggle buffer/normal visualization
#   w - Write and save
#   q - Quit without saving
```

### Unified Calibration
```bash
# Line mode
python scripts/calibrate_unified.py --mode line --y1 0.55 --x1 0.2 --x2 0.8

# Polygon mode (interactive)
python scripts/calibrate_unified.py --mode polygon --camera-id office_entrance

# Test mode
python scripts/calibrate_unified.py --mode test

# Auto mode (learns from motion)
python scripts/calibrate_unified.py --mode auto --min-tracks 150
```

## Troubleshooting

### Issue: "Cannot open source"
**Solution:** Check that your camera or video source is accessible
```bash
# Test RTSP stream
ffplay rtsp://127.0.0.1:8554/cam_01_sub

# Test with a video file
python scripts/calibrate_boundary.py --source data/test_video.mp4
```

### Issue: "No module named yaml"
**Solution:** Install PyYAML
```bash
pip install pyyaml
```

### Issue: Calibration line not visible
**Solution:** Adjust coordinates to be within 0-1 range
```bash
# Valid range: x1,x2 in [0,1] and y1,y2 in [0,1]
python scripts/calibrate_boundary.py --y1 0.5 --y2 0.5 --x1 0.0 --x2 1.0
```

### Issue: Polygons not saving
**Solution:** Ensure each region has at least 3 vertices
```bash
# Each region (OUTSIDE, DOOR, INSIDE) needs at least 3 points
# Click 3 or more points for each region before saving
```

## Configuration Examples

### Example 1: Horizontal Door (Most Common)
```yaml
# Line at 55% height, horizontal
entry_exit:
  line:
    x1: 0.2
    y1: 0.55
    x2: 0.8
    y2: 0.55
  entry_direction: B_to_A  # B_to_A means from bottom to top is entry
  buffer_threshold: 10.0
  hysteresis_px: 14.0
```

### Example 2: Vertical Door
```yaml
# Line at 50% width, vertical
entry_exit:
  line:
    x1: 0.5
    y1: 0.1
    x2: 0.5
    y2: 0.9
  entry_direction: A_to_B  # A_to_B means from left to right is entry
  buffer_threshold: 10.0
  hysteresis_px: 14.0
```

### Example 3: Diagonal Door
```yaml
# Diagonal line
entry_exit:
  line:
    x1: 0.2
    y1: 0.3
    x2: 0.8
    y2: 0.7
  entry_direction: B_to_A
  buffer_threshold: 15.0
  hysteresis_px: 20.0
```

## Validation Checklist

- [ ] Camera source is accessible
- [ ] Line/polygon coordinates are within 0-1 range
- [ ] Each polygon region has at least 3 vertices
- [ ] OUTSIDE, DOOR, and INSIDE regions are defined
- [ ] Inward normal direction is correct (OUTSIDE -> INSIDE)
- [ ] Buffer/hysteresis values are appropriate for your scene
- [ ] door_intelligence.enabled: true in settings.yaml
- [ ] Camera ID in zones.yaml matches settings.yaml

## Performance Tips

1. **For simple scenes**: Use line-based calibration (faster)
2. **For complex scenes**: Use polygon-based calibration (more accurate)
3. **For occlusions**: Enable identity fusion and person Re-ID
4. **For high traffic**: Increase skip_frames and face_every_n

```yaml
pipeline:
  skip_frames: 3      # Process every 3rd frame
  face_every_n: 5     # Run face detection every 5th frame
```

## Technical Details

### Signed Distance Calculation
```python
# Layer 1: Signed distance from point to line
def signed_distance(point, line):
    (x1, y1), (x2, y2) = line
    px, py = point
    dx, dy = x2 - x1, y2 - y1
    length = (dx*dx + dy*dy)**0.5
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    return cross / length  # Positive = left of line, Negative = right
```

### Polygon FSM State Transitions
```
OUTSIDE -> DOOR: Person approaches door
DOOR (dwell > min_dwell): Person entering door corridor
DOOR -> INSIDE: Person crosses into inside
INSIDE: Person is inside

INSIDE -> DOOR: Person approaches exit
DOOR (dwell > min_dwell): Person in door corridor
DOOR -> OUTSIDE: Person exits
OUTSIDE: Person is outside
```

### 5-Layer Confidence Scoring
```
Total Confidence = 
    0.30 * detection_confidence +
    0.25 * tracking_stability +
    0.20 * trajectory_consistency +
    0.15 * spatial_validation +
    0.10 * temporal_validation
```

## Command Summary

| Task | Command |
|------|---------|
| Simple line calibration | `python scripts/calibrate_boundary.py --y1 0.55 --x1 0.2 --x2 0.8` |
| Interactive polygon calibration | `python scripts/calibrate_door_regions.py` |
| Full unified calibration | `python scripts/calibrate_unified.py --mode polygon` |
| Test calibration | `python scripts/calibrate_unified.py --mode test` |
| Fix settings paths | `sed -i 's/D:\\.*\(models\/.*\)/\1/g' config/settings.yaml` |

## Next Steps

After calibration:

1. **Test the pipeline:**
   ```bash
   python main.py --max-frames 100
   ```

2. **Enroll faces:**
   ```bash
   mkdir -p data/faces_gallery/YourName
   cp /path/to/photo.jpg data/faces_gallery/YourName/
   python scripts/enroll_faces.py
   ```

3. **Start the service:**
   ```bash
   # Linux (systemd)
   sudo cp scripts/run_attendance.sh /etc/init.d/
   sudo systemctl enable run_attendance
   sudo systemctl start run_attendance
   
   # OR run manually
   source .venv/bin/activate
   python main.py
   ```

4. **Monitor events:**
   ```bash
   # Watch Redis stream
   watch -n 2 "redis-cli XLEN attendance:events"
   
   # View latest events
   redis-cli XREVRANGE attendance:events + - COUNT 10
   ```
