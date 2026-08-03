# Attendance System (Ubuntu Linux)

Face recognition + person tracking attendance system with auto check-in/check-out.

## Features

- Person detection (YOLO nano)
- Face detection + recognition (InsightFace ArcFace)
- Tracking (ByteTrack)
- Entry/exit events via virtual boundary line
- Auto attendance: check-in/check-out with Late/Early status
- Identity fusion (re-identification across track fragments)
- Guest IDs for unknown persons

## Ubuntu Setup

```bash
# 1. Install system dependencies
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip ffmpeg libgl1-mesa-glx libglib2.0-0

# 2. Clone and setup
git clone https://github.com/ahmedA-gif/attendance-system.git
cd attendance-system
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt
pip install faiss-cpu

# 4. Download models (if not included)
# YOLO: models/yolo/yolo11n.pt + yolo11n.onnx
# InsightFace buffalo_l: auto-downloads on first run

# 5. Configure camera
# Edit config/settings.yaml:
#   camera.source: "rtsp://admin:password@192.168.1.112:554/cam/realmonitor?channel=1&subtype=0"
#   camera.password: "your_password"
#   entry_exit.line: adjust to match your gate geometry

# 6. Run
./scripts/start_go2rtc.sh
python3 scripts/run_video.py
```

## Gate Line Configuration

The boundary line is configured in `config/settings.yaml`:

```yaml
entry_exit:
  line:
    x1: 0.25    # left edge (normalized 0-1)
    y1: 0.82    # top edge
    x2: 0.75    # right edge
    y2: 0.82    # bottom edge
  entry_direction: "B_to_A"  # B→A = entry (outside → inside)
```

- Adjust `x1/y1/x2/y2` to match your gate/door threshold
- `entry_direction`: which crossing direction = "entry"

## Test Commands

```bash
# Quick test (no display, 160 frames)
python3 scripts/run_video.py --source data/test_video.mp4 --max-frames 160 --no-display

# Full test with display
python3 scripts/run_video.py --source data/test_video.mp4

# CCTV live
python3 scripts/run_video.py
```

## Project Structure

```text
attendance-system/
├── config/settings.yaml       # camera, thresholds, line config
├── src/
│   ├── capture/               # camera stream
│   ├── detection/             # YOLO person detection
│   ├── tracking/              # ByteTrack + identity fusion
│   ├── recognition/           # face detection + ArcFace
│   ├── events/                # entry/exit + dynamic boundary
│   ├── reasoning/             # spatial-temporal reasoning
│   ├── attendance/            # check-in/check-out manager
│   ├── overlay/               # live visualization
│   └── utils/
├── scripts/                   # run scripts
├── tests/                     # unit tests
└── data/
    ├── faces_gallery/         # enrolled face images
    ├── db/                    # SQLite databases
    └── snapshots/             # event crops
```

## Troubleshooting

- **Camera not connecting**: Check IP, credentials, and interface (`ip addr show`)
- **No face detection**: Ensure face models downloaded, increase `min_face_px`
- **Wrong entry/exit**: Adjust `entry_exit.line` and `entry_direction` in settings.yaml
- **Slow performance**: Use sub stream (`subtype=0`), reduce `yolo_imgsz` to 320
