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

## Ubuntu Setup (Copy & Run)

### Step 1: Install system dependencies
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip ffmpeg libgl1-mesa-glx libglib2.0-0 net-tools
```

### Step 2: Clone and setup
```bash
git clone https://github.com/ahmedA-gif/Face-recognition-ubunto.git
cd Face-recognition-ubunto
python3.12 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Python dependencies
```bash
pip install -r requirements.txt
pip install faiss-cpu
```

### Step 4: Download models
```bash
# YOLO (person detection)
mkdir -p models/yolo
python3 -c "from ultralytics import YOLO; m=YOLO('yolo11n.pt'); m.export(format='onnx',imgsz=416,simplify=True,dynamic=False)"
mv yolo11n.pt models/yolo/
mv yolo11n.onnx models/yolo/

# InsightFace buffalo_l (face detection + recognition)
mkdir -p models/face/models
python3 -c "from insightface.app import FaceAnalysis; app=FaceAnalysis(name='buffalo_l',root='models/face',providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1,det_size=(640,640))"

# Verify all models
python3 scripts/check_models.py
```

### Step 5: Setup network for CCTV camera
```bash
# Check your network interfaces
ip addr show

# Find which interface is connected to camera (UP state, not wl/wifi)
# Example: enxc8a362908ac9 or eno1

# Assign IP to camera interface (replace INTERFACE_NAME with yours)
sudo ip addr add 192.168.2.100/24 dev INTERFACE_NAME
sudo ip link set INTERFACE_NAME up

# Test camera connectivity
ping 192.168.2.112
```

### Step 6: Configure camera in go2rtc.yaml
```bash
# Edit go2rtc.yaml in project root
nano go2rtc.yaml
```

Make sure it has your camera IP:
```yaml
streams:
  cam_01:
    - rtsp://admin:admin1234@192.168.2.112:554/cam/realmonitor?channel=1&subtype=0
  cam_01_sub:
    - rtsp://admin:admin1234@192.168.2.112:554/cam/realmonitor?channel=1&subtype=1

api:
  listen: ":1984"

rtsp:
  listen: ":8554"
```

### Step 7: Test RTSP before go2rtc
```bash
# Test direct camera access
ffplay rtsp://admin:admin1234@192.168.2.112:554/cam/realmonitor?channel=1&subtype=1
```

### Step 8: Start go2rtc and run
```bash
# Terminal 1: Start go2rtc (auto-downloads on first run)
./scripts/start_go2rtc.sh

# Terminal 2: Run attendance system
source .venv/bin/activate
python3 scripts/run_video.py
```

### Step 9: Enroll faces (first time)
```bash
mkdir -p data/faces_gallery/YourName
cp /path/to/your/photo.jpg data/faces_gallery/YourName/
python3 scripts/enroll_faces.py
```

## Quick Test (Full Commands)

```bash
# One-time setup
git clone https://github.com/ahmedA-gif/Face-recognition-ubunto.git
cd Face-recognition-ubunto
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install faiss-cpu

# Download models
mkdir -p models/yolo models/face/models
python3 -c "from ultralytics import YOLO; m=YOLO('yolo11n.pt'); m.export(format='onnx',imgsz=416,simplify=True,dynamic=False)"
mv yolo11n.pt models/yolo/ && mv yolo11n.onnx models/yolo/
python3 -c "from insightface.app import FaceAnalysis; app=FaceAnalysis(name='buffalo_l',root='models/face',providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1,det_size=(640,640))"

# Setup network (replace INTERFACE_NAME)
sudo ip addr add 192.168.2.100/24 dev INTERFACE_NAME
sudo ip link set INTERFACE_NAME up
ping 192.168.2.112

# Test camera
ffplay rtsp://admin:admin1234@192.168.2.112:554/cam/realmonitor?channel=1&subtype=1

# Run
./scripts/start_go2rtc.sh
python3 scripts/run_video.py
```

## Model Downloads

| Model | Size | Path | Source |
|-------|------|------|--------|
| YOLOv11n (PT) | ~5.6 MB | `models/yolo/yolo11n.pt` | Ultralytics |
| YOLOv11n (ONNX) | ~10.2 MB | `models/yolo/yolo11n.onnx` | Exported from PT |
| buffalo_l (5 ONNX) | ~330 MB | `models/face/models/buffalo_l/` | InsightFace |

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

## Face Enrollment

### Folder Structure
```
data/faces_gallery/
    Ahmed/
        photo1.jpg
        photo2.png
    Sara/
        headshot.jpg
```

### Commands
```bash
# Single person — one photo
python3 scripts/enroll_faces.py --name Ahmed --image path/to/photo.jpg

# Bulk enroll — all people in data/faces_gallery/
python3 scripts/enroll_faces.py

# Re-enroll from scratch (wipes DB first)
python3 scripts/enroll_faces.py --clear

# List enrolled people
python3 scripts/enroll_faces.py --list
```

## Test Commands
```bash
# Quick test (video file, no display)
python3 scripts/run_video.py --source data/test_video.mp4 --max-frames 160 --no-display

# Full test with display
python3 scripts/run_video.py --source data/test_video.mp4

# CCTV live (start go2rtc first)
python3 scripts/run_video.py
```

## Project Structure
```text
Face-recognition-ubunto/
├── config/
│   ├── settings.yaml         # camera, thresholds, line config
│   └── go2rtc.yaml           # RTSP streaming proxy config
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

| Error | Fix |
|-------|-----|
| `Cannot open camera source: 2` | Don't use camera index. Use go2rtc with RTSP URL |
| `method DESCRIBE failed: 404` | go2rtc can't reach camera. Check IP and network |
| `Connection refused tcp://127.0.0.1:8554` | go2rtc not running. Run `./scripts/start_go2rtc.sh` |
| `No face detection` | Ensure face models downloaded, increase `min_face_px` |
| `Wrong entry/exit` | Adjust `entry_exit.line` and `entry_direction` in settings.yaml |
| `Slow performance` | Use sub stream (`subtype=1`), reduce `yolo_imgsz` to 320 |

### Network Debug
```bash
# Check your IP
ip addr show

# Find camera on network
nmap -p 554 192.168.2.0/24

# Check go2rtc logs
cat data/go2rtc.log

# Test camera directly (without go2rtc)
ffplay rtsp://admin:admin1234@192.168.2.112:554/cam/realmonitor?channel=1&subtype=1
```
