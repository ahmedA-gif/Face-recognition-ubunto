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
git clone https://github.com/ahmedA-gif/Face-recognition-ubunto.git
cd Face-recognition-ubunto
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt
pip install faiss-cpu

# 4. Download models
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

# 5. Configure camera
# Edit config/settings.yaml:
#   camera.source: "rtsp://admin:password@192.168.1.112:554/cam/realmonitor?channel=1&subtype=0"
#   camera.password: "your_password"
#   entry_exit.line: adjust to match your gate geometry

# 6. Run
./scripts/start_go2rtc.sh
python3 scripts/run_video.py

# 7. Enroll faces (first time)
mkdir -p data/faces_gallery/YourName
cp /path/to/your/photo.jpg data/faces_gallery/YourName/
python3 scripts/enroll_faces.py
```

## Model Downloads

| Model | Size | Path | Source |
|-------|------|------|--------|
| YOLOv11n (PT) | ~5.6 MB | `models/yolo/yolo11n.pt` | Ultralytics |
| YOLOv11n (ONNX) | ~10.2 MB | `models/yolo/yolo11n.onnx` | Exported from PT |
| buffalo_l (5 ONNX) | ~330 MB | `models/face/models/buffalo_l/` | InsightFace |

```bash
# Download all models (run once after pip install)
python3 -c "
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import os

# YOLO
os.makedirs('models/yolo', exist_ok=True)
m = YOLO('yolo11n.pt')
m.export(format='onnx', imgsz=416, simplify=True, dynamic=False)
import shutil; shutil.move('yolo11n.pt','models/yolo/'); shutil.move('yolo11n.onnx','models/yolo/')

# InsightFace buffalo_l
os.makedirs('models/face/models', exist_ok=True)
app = FaceAnalysis(name='buffalo_l', root='models/face', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640,640))
print('All models downloaded.')
"
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

## Face Enrollment

Enroll face photos so the system can recognize people.

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

# Quick enroll (simpler script, same bulk behavior)
python3 scripts/enroll_face.py
```

### Verify Enrollment

```bash
# Check gallery status
python3 scripts/enroll_faces.py --list

# Test recognition on a video
python3 scripts/run_video.py --source data/test_video.mp4 --max-frames 160 --no-display
```

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
