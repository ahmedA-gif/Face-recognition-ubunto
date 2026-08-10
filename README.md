# Face Recognition Attendance System (Ubuntu)

Real-time face recognition + person tracking attendance system with polygon-based door intelligence, Redis Streams event bus, and SQLite storage. Designed for a live Dahua CCTV camera on Ubuntu.

## Architecture

```
CCTV Camera (Dahua 192.168.2.112)
    │
    ▼
go2rtc (RTSP proxy, port 8554)
    │
    ▼
YOLOv11n (person detection) ─► ByteTrack (multi-object tracking)
    │                                   │
    ▼                                   ▼
InsightFace buffalo_s           Identity Fusion (Re-ID)
(face detection + embed)        (stitches track fragments)
    │                                   │
    └───────────┬───────────────────────┘
                ▼
    Door Intelligence Engine (polygon FSM)
        │           │           │
        ▼           ▼           ▼
    SQLite     Redis Streams   Overlay
    (events)   (live bus)     (live display)
        │           │
        ▼           ▼
    Attendance   Worker / API
    (check-in)
```

## Features

- **Person Detection**: YOLOv11n ONNX (CPU, ~416px, ~15ms/frame)
- **Face Recognition**: InsightFace ArcFace (buffalo_s pack, 320x320 det)
- **Multi-Object Tracking**: ByteTrack with identity fusion (Re-ID across track fragments)
- **Polygon Door Intelligence**: Three-zone FSM (OUTSIDE / DOOR / INSIDE) with calibrated polygons — replaces the legacy virtual boundary line
- **Redis Streams**: Real-time event bus (`attendance:events` stream) with JSONL fallback
- **SQLite Storage**: WAL-mode event log with deduplication by event UUID
- **Attendance Manager**: Auto check-in/check-out with Late/Early detection
- **Live Overlay**: Bounding boxes, face labels, zone polygons, HUD
- **Spatial-Temporal Reasoning**: U-turn/loitering detection, anti-tailgating alerts
- **83 Unit Tests**: Door FSM, geometry, gallery, reasoning, attendance

## Quick Start

### Step 1: Clone and setup

```bash
git clone https://github.com/ahmedA-gif/Face-recognition-ubunto.git
cd Face-recognition-ubunto
python3.12 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install dependencies

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip ffmpeg libgl1-mesa-glx libglib2.0-0 redis-server

pip install -r requirements.txt
pip install faiss-cpu redis
```

### Step 3: Start Redis

```bash
sudo systemctl start redis
# or
redis-server --daemonize yes

# Verify
redis-cli ping
# Should return: PONG
```

### Step 4: Download models

```bash
# YOLO (person detection)
mkdir -p models/yolo
python3 -c "from ultralytics import YOLO; m=YOLO('yolo11n.pt'); m.export(format='onnx',imgsz=416,simplify=True,dynamic=False)"
mv yolo11n.pt models/yolo/
mv yolo11n.onnx models/yolo/

# InsightFace buffalo_s (face detection + recognition)
mkdir -p models/face/models
python3 -c "from insightface.app import FaceAnalysis; app=FaceAnalysis(name='buffalo_s',root='models/face',providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1,det_size=(320,320))"

# Verify
python3 scripts/check_models.py
```

### Step 5: Setup camera network

Camera is connected via **direct Ethernet cable** (not WiFi).

```bash
# Check interfaces
ip addr show

# Assign IP to camera interface (replace INTERFACE_NAME)
sudo ip addr add 192.168.2.100/24 dev INTERFACE_NAME
sudo ip link set INTERFACE_NAME up

# Test
ping 192.168.2.112
```

### Step 6: Start go2rtc and run

```bash
# Terminal 1: go2rtc
./scripts/start_go2rtc.sh

# Terminal 2: Watch Redis events
watch -n 2 "redis-cli XLEN attendance:events"

# Terminal 3: Run pipeline
source .venv/bin/activate
python main.py
```

### Step 7: Enroll faces

```bash
mkdir -p data/faces_gallery/YourName
cp /path/to/photo.jpg data/faces_gallery/YourName/
python3 scripts/enroll_faces.py
```

## Polygon Door Zones

The system uses three polygon regions instead of a single virtual line. Calibrated from the camera frame in `config/zones.yaml`:

```yaml
camera_1:
  zones:
    outside:     # Visible outdoor area through door opening
      - [0.33, 0.14]
      - [0.70, 0.14]
      - [0.70, 0.66]
      - [0.33, 0.66]
    door_corridor:  # Threshold band at door ground
      - [0.29, 0.66]
      - [0.73, 0.66]
      - [0.73, 0.82]
      - [0.29, 0.82]
    inside:       # Room interior (concave polygon wrapping around door)
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

**Recalibrate**: run `python3 scripts/calibrate_door_regions.py` and click the three regions on a camera frame.

## Redis Streams

Events are published to `attendance:events` stream in real-time.

### Monitor commands

```bash
# Event count (live)
watch -n 2 "redis-cli XLEN attendance:events"

# All events
redis-cli XRANGE attendance:events - +

# Latest event only
redis-cli XREVRANGE attendance:events + - COUNT 1

# Stream info
redis-cli XINFO STREAM attendance:events
```

### Event payload

```json
{
  "event_id": "uuid",
  "camera_id": "office_entrance",
  "track_id": 1,
  "global_id": "Guest#001",
  "employee_id": "Ahmed",
  "event": "ENTRY",
  "confidence": 0.87,
  "fsm_path": ["OUTSIDE", "DOOR", "INSIDE"],
  "timestamp": "2026-08-08T13:09:53Z",
  "date": "2026-08-08",
  "time": "18:09:51"
}
```

### Worker (consume events)

```bash
python3 scripts/attendance_worker.py
```

## Configuration

All settings in `config/settings.yaml`:

| Section | Key | Description |
|---------|-----|-------------|
| `door_intelligence.enabled` | `true` | Use polygon FSM (recommended) |
| `door_intelligence.lock_after_event` | `false` | Allow same person re-entry |
| `door_intelligence.min_inside_frames` | `3` | Stable frames before ENTRY fires |
| `door_intelligence.min_outside_frames` | `3` | Stable frames before EXIT fires |
| `redis.enabled` | `true` | Publish events to Redis |
| `redis.stream` | `attendance:events` | Stream name |
| `attendance.shift_start` | `09:00` | Shift start for Late detection |
| `attendance.shift_end` | `17:00` | Shift end for Early exit detection |

## Project Structure

```
Face-recognition-ubunto/
├── main.py                          # Entry point
├── config/
│   ├── settings.yaml                # All pipeline settings
│   ├── zones.yaml                   # Door zone polygons
│   └── go2rtc.yaml                  # RTSP proxy config
├── src/
│   ├── capture/stream.py            # Camera stream reader
│   ├── detection/person_yolo.py     # YOLO person detector
│   ├── tracking/
│   │   ├── bytetrack.py             # ByteTrack tracker
│   │   ├── motion.py                # Kalman velocity smoother
│   │   └── identity_fusion.py       # Re-ID across tracks
│   ├── recognition/
│   │   ├── face_engine.py           # InsightFace detection + embedding
│   │   └── gallery.py               # FAISS face gallery
│   ├── events/
│   │   ├── door_intelligence.py     # Polygon FSM engine (3-zone)
│   │   ├── entry_exit.py            # Legacy line engine
│   │   ├── store.py                 # SQLite event store
│   │   └── redis_publisher.py       # Redis Streams publisher
│   ├── reasoning/spatial_temporal.py # U-turn, tailgate, window bias
│   ├── attendance/
│   │   ├── db.py                    # Attendance SQLite DB
│   │   └── manager.py               # Check-in/check-out logic
│   ├── overlay/draw.py              # Live visualization
│   └── utils/
│       ├── geometry.py              # Polygon math, point-in-polygon
│       ├── config.py                # Settings + zone loader
│       └── assign.py                # Face-to-track assignment
├── scripts/
│   ├── start_go2rtc.sh              # Start RTSP proxy
│   ├── enroll_faces.py              # Face enrollment
│   ├── calibrate_door_regions.py    # Zone polygon calibration
│   ├── attendance_worker.py         # Redis Stream consumer
│   └── run_attendance.sh            # 24/7 service wrapper
├── tests/                           # 83 unit tests
│   ├── test_door_intelligence.py    # FSM state machine tests
│   ├── test_fetch.py                # Store, gallery, perf tests
│   ├── test_geometry.py             # Polygon geometry tests
│   ├── test_reasoning_attendance.py # Reasoning + attendance tests
│   └── test_identity_fusion.py      # Re-ID tests
└── data/
    ├── faces_gallery/               # Enrolled face images
    ├── db/                          # SQLite databases
    └── snapshots/                   # Event face crops
```

## Known Issues

| Issue | Status |
|-------|--------|
| **Appearance Re-ID false merge** — HSV body-histogram re-ID can merge two different people wearing similar-coloured clothes (test showed cosine 0.81–0.85 for distinct persons). To be fixed: use a stricter threshold or a dedicated person-ReID model. | ⚠️ To fix later |
| **Guest counter burns IDs on fleeting tracks** — fixed: `_expire()` no longer calls `_new_guest()` for tracks that vanish before identity resolution. | ✅ Fixed |
| **Output buffered when run via systemd** — pipeline logs only streamed to journald after `-u` (unbuffered) flag was added to `run_attendance.sh`. | ✅ Fixed |

## Testing

```bash
source .venv/bin/activate

# Run all 83 tests
python -m pytest tests/ -v

# Run door intelligence tests only
python -m pytest tests/test_door_intelligence.py -v
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `redis-cli: command not found` | `sudo apt install redis-server && sudo systemctl start redis` |
| `redis-cli XLEN attendace:events` returns 0 | Typo! Use `attendance:events` (with 'n') |
| No events in Redis | Check `redis-cli ping` returns PONG, then check `watch -n 2 "redis-cli XLEN attendance:events"` |
| EXIT not firing | Ensure person walks through all 3 zones (OUTSIDE → DOOR → INSIDE) |
| `sqlite3.OperationalError: disk I/O error` | Delete WAL files: `rm -f data/db/*.db-wal data/db/*.db-shm` |
| No person detected | Camera frame must show a person walking through the door zone |
| Slow performance | Use sub stream (`cam_01_sub`), reduce `yolo_imgsz` to 320 |
| `Connection refused 127.0.0.1:8554` | go2rtc not running: `./scripts/start_go2rtc.sh` |

## Camera Setup

| Setting | Value |
|---------|-------|
| Camera | Dahua IPC |
| IP | 192.168.2.112 |
| Connection | Direct Ethernet (USB adapter) |
| Stream | Sub stream via go2rtc (`127.0.0.1:8554/cam_01_sub`) |
| Resolution | 704x576 (sub stream) |
| FPS | 15 |

## License

Personal project for office attendance monitoring.
