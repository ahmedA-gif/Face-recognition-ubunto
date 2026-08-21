# VisionAttend AI — CCTV Attendance System

Real-time face recognition + person tracking attendance system with a full web dashboard, crossing-line entry/exit detection, identity fusion, GPU auto-switch, and 24/7 operation. Connects to a live Dahua CCTV camera via RTSP.

## Features

- **Web Dashboard** — 12-page Flask UI (http://localhost:5000): Dashboard, Live Monitoring, Snapshots, Attendance, People, Cameras, Events, Tracking, Analytics, Database, System, Settings
- **Live Video Stream** — MJPEG streaming from pipeline to browser with bounding boxes, face labels, boundary line, HUD overlay
- **Crossing-Line Entry/Exit** — ObjectCounter-style line crossing with Shapely intersection, per-track dedup, configurable direction (upward/downward/leftward/rightward)
- **Face Recognition** — InsightFace ArcFace buffalo_s (320x320 det), FAISS gallery matching
- **Person Detection** — YOLOv11n ONNX (GPU/CPU auto-detect)
- **Multi-Object Tracking** — ByteTrack with identity fusion (face > appearance > person-ReID)
- **Identity Fusion** — Stitching, face merges, appearance Re-ID across track fragments
- **Attendance Snapshots** — Auto-captures face crop on entry/exit, shows on /snapshots page
- **Name Unknown Persons** — Click "NAME IT" on snapshot to identify unknowns, auto-learns face embedding
- **Attendance Manager** — Auto check-in/check-out, Late/Early detection, shift-based
- **GPU Auto-Switch** — Monitors VRAM via nvidia-smi, falls back to CPU when exceeded, retries after cooldown
- **System Monitor** — Live CPU/GPU temperature, RAM, disk, network gauges
- **24/7 Operation** — Auto-restarts on stream loss, 5s retry with backoff
- **Spatial-Temporal Reasoning** — U-turn/loitering detection, anti-tailgating, time-window bias
- **Redis Streams** — Real-time event bus with JSONL fallback
- **SQLite Storage** — WAL-mode events, attendance, face gallery databases

## Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/ahmedA-gif/Face-recognition-ubunto.git
cd Face-recognition-ubunto
python -m venv .venv
.\.venv\Scripts\activate          # Windows
# source .venv/bin/activate       # Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install flask flask-socketio shapely psutil wmi pywin32
```

### 3. Configure camera

Edit `config/settings.yaml`:

```yaml
camera:
  source: rtsp://admin:password@192.168.2.112:554/cam/realmonitor?channel=1&subtype=1

crossing_line:
  enabled: true
  line:
    x1: 0.1
    y1: 0.95    # adjust to your camera angle
    x2: 0.9
    y2: 0.95
  entry_direction: downward  # upward/downward/leftward/rightward
```

### 4. Run

```bash
# Web dashboard (recommended)
python main.py --web
# Open http://localhost:5000 → Live Monitoring → START

# CLI mode (OpenCV window)
python main.py
```

### 5. Enroll faces

```bash
# Via web: People page → Add Person → upload photo
# Or via CLI:
mkdir -p data/faces_gallery/YourName
cp photo.jpg data/faces_gallery/YourName/
python scripts/enroll_faces.py
```

## Architecture

```
CCTV Camera (Dahua RTSP)
        │
        ▼
CameraStream (OpenCV capture)
        │
        ▼
YOLOv11n (person detection) ──► ByteTrack (tracking)
        │                              │
        ▼                              ▼
InsightFace (face recognition)   Identity Fusion (Re-ID)
        │                              │
        └──────────┬───────────────────┘
                   ▼
         CrossingLine Engine (line crossing)
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   Events DB   Attendance   Snapshots
   (SQLite)    (check-in)   (face crops)
        │          │
        ▼          ▼
   Web Dashboard (Flask)
   - Live video stream (MJPEG)
   - Events feed
   - Snapshots + naming
   - System monitoring
```

## Web Dashboard Pages

| Page | Description |
|------|-------------|
| `/` | Overview: KPIs, today's events, attendance summary |
| `/live` | Live video feed, start/stop pipeline, event feed |
| `/snapshots` | Face captures with name unknown feature |
| `/attendance` | Daily check-in/out table with status |
| `/people` | CRUD for face gallery with enrollment |
| `/cameras` | Camera management and status |
| `/events` | Full event history |
| `/tracking` | Real-time track visualization |
| `/analytics` | Entry/exit charts and statistics |
| `/database` | Backup/clear databases |
| `/system` | CPU/GPU temp, RAM, disk monitoring |
| `/settings` | Live config editing with YAML save |

## Configuration

All settings in `config/settings.yaml`:

| Section | Key | Description |
|---------|-----|-------------|
| `camera.source` | RTSP URL | Camera stream source |
| `crossing_line.enabled` | `true` | Enable crossing-line detection |
| `crossing_line.line` | `{x1,y1,x2,y2}` | Normalized line coordinates |
| `crossing_line.entry_direction` | `downward` | Which direction = entry |
| `crossing_line.min_track_frames` | `3` | Min frames before crossing check |
| `crossing_line.cooldown_sec` | `2.0` | Min seconds between events |
| `attendance.shift_start` | `09:00` | Shift start for Late detection |
| `attendance.shift_end` | `17:00` | Shift end for Early exit detection |
| `identity_fusion.enabled` | `true` | Enable identity fusion |
| `pipeline.device` | `cuda:0` | Device (cuda:0 or cpu) |
| `pipeline.skip_frames` | `2` | Process every Nth frame |
| `pipeline.face_every_n` | `3` | Run face recognition every Nth frame |
| `overlay.show_boundary` | `true` | Show crossing line on video |
| `gpu.vram_threshold_gb` | `2.0` | VRAM limit before CPU fallback |

## Project Structure

```
Face-recognition-ubunto/
├── main.py                        # Entry point (--web for dashboard)
├── config/settings.yaml           # All pipeline settings
├── src/
│   ├── capture/stream.py          # Camera RTSP reader
│   ├── detection/person_yolo.py   # YOLO person detector
│   ├── tracking/
│   │   ├── bytetrack.py           # ByteTrack tracker
│   │   └── identity_fusion.py     # Re-ID across tracks
│   ├── recognition/
│   │   ├── face_engine.py         # InsightFace detection + embedding
│   │   └── gallery.py             # FAISS face gallery
│   ├── events/
│   │   ├── crossing_line.py       # Crossing-line entry/exit engine
│   │   ├── store.py               # SQLite event store
│   │   └── redis_publisher.py     # Redis Streams publisher
│   ├── pipeline/runner.py         # Main pipeline (on_frame callback)
│   ├── reasoning/spatial_temporal.py
│   ├── attendance/
│   │   ├── db.py                  # Attendance SQLite DB
│   │   └── manager.py             # Check-in/check-out logic
│   ├── overlay/draw.py            # Live visualization
│   ├── hardware/
│   │   ├── gpu_monitor.py         # GPU VRAM auto-switch
│   │   └── system_monitor.py      # CPU/GPU temp, RAM monitoring
│   └── utils/
│       ├── config.py              # Settings loader
│       └── assign.py              # Face-to-track assignment
├── web/
│   ├── app.py                     # Flask app (12 pages, 16+ APIs)
│   └── templates/                 # Jinja2 HTML templates
├── data/
│   ├── snapshots/                 # Event face captures
│   ├── db/                        # SQLite databases
│   └── faces_gallery/             # Enrolled face images
└── models/
    ├── yolo/                      # YOLOv11n ONNX
    └── face/                      # InsightFace buffalo_s
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Pipeline won't start | Check camera URL in settings.yaml, verify `ping 192.168.2.112` |
| No events firing | Adjust `crossing_line.line.y1` to match where people walk in your camera view |
| Entry/exit swapped | Change `entry_direction` to opposite (upward↔downward) |
| No snapshots | Check `data/snapshots/` folder exists and is writable |
| GPU not detected | Run `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"` |
| Redis warning | Redis not required — falls back to JSONL automatically |
| Attendance page empty | Ensure pipeline has run and events exist in `/events` page |

## Camera Setup

| Setting | Value |
|---------|-------|
| Camera | Dahua IPC |
| IP | 192.168.2.112 |
| Connection | Direct Ethernet |
| Stream | Sub stream (`subtype=1`) |
| Resolution | 1280x720 |

## License

Personal project for office attendance monitoring.
