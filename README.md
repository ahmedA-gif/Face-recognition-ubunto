# Person Face Events (CPU)

Lean CPU pipeline for:

- **Person detection only** (YOLO nano)
- **Face detection + recognition** (InsightFace SCRFD + ArcFace)
- **Tracking** (ByteTrack)
- **Counting + Entry/Exit events** (virtual line)
- **Low-latency stream** via go2rtc (optional)
- **Live overlay animation** on camera frames

## Event fields

Each entry/exit stores: `date`, `time`, `person`, `direction` (`entry` | `exit`).

## Project structure

```text
person-face-events/
├── config/
│   ├── settings.yaml          # app settings (camera, thresholds, line)
│   └── go2rtc.yaml            # low-latency streaming
├── notebooks/
│   ├── 00_project_runner.ipynb
│   ├── 01_download_yolo.ipynb
│   ├── 02_download_bytetrack.ipynb
│   ├── 03_download_face_models.ipynb
│   └── 04_verify_models_cpu.ipynb
│   └── 05_test_uploaded_video.ipynb
├── models/
│   ├── yolo/                  # yolov8n / yolo11n weights + onnx
│   ├── face/                  # InsightFace buffalo_s pack
│   └── tracker/               # ByteTrack config / notes
├── data/
│   ├── faces_gallery/         # enrolled face images (name folders)
│   ├── snapshots/             # optional event crops
│   └── db/                    # SQLite: faces + events
├── src/
│   ├── capture/               # OpenCV / go2rtc RTSP
│   ├── detection/             # person YOLO
│   ├── tracking/              # ByteTrack
│   ├── recognition/           # face detect + ArcFace match
│   ├── events/                # entry/exit + SQLite
│   ├── overlay/               # boxes + pulse animation
│   └── utils/
├── scripts/
│   ├── enroll_face.py
│   └── run_pipeline.py
├── web/                       # optional MJPEG preview
├── tests/
├── requirements.txt
└── main.py
```

## Quick start

```bash
cd ~/Desktop/person-face-events
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Use a fresh venv for this project if possible; mixing in unrelated ML packages
# can force incompatible NumPy / SciPy / Torch versions.

# 1) Download models via notebooks (or run cells top-to-bottom)
jupyter notebook notebooks/

# 2) Enroll faces: data/faces_gallery/Ahmed/img1.jpg ...
python scripts/enroll_face.py

# 3) Test an uploaded video first
python main.py --source /path/to/video.mp4 --output outputs/test_annotated.mp4 --no-display

# 4) Then run CCTV / go2rtc
python main.py --source rtsp://127.0.0.1:8554/cam_01_sub
```

If you want the notebooks or the optional preview API, install those extras separately:

```bash
pip install jupyter ipywidgets tqdm requests fastapi uvicorn
```

## CPU notes

- YOLO: `yolo11n` or `yolov8n` (nano only)
- Face: InsightFace `buffalo_s`
- Tracker: ByteTrack (CPU)
- Prefer go2rtc substream + `CAP_PROP_BUFFERSIZE=1`
- Process every Nth frame; tracker fills gaps
- Use `main.py --source <video-or-rtsp>` to switch between uploaded video testing and CCTV
