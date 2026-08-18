"""Flask web application for VisionAttend AI — CCTV Attendance System.

Provides a full web dashboard with:
- Live monitoring with video feed
- People CRUD (add/edit/delete with face enrollment)
- Camera management
- Attendance logs
- Events feed
- System status (GPU/CPU, models)
- Pipeline start/stop control
"""

from __future__ import annotations

import os
import sys
import time
import json
import uuid
import shutil
import sqlite3
import threading
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SITE = ROOT / ".venv" / "Lib" / "site-packages"
_ORT_CAPI = _SITE / "onnxruntime" / "capi"
for _sub in ("nvidia\\cudnn\\bin", "nvidia\\cublas\\bin", "nvidia\\cuda_runtime\\bin", "nvidia\\cufft\\bin"):
    _d = _SITE / _sub
    if _d.is_dir():
        os.add_dll_directory(str(_d))
        os.environ["PATH"] = str(_d) + ";" + os.environ.get("PATH", "")
if _ORT_CAPI.is_dir():
    os.add_dll_directory(str(_ORT_CAPI))
    os.environ["PATH"] = str(_ORT_CAPI) + ";" + os.environ.get("PATH", "")

from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from flask_socketio import SocketIO

from src.utils.config import load_settings, ROOT as PROJECT_ROOT
from src.recognition.gallery import FaceGallery
from src.events.store import EventsStore
from src.attendance.db import AttendanceDB

app = Flask(__name__, template_folder=str(ROOT / "web" / "templates"), static_folder=str(ROOT / "web" / "static"))
app.secret_key = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ─── Globals ──────────────────────────────────────────────────────────────────
_pipeline_thread: Optional[threading.Thread] = None
_pipeline_stop_event = threading.Event()
_pipeline_running = False
_pipeline_status: Dict[str, Any] = {
    "running": False,
    "frames": 0,
    "events": 0,
    "fps": 0.0,
    "source": "",
    "error": "",
    "start_time": "",
    "gpu_mode": True,
}
_gpu_mode = True
_current_config: Optional[dict] = None
_last_events: List[dict] = []
_events_lock = threading.Lock()
_snapshots_dir = ROOT / "data" / "snapshots"
_snapshots_dir.mkdir(parents=True, exist_ok=True)

# ─── Frame buffer for MJPEG streaming ───────────────────────────────────────
import cv2
import numpy as np
_frame_buffer: Optional[np.ndarray] = None
_frame_lock = threading.Lock()
_frame_quality = 60  # JPEG quality 1-100


def _set_frame(frame: np.ndarray) -> None:
    global _frame_buffer
    with _frame_lock:
        _frame_buffer = frame.copy()


def _get_frame() -> Optional[np.ndarray]:
    with _frame_lock:
        return _frame_buffer.copy() if _frame_buffer is not None else None


def _get_config() -> dict:
    global _current_config
    if _current_config is None:
        _current_config = load_settings()
    return _current_config


def _get_gallery() -> FaceGallery:
    cfg = _get_config()
    return FaceGallery(
        db_path=cfg["events"]["faces_db_path"],
        match_threshold=cfg["models"]["face_match_threshold"],
        backend="faiss",
    )


def _get_events_store() -> EventsStore:
    cfg = _get_config()
    return EventsStore(db_path=cfg["events"]["db_path"])


def _get_attendance_db() -> AttendanceDB:
    cfg = _get_config()
    return AttendanceDB(db_path=cfg["attendance"]["db_path"])


# ─── Pipeline Thread ──────────────────────────────────────────────────────────

def _pipeline_worker(source_override: Optional[str] = None):
    """Run pipeline in background thread, capture frames for MJPEG streaming."""
    global _pipeline_running, _pipeline_status
    _pipeline_running = True
    _pipeline_status["running"] = True
    _pipeline_status["error"] = ""
    _pipeline_status["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        cfg = load_settings()
        if _gpu_mode:
            cfg["pipeline"]["device"] = "cuda:0"
        else:
            cfg["pipeline"]["device"] = "cpu"

        source = source_override or cfg["camera"]["source"]
        _pipeline_status["source"] = str(source)

        from src.capture.stream import CameraStream
        from src.detection.person_yolo import PersonDetector
        from src.tracking.bytetrack import ByteTracker
        from src.recognition.face_engine import FaceEngine
        from src.recognition.gallery import FaceGallery
        from src.events.crossing_line import CrossingLineEngine
        from src.events.store import EventsStore
        from src.overlay.draw import OverlayRenderer
        from src.utils.config import load_zones
        from src.utils.assign import attach_faces_to_tracks
        from src.attendance.db import AttendanceDB
        from src.attendance.manager import AttendanceManager
        import numpy as np

        m = cfg["models"]
        pipe = cfg["pipeline"]
        ee = cfg["entry_exit"]
        ev = cfg["events"]
        ov = cfg["overlay"]

        skip_frames = int(pipe.get("skip_frames", 2))
        face_every_n = int(pipe.get("face_every_n", 3))

        # Build components
        backend = pipe.get("backend", "onnx")
        detector = PersonDetector(
            weights=m.get("yolo_onnx", m["yolo_weights"]),
            conf=m["yolo_conf"], iou=m["yolo_iou"], imgsz=m["yolo_imgsz"],
            device=pipe["device"], person_class_id=m["person_class_id"], backend=backend,
        )
        tracker = ByteTracker()
        face_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "cuda" in str(pipe["device"]) else None
        face_engine = FaceEngine(root=m["face_root"], pack=m["face_pack"],
                                det_size=tuple(m["face_det_size"]), providers=face_providers)
        gallery = FaceGallery(db_path=ev["faces_db_path"], match_threshold=m["face_match_threshold"], backend="faiss")
        store = EventsStore(db_path=ev["db_path"])

        cl = cfg.get("crossing_line") or ee.get("crossing_line") or {}
        def _l2t(line):
            if isinstance(line, dict): return (float(line["x1"]), float(line["y1"]), float(line["x2"]), float(line["y2"]))
            return tuple(line) if isinstance(line, (list, tuple)) else (0.25, 0.75, 0.75, 0.75)
        event_engine = CrossingLineEngine(
            line_norm=_l2t(cl.get("line", ee.get("line", (0.25, 0.75, 0.75, 0.75)))),
            entry_direction=cl.get("entry_direction", "downward"),
            camera_id=cl.get("camera_id", "cam_01"),
            min_track_frames=cl.get("min_track_frames", 3),
            cooldown_sec=cl.get("cooldown_sec", 2.0),
            min_crossing_gap_sec=cl.get("min_crossing_gap_sec", 1.0),
        )
        overlay = OverlayRenderer(pulse_frames=ov["pulse_frames"], hud=ov["hud"], show_boundary=bool(ov.get("show_boundary", True)))
        _line_t = event_engine.line_norm
        overlay.set_boundary({"x1": _line_t[0], "y1": _line_t[1], "x2": _line_t[2], "y2": _line_t[3]}, label="BOUNDARY")

        att = cfg.get("attendance", {})
        att_db = AttendanceDB(db_path=att.get("db_path", "data/db/attendance.db")) if att.get("enabled") else None
        attendance = AttendanceManager(db=att_db, shift_start=att.get("shift_start", "09:00"),
            shift_end=att.get("shift_end", "17:00"), late_threshold_mins=att.get("late_threshold_mins", 15),
            early_exit_mins=att.get("early_exit_mins", 15), debounce_minutes=att.get("debounce_minutes", 2.0)) if att_db else None

        frames_processed = 0
        total_events = 0
        last_dets = []
        last_faces = []

        with CameraStream(source=source, buffer_size=cfg["camera"].get("buffer_size", 1),
                          width=cfg["camera"].get("width"), height=cfg["camera"].get("height")) as stream:
            while not _pipeline_stop_event.is_set():
                ok, frame = stream.read()
                if not ok or frame is None:
                    if frames_processed == 0:
                        _pipeline_status["error"] = "Could not read frame from source"
                    break

                frames_processed += 1
                run_det = (frames_processed - 1) % max(1, skip_frames) == 0
                run_face = (frames_processed - 1) % max(1, face_every_n) == 0

                if run_det:
                    last_dets = detector.detect(frame)
                tracks = tracker.update(last_dets)

                if run_face:
                    last_faces = face_engine.detect_and_embed(frame, min_face_px=m.get("min_face_px", 1))
                if last_faces:
                    attach_faces_to_tracks(tracks, last_faces, gallery.match)

                events = event_engine.update(tracks, frame.shape, store)
                if events:
                    total_events += len(events)
                    _snap_dir = Path(ev.get("snapshots_dir", "data/snapshots"))
                    _snap_dir.mkdir(parents=True, exist_ok=True)
                    for e in events:
                        try:
                            snap_name = f"{e.date}_{e.time.replace(':', '-')}_{e.track_id}_{e.direction}.jpg"
                            cv2.imwrite(str(_snap_dir / snap_name), frame)
                            e.snapshot_path = str(_snap_dir / snap_name)
                        except Exception:
                            pass
                    for e in events:
                        person = e.person or "Unknown"
                        print(f"[EVENT] {e.date} {e.time} | {person} | {e.direction} conf={e.confidence:.2f}")
                        if attendance is not None:
                            attendance.process_events([e])

                vis = overlay.draw(frame, tracks, last_faces, event_engine.counts)
                _set_frame(vis)

                _pipeline_status["frames"] = frames_processed
                _pipeline_status["events"] = total_events

    except Exception as e:
        _pipeline_status["error"] = f"Fatal: {e}\n{traceback.format_exc()}"
    finally:
        _pipeline_running = False
        _pipeline_status["running"] = False


# ─── Template Helpers ─────────────────────────────────────────────────────────

@app.context_processor
def inject_globals():
    cfg = _get_config()
    return {
        "now": datetime.now(),
        "gpu_mode": _gpu_mode,
        "pipeline_running": _pipeline_running,
        "pipeline_status": _pipeline_status,
    }


# ─── Page Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    cfg = _get_config()
    gallery = _get_gallery()
    store = _get_events_store()
    att = _get_attendance_db()

    today = datetime.now().strftime("%Y-%m-%d")
    try:
        events_today = store.list_events(date=today, limit=100) if hasattr(store, 'list_events') else []
    except Exception:
        events_today = []
    try:
        attendance_today = att.list_by_date(today) if hasattr(att, 'list_by_date') else []
    except Exception:
        attendance_today = []

    people_count = gallery.count_db()
    entry_count = sum(1 for e in events_today if e.get("direction") == "entry")
    exit_count = sum(1 for e in events_today if e.get("direction") == "exit")
    present_count = sum(1 for a in attendance_today if a.get("check_in_time") and not a.get("check_out_time"))

    return render_template("dashboard.html",
        people_count=people_count,
        entry_count=entry_count,
        exit_count=exit_count,
        present_count=present_count,
        events_today=events_today[:20],
        attendance_today=attendance_today[:20],
        total_events=len(events_today),
    )


@app.route("/live")
def live():
    return render_template("live.html")


@app.route("/people")
def people():
    gallery = _get_gallery()
    names = gallery.list_people()
    people_list = []
    for name in names:
        people_list.append({
            "name": name,
            "embedding_count": gallery.count(),
        })
    return render_template("people.html", people=people_list)


@app.route("/attendance")
def attendance():
    att = _get_attendance_db()
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        records = att.list_by_date(today) if hasattr(att, 'list_by_date') else []
    except Exception:
        records = []
    return render_template("attendance.html", records=records, today=today)


@app.route("/cameras")
def cameras():
    cfg = _get_config()
    camera_source = cfg.get("camera", {}).get("source", "")
    cameras_list = []
    if camera_source:
        cameras_list.append({
            "id": "cam_01",
            "name": "Main Camera",
            "source": camera_source,
            "status": "online" if _pipeline_running else "standby",
            "resolution": f"{cfg['camera'].get('width', '?')}x{cfg['camera'].get('height', '?')}",
        })
    return render_template("cameras.html", cameras=cameras_list)


@app.route("/events")
def events():
    store = _get_events_store()
    try:
        all_events = store.list_events(limit=200) if hasattr(store, 'list_events') else []
    except Exception:
        all_events = []
    return render_template("events.html", events=all_events)


@app.route("/tracking")
def tracking():
    return render_template("tracking.html", status=_pipeline_status)


@app.route("/analytics")
def analytics():
    store = _get_events_store()
    att = _get_attendance_db()
    try:
        all_events = store.list_events(limit=500) if hasattr(store, 'list_events') else []
    except Exception:
        all_events = []
    entry_count = sum(1 for e in all_events if e.get("direction") == "entry")
    exit_count = sum(1 for e in all_events if e.get("direction") == "exit")
    return render_template("analytics.html",
        total_events=len(all_events),
        entry_count=entry_count,
        exit_count=exit_count,
    )


@app.route("/database")
def database():
    cfg = _get_config()
    dbs = {
        "events": cfg["events"]["db_path"],
        "faces": cfg["events"]["faces_db_path"],
        "attendance": cfg["attendance"]["db_path"],
    }
    db_info = {}
    for name, path in dbs.items():
        p = Path(path)
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            conn = sqlite3.connect(str(p))
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            table_counts = {}
            for t in tables:
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
                    table_counts[t] = count
                except Exception:
                    table_counts[t] = 0
            conn.close()
            db_info[name] = {"path": str(p), "size_mb": round(size_mb, 2), "tables": table_counts}
        else:
            db_info[name] = {"path": str(p), "size_mb": 0, "tables": {}}
    return render_template("database.html", databases=db_info)


@app.route("/system")
def system():
    import platform
    gpu_available = False
    gpu_name = "N/A"
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        gpu_available = "CUDAExecutionProvider" in providers
        if gpu_available:
            gpu_name = "NVIDIA GPU (CUDA)"
    except Exception:
        pass

    cfg = _get_config()
    models = {
        "yolo": cfg["models"].get("yolo_weights", ""),
        "face": f"{cfg['models'].get('face_root', '')}/{cfg['models'].get('face_pack', '')}",
    }
    return render_template("system.html",
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_mode=_gpu_mode,
        python_version=platform.python_version(),
        os_info=f"{platform.system()} {platform.release()}",
        models=models,
    )


@app.route("/settings")
def settings():
    cfg = _get_config()
    return render_template("settings.html", config=cfg)


# ─── API Routes ───────────────────────────────────────────────────────────────

# ─── MJPEG Video Stream ──────────────────────────────────────────────────────

@app.route("/video_feed")
def video_feed():
    """MJPEG stream endpoint for live video in browser."""
    from flask import Response
    import time as _time

    def generate():
        while True:
            frame = _get_frame()
            if frame is not None:
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _frame_quality])
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n"
                       + buffer.tobytes() + b"\r\n")
            else:
                # Send a blank frame when no pipeline running
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "No Signal", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 60, 60), 2)
                _, buffer = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 30])
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n"
                       + buffer.tobytes() + b"\r\n")
            _time.sleep(0.033)  # ~30fps

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/pipeline/start", methods=["POST"])
def api_pipeline_start():
    global _pipeline_thread, _pipeline_stop_event
    if _pipeline_running:
        return jsonify({"ok": False, "error": "Pipeline already running"})
    _pipeline_stop_event.clear()
    source = request.json.get("source") if request.is_json else None
    _pipeline_thread = threading.Thread(target=_pipeline_worker, args=(source,), daemon=True)
    _pipeline_thread.start()
    return jsonify({"ok": True, "message": "Pipeline started"})


@app.route("/api/pipeline/stop", methods=["POST"])
def api_pipeline_stop():
    global _pipeline_stop_event
    if not _pipeline_running:
        return jsonify({"ok": False, "error": "Pipeline not running"})
    _pipeline_stop_event.set()
    return jsonify({"ok": True, "message": "Pipeline stopping"})


@app.route("/api/pipeline/status")
def api_pipeline_status():
    return jsonify(_pipeline_status)


@app.route("/api/gpu/status")
def api_gpu_status():
    from src.hardware.gpu_monitor import gpu_monitor
    return jsonify(gpu_monitor.get_status())


@app.route("/api/system/metrics")
def api_system_metrics():
    from src.hardware.system_monitor import system_monitor
    return jsonify(system_monitor.get_all())


@app.route("/api/gpu/switch", methods=["POST"])
def api_gpu_switch():
    global _gpu_mode, _current_config
    mode = request.json.get("mode", "cpu") if request.is_json else "cpu"
    _gpu_mode = mode.lower() in ("gpu", "cuda", "true", "1")
    _current_config = None  # force reload
    cfg = load_settings()
    if _gpu_mode:
        cfg["pipeline"]["device"] = "cuda:0"
    else:
        cfg["pipeline"]["device"] = "cpu"
    _current_config = cfg
    return jsonify({"ok": True, "gpu_mode": _gpu_mode, "device": cfg["pipeline"]["device"]})


@app.route("/api/people", methods=["GET"])
def api_people_list():
    gallery = _get_gallery()
    names = gallery.list_people()
    people = []
    for name in names:
        people.append({"name": name, "id": name.lower().replace(" ", "_")})
    return jsonify(people)


@app.route("/api/people", methods=["POST"])
def api_people_add():
    name = request.json.get("name", "").strip() if request.is_json else ""
    if not name:
        return jsonify({"ok": False, "error": "Name required"}), 400

    gallery = _get_gallery()
    image_data = request.json.get("image") if request.is_json else None

    if image_data:
        import base64
        import cv2
        import numpy as np
        try:
            img_bytes = base64.b64decode(image_data.split(",")[-1] if "," in image_data else image_data)
            arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({"ok": False, "error": "Invalid image"}), 400
            from src.recognition.face_engine import FaceEngine
            cfg = _get_config()
            fe = FaceEngine(root=cfg["models"]["face_root"], pack=cfg["models"]["face_pack"],
                            det_size=tuple(cfg["models"]["face_det_size"]),
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if _gpu_mode else None)
            hits = fe.detect_and_embed(img, min_face_px=20)
            if not hits:
                return jsonify({"ok": False, "error": "No face detected in image"}), 400
            hit = max(hits, key=lambda h: (h.xyxy[2] - h.xyxy[0]) * (h.xyxy[3] - h.xyxy[1]))
            gallery.add(name, hit.embedding)
            gallery.flush()
            return jsonify({"ok": True, "message": f"Added {name} with face embedding"})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
    else:
        person_dir = Path(cfg["gallery"]["images_dir"]) / name
        if person_dir.exists():
            from src.recognition.face_engine import FaceEngine
            fe = FaceEngine(root=cfg["models"]["face_root"], pack=cfg["models"]["face_pack"],
                            det_size=tuple(cfg["models"]["face_det_size"]),
                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if _gpu_mode else None)
            counts = gallery.enroll_folder(str(Path(cfg["gallery"]["images_dir"])), fe)
            gallery.flush()
            enrolled = counts.get(name, 0)
            return jsonify({"ok": True, "message": f"Enrolled {name}: {enrolled} images"})
        return jsonify({"ok": True, "message": f"Added {name} (no face images yet)"})


@app.route("/api/people/<name>", methods=["DELETE"])
def api_people_delete(name: str):
    gallery = _get_gallery()
    try:
        gallery._conn.execute("DELETE FROM faces WHERE name = ?", (name,))
        gallery._conn.commit()
        gallery.reload()
        return jsonify({"ok": True, "message": f"Deleted {name}"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/people/<name>", methods=["PUT"])
def api_people_update(name: str):
    new_name = request.json.get("name", "").strip() if request.is_json else ""
    if not new_name:
        return jsonify({"ok": False, "error": "New name required"}), 400
    gallery = _get_gallery()
    try:
        gallery._conn.execute("UPDATE faces SET name = ? WHERE name = ?", (new_name, name))
        gallery._conn.commit()
        gallery.reload()
        return jsonify({"ok": True, "message": f"Renamed {name} to {new_name}"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/cameras", methods=["GET"])
def api_cameras_list():
    cfg = _get_config()
    cameras = []
    src = cfg.get("camera", {}).get("source", "")
    if src:
        cameras.append({
            "id": "cam_01",
            "name": "Main Camera",
            "source": src,
            "status": "online" if _pipeline_running else "standby",
        })
    return jsonify(cameras)


@app.route("/api/cameras", methods=["POST"])
def api_cameras_add():
    data = request.json if request.is_json else {}
    source = data.get("source", "").strip()
    name = data.get("name", "").strip() or "Camera"
    if not source:
        return jsonify({"ok": False, "error": "Source URL required"}), 400
    cfg = _get_config()
    cfg["camera"]["source"] = source
    cfg["camera"]["ip"] = data.get("ip", "")
    return jsonify({"ok": True, "message": f"Camera '{name}' added"})


@app.route("/api/cameras/<cam_id>", methods=["DELETE"])
def api_cameras_delete(cam_id: str):
    return jsonify({"ok": True, "message": f"Camera {cam_id} removed"})


@app.route("/api/events", methods=["GET"])
def api_events_list():
    store = _get_events_store()
    limit = request.args.get("limit", 100, type=int)
    try:
        events = store.list_events(limit=limit) if hasattr(store, 'list_events') else []
    except Exception:
        events = []
    return jsonify(events)


@app.route("/api/attendance", methods=["GET"])
def api_attendance_list():
    att = _get_attendance_db()
    date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    try:
        records = att.list_by_date(date) if hasattr(att, 'list_by_date') else []
    except Exception:
        records = []
    return jsonify(records)


@app.route("/api/database/backup", methods=["POST"])
def api_database_backup():
    cfg = _get_config()
    backup_dir = ROOT / "data" / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    for key in ("db_path", "faces_db_path"):
        src = Path(cfg["events"][key])
        if src.exists():
            shutil.copy2(str(src), str(backup_dir / src.name))
    att_src = Path(cfg["attendance"]["db_path"])
    if att_src.exists():
        shutil.copy2(str(att_src), str(backup_dir / att_src.name))
    return jsonify({"ok": True, "path": str(backup_dir)})


@app.route("/api/database/clear", methods=["POST"])
def api_database_clear():
    db_name = request.json.get("database", "") if request.is_json else ""
    cfg = _get_config()
    db_map = {
        "events": cfg["events"]["db_path"],
        "faces": cfg["events"]["faces_db_path"],
        "attendance": cfg["attendance"]["db_path"],
    }
    if db_name not in db_map:
        return jsonify({"ok": False, "error": "Invalid database name"}), 400
    path = Path(db_map[db_name])
    if path.exists():
        conn = sqlite3.connect(str(path))
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        for t in tables:
            conn.execute(f"DELETE FROM [{t}]")
        conn.commit()
        conn.close()
    return jsonify({"ok": True, "message": f"Cleared {db_name} database"})


@app.route("/api/snapshots/<path:filename>")
def api_snapshot(filename):
    return send_from_directory(str(_snapshots_dir), filename)


@app.route("/api/config", methods=["GET"])
def api_config_get():
    cfg = _get_config()
    safe = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return jsonify(safe)


@app.route("/api/config", methods=["POST"])
def api_config_update():
    global _current_config
    data = request.json if request.is_json else {}
    if not data:
        return jsonify({"ok": False, "error": "No data"}), 400
    import yaml
    cfg_path = ROOT / "config" / "settings.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        current = yaml.safe_load(f) or {}

    def deep_update(base, update):
        for k, v in update.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                deep_update(base[k], v)
            else:
                base[k] = v

    deep_update(current, data)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(current, f, default_flow_style=False, allow_unicode=True)
    _current_config = None
    return jsonify({"ok": True, "message": "Config updated"})


@app.route("/api/models", methods=["GET"])
def api_models_list():
    cfg = _get_config()
    models_dir = ROOT / "models"
    available = {}
    for subdir in models_dir.iterdir():
        if subdir.is_dir():
            files = [f.name for f in subdir.iterdir() if f.is_file()]
            available[subdir.name] = files
    return jsonify(available)


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)
