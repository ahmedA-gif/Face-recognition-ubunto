"""Frigate attendance bridge.

Frigate owns both object identity and face recognition.  This process only
correlates Frigate event updates, validates a door-zone crossing, and mirrors
the resulting record to the attendance database/dashboard.
"""
import json
import os
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
try:  # Installed in the production bridge image; keeps state-machine tests light.
    import paho.mqtt.client as mqtt
except ImportError:  # pragma: no cover - production always installs paho
    mqtt = None
try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover - DB writes simply remain unavailable
    psycopg2 = None
import requests
from flask import Flask, Response, jsonify, render_template, request

FRIGATE_API = os.environ.get("FRIGATE_API", "http://frigate:5000")
MQTT_HOST = os.environ.get("MQTT_HOST", "mqtt")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
WEB_PORT = int(os.environ.get("WEB_PORT", "5001"))
DATABASE_URL = os.environ.get("DATABASE_URL", "")
NAME_WAIT_SEC = float(os.environ.get("NAME_WAIT_SEC", "6"))
MIN_TRACK_DURATION_SEC = float(os.environ.get("MIN_TRACK_DURATION_SEC", "1.5"))
MIN_TRACK_DISTANCE = float(os.environ.get("MIN_TRACK_DISTANCE", "0.03"))
EVENT_COOLDOWN_SEC = float(os.environ.get("EVENT_COOLDOWN_SEC", "20"))
FACE_UNKNOWN_SCORE = float(os.environ.get("FACE_UNKNOWN_SCORE", "0.75"))
FRIGATE_DETECT_WIDTH = float(os.environ.get("FRIGATE_DETECT_WIDTH", "640"))
FRIGATE_DETECT_HEIGHT = float(os.environ.get("FRIGATE_DETECT_HEIGHT", "480"))
SHIFT_START = os.environ.get("SHIFT_START", "09:00")
LATE_THRESHOLD_MINS = int(os.environ.get("LATE_THRESHOLD_MINS", "15"))

# Zone names deliberately live here too: a missing/renamed zone must fail
# closed, rather than falling back to the camera name.
ZONE_PATHS = {
    "cam_entry": {
        ("outside_door", "door_threshold", "inside_room"): "ENTRY",
        ("inside_room", "door_threshold", "outside_door"): "EXIT",
    },
    "cam_exit": {
        ("inside_exit", "door_exit", "outside_exit"): "EXIT",
        ("outside_exit", "door_exit", "inside_exit"): "ENTRY",
    },
}

app = Flask(__name__)
_db_conn = None
_db_lock = threading.RLock()
_tracks = {}
_tracks_lock = threading.RLock()
_latest_events = []
_events_lock = threading.Lock()
_person_state = {}
_person_state_lock = threading.Lock()
_person_cooldowns = {}
_unknown_counter = 0
_counter_lock = threading.Lock()
_frigate_status = {"connected": False, "last_event": "", "frigate_version": "unknown", "mqtt": False}
_person_counts = {camera: {"total": 0, "current": 0, "entered": 0, "exited": 0} for camera in ZONE_PATHS}
_counts_lock = threading.Lock()


def _serialized_db(fn):
    """Serialize use of the single shared psycopg connection across threads."""
    def wrapped(*args, **kwargs):
        with _db_lock:
            return fn(*args, **kwargs)
    return wrapped


def _name_from_sublabel(value):
    """Return a trustworthy Frigate label, accepting both supported shapes."""
    name, score = None, None
    if isinstance(value, str):
        name = value
    elif isinstance(value, (list, tuple)) and value:
        name = value[0]
        if len(value) > 1:
            try:
                score = float(value[1])
            except (TypeError, ValueError):
                pass
    if not isinstance(name, str) or not name.strip() or name.lower() == "unknown":
        return None
    # A score is optional in Frigate payloads.  If present, use the same floor
    # as the bridge/config rather than promoting a weak face match.
    if score is not None and score < FACE_UNKNOWN_SCORE:
        return None
    return name.strip()


def _event_timestamp(event):
    for value in (event.get("end_time"), event.get("start_time")):
        if value:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return time.time()


def extract_foot_point(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = map(float, box)
        return ((min(x1, x2) + max(x1, x2)) / 2, max(y1, y2))
    except (TypeError, ValueError):
        return None


class TrackState:
    """One Frigate object/event, never a camera-wide aggregate."""
    def __init__(self, event_id, camera):
        self.track_id = event_id
        self.camera = camera
        self.first_seen = None
        self.last_update = time.time()
        self.end_time = None
        self.name = None
        self.current_zones = []
        self.entered_zones = []
        self.points = []
        self.last_box = None
        self.last_score = 0.0
        self.emitted = False
        self.direction = None
        self.db_written = False

    def update(self, event):
        now = time.time()
        start = event.get("start_time")
        if start is not None:
            try:
                self.first_seen = float(start) if self.first_seen is None else min(self.first_seen, float(start))
            except (TypeError, ValueError):
                pass
        if self.first_seen is None:
            self.first_seen = now
        if event.get("end_time"):
            try:
                self.end_time = float(event["end_time"])
            except (TypeError, ValueError):
                self.end_time = now
        self.last_update = now
        self.name = _name_from_sublabel(event.get("sub_label")) or self.name
        data = event.get("data") or {}
        self.last_box = data.get("box") or self.last_box
        self.last_score = float(data.get("top_score") or self.last_score or 0)
        point = extract_foot_point(self.last_box)
        if point:
            self.points.append((point[0], point[1], _event_timestamp(event)))
            self.points = self.points[-100:]
        zones = data.get("current_zones") or event.get("current_zones") or event.get("zones") or []
        if isinstance(zones, str):
            zones = [zones]
        self.current_zones = [z for z in zones if isinstance(z, str)]
        # Zone arrays are sets from Frigate, not a trajectory. Make each
        # payload stable against arbitrary array order before updating the FSM.
        zone_rank = {}
        for path in ZONE_PATHS.get(self.camera, {}):
            for index, zone in enumerate(path):
                zone_rank[zone] = min(zone_rank.get(zone, index), index)
        for zone in sorted(self.current_zones, key=lambda value: zone_rank.get(value, 99)):
            if zone in self._valid_zones() and (not self.entered_zones or self.entered_zones[-1] != zone):
                self.entered_zones.append(zone)

    def _valid_zones(self):
        return {z for path in ZONE_PATHS.get(self.camera, {}) for z in path}

    def duration(self):
        end = self.end_time or (self.points[-1][2] if self.points else self.last_update)
        return max(0.0, end - (self.first_seen or end))

    def displacement(self):
        if len(self.points) < 2:
            return 0.0
        return self.points[-1][1] - self.points[0][1]

    def direction_from_zones(self):
        history = self.entered_zones
        for path, direction in ZONE_PATHS.get(self.camera, {}).items():
            pos = 0
            for zone in history:
                if zone == path[pos]:
                    pos += 1
                    if pos == len(path):
                        return direction
        return None

    def ready_direction(self):
        direction = self.direction_from_zones()
        if not direction or self.duration() < MIN_TRACK_DURATION_SEC:
            return None
        # A three-zone sequence is required. It has already ruled out a
        # threshold peek/reversal; the displacement guard rules out flicker.
        if len(self.entered_zones) < 3 or abs(self.displacement()) < MIN_TRACK_DISTANCE:
            return None
        return direction

    def to_dict(self):
        return {"track_id": self.track_id, "camera": self.camera, "duration": round(self.duration(), 2),
                "displacement": round(self.displacement(), 3), "zone_sequence": self.entered_zones,
                "current_zones": self.current_zones, "person_name": self.name, "event_emitted": self.emitted}


def _get_db():
    global _db_conn
    if not DATABASE_URL or psycopg2 is None:
        return None
    with _db_lock:
        try:
            if _db_conn is None or _db_conn.closed:
                _db_conn = psycopg2.connect(DATABASE_URL)
            else:
                with _db_conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return _db_conn
        except Exception as exc:
            print(f"[DB] connection failed: {exc}")
            _db_conn = None
            return None


def _calc_status(time_str):
    h, m, _ = map(int, time_str.split(":"))
    sh, sm = map(int, SHIFT_START.split(":"))
    return "Late" if h > sh or (h == sh and m > sm + LATE_THRESHOLD_MINS) else "On Time"


@_serialized_db
def _write_event(track, person_name, direction):
    """Insert once by Frigate event id and attach checkout by person name."""
    conn = _get_db()
    if conn is None:
        return False
    timestamp = datetime.fromtimestamp(track.end_time or time.time())
    date_str, time_str = timestamp.strftime("%Y-%m-%d"), timestamp.strftime("%H:%M:%S")
    foot = extract_foot_point(track.last_box)
    box = " ".join(map(str, track.last_box)) if track.last_box else None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM person_events WHERE track_id = %s LIMIT 1", (track.track_id,))
            if not cur.fetchone():
                cur.execute(
                    "INSERT INTO person_events(date, track_id, person_name, event_type, event_time, confidence, camera_id, zone, foot_y, bounding_box) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (date_str, track.track_id, person_name, direction, time_str, track.last_score, track.camera,
                     ",".join(track.entered_zones), foot[1] if foot else None, box),
                )
            if direction == "ENTRY":
                cur.execute("SELECT id FROM attendance WHERE date=%s AND person_name=%s AND check_out_time IS NULL LIMIT 1", (date_str, person_name))
                if not cur.fetchone():
                    cur.execute("INSERT INTO attendance(date, track_id, person_name, check_in_time, status, confidence, camera_id) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                                (date_str, track.track_id, person_name, time_str, _calc_status(time_str), track.last_score, track.camera))
            else:
                # An exit must close an open attendance row for this person,
                # not a different Frigate object id. Unknown exits are kept as
                # event records but cannot safely be attached to attendance.
                if not person_name.startswith("Unknown#"):
                    cur.execute("SELECT id,date,check_in_time FROM attendance WHERE person_name=%s AND check_out_time IS NULL ORDER BY id DESC LIMIT 1", (person_name,))
                    row = cur.fetchone()
                    if row:
                        checkin = datetime.strptime(f"{row[1]} {row[2]}", "%Y-%m-%d %H:%M:%S")
                        hours = round((timestamp - checkin).total_seconds() / 3600, 2)
                        cur.execute("UPDATE attendance SET check_out_time=%s, work_hours=%s, camera_id=%s, updated_at=NOW() WHERE id=%s", (time_str, hours, track.camera, row[0]))
        conn.commit()
        return True
    except Exception as exc:
        print(f"[DB] write failed: {exc}")
        conn.rollback()
        return False


@_serialized_db
def _update_event_sublabel(event_id, sub_label):
    """Late Frigate identity can replace only Unknown/empty rows, never a user edit."""
    person_name = _name_from_sublabel(sub_label)
    if not person_name:
        return False
    conn = _get_db()
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT person_name,event_type FROM person_events WHERE track_id=%s AND (person_name IS NULL OR person_name='' OR person_name LIKE 'Unknown%%')", (event_id,))
            row = cur.fetchone()
            cur.execute("UPDATE person_events SET person_name=%s WHERE track_id=%s AND (person_name IS NULL OR person_name='' OR person_name LIKE 'Unknown%%')", (person_name, event_id))
            changed = cur.rowcount > 0
            cur.execute("UPDATE attendance SET person_name=%s WHERE track_id=%s AND (person_name IS NULL OR person_name='' OR person_name LIKE 'Unknown%%')", (person_name, event_id))
        conn.commit()
        if changed and row:
            with _person_state_lock:
                old = row[0]
                if old in _person_state:
                    _person_state[person_name] = _person_state.pop(old)
        return changed
    except Exception as exc:
        print(f"[DB] sublabel update failed: {exc}")
        conn.rollback()
        return False


def _unknown_name():
    global _unknown_counter
    with _counter_lock:
        _unknown_counter += 1
        return f"Unknown#{_unknown_counter:02d}"


def _unknown_cooldown_key(track, direction):
    """Deduplicate re-acquired unknown people in a camera-relative 10% cell."""
    foot = extract_foot_point(track.last_box)
    if foot is None:
        return (track.camera, -1, -1, direction)
    x, y = foot
    if abs(x) > 1 or abs(y) > 1:  # Frigate normally supplies pixel boxes.
        x, y = x / FRIGATE_DETECT_WIDTH, y / FRIGATE_DETECT_HEIGHT
    return (track.camera, int(max(0, min(9, x * 10))), int(max(0, min(9, y * 10))), direction)


def _cleanup_person_cooldowns(now=None):
    """Keep the cooldown map bounded for long-running bridge processes."""
    cutoff = (time.time() if now is None else now) - 2 * EVENT_COOLDOWN_SEC
    for key in [key for key, seen_at in _person_cooldowns.items() if seen_at < cutoff]:
        del _person_cooldowns[key]


def _person_inside(name):
    with _person_state_lock:
        return bool(_person_state.get(name, {}).get("inside"))


def _set_person_state(name, direction):
    if name.startswith("Unknown#"):
        return
    with _person_state_lock:
        _person_state[name] = {"inside": direction == "ENTRY", "last_update": time.time()}


def _add_dashboard_event(track, name, direction):
    stamp = datetime.fromtimestamp(track.end_time or time.time()).strftime("%Y-%m-%d %H:%M:%S")
    value = {"person": name, "track_id": track.track_id, "direction": direction, "confidence": f"{track.last_score:.0%}",
             "face_score": 0.0, "timestamp": stamp, "camera": track.camera, "zones": track.entered_zones,
             "score": round(track.last_score, 3), "type": "zone_sequence", "displacement": round(track.displacement(), 3),
             "traj_points": len(track.points), "duration": round(track.duration(), 2), "snapshot_url": f"/api/events/{track.track_id}/snapshot.jpg"}
    with _events_lock:
        _latest_events.insert(0, value)
        del _latest_events[200:]


def _try_emit(track):
    if track.emitted:
        return
    direction = track.ready_direction()
    if not direction:
        return
    # Wait for face recognition to settle once Frigate closes the object.
    if not track.name and (not track.end_time or time.time() - track.end_time < NAME_WAIT_SEC):
        return
    name = track.name or _unknown_name()
    cooldown_key = (name, direction) if not name.startswith("Unknown#") else _unknown_cooldown_key(track, direction)
    last = _person_cooldowns.get(cooldown_key, 0)
    if time.time() - last < EVENT_COOLDOWN_SEC:
        track.emitted = True
        return
    # Fail closed on known exits: an exit for someone not inside is not an
    # attendance transition. During startup state is hydrated before MQTT.
    if direction == "EXIT" and not name.startswith("Unknown#") and not _person_inside(name):
        print(f"[Bridge] ignoring EXIT for {name}: not currently inside")
        track.emitted = True
        return
    if not _write_event(track, name, direction):
        return
    track.emitted, track.direction, track.db_written = True, direction, True
    _person_cooldowns[cooldown_key] = time.time()
    _set_person_state(name, direction)
    with _counts_lock:
        stats = _person_counts.setdefault(track.camera, {"total": 0, "current": 0, "entered": 0, "exited": 0})
        stats["total"] += 1
        stats["entered" if direction == "ENTRY" else "exited"] += 1
    _add_dashboard_event(track, name, direction)
    print(f"[Bridge] {name} {direction} {track.track_id} zones={track.entered_zones}")


def _ingest_event(event, source="mqtt"):
    if not isinstance(event, dict):
        return
    # MQTT is {before, after, type}; REST already is the after event.
    event = event.get("after") or event
    if event.get("label") != "person":
        return
    event_id, camera = event.get("id"), event.get("camera")
    if not event_id or camera not in ZONE_PATHS:
        return
    with _tracks_lock:
        track = _tracks.get(event_id)
        if track is None:
            track = _tracks[event_id] = TrackState(event_id, camera)
        track.update(event)
        if track.name and track.db_written:
            _update_event_sublabel(event_id, event.get("sub_label"))
        _try_emit(track)
    _frigate_status["last_event"] = datetime.now().strftime("%H:%M:%S")


def _on_mqtt_connect(client, userdata, flags, reason_code, properties=None):
    if getattr(reason_code, "is_failure", False):
        print(f"[MQTT] connection failed: {reason_code}")
        return
    client.subscribe("frigate/events", qos=1)
    _frigate_status["mqtt"] = True
    print("[MQTT] subscribed to frigate/events")


def _on_mqtt_disconnect(client, userdata, flags, reason_code, properties=None):
    _frigate_status["mqtt"] = False


def _on_mqtt_message(client, userdata, message):
    try:
        _ingest_event(json.loads(message.payload.decode("utf-8")), "mqtt")
    except Exception as exc:
        print(f"[MQTT] event error: {exc}")


def _mqtt_listener():
    if mqtt is None:
        print("[MQTT] paho-mqtt is not installed")
        return
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="attendance-bridge")
    client.on_connect, client.on_disconnect, client.on_message = _on_mqtt_connect, _on_mqtt_disconnect, _on_mqtt_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    while True:
        try:
            client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
            client.loop_forever(retry_first_connection=True)
        except Exception as exc:
            _frigate_status["mqtt"] = False
            print(f"[MQTT] unavailable: {exc}")
            time.sleep(3)


def _poll_frigate():
    """Backfill only: MQTT carries the live new/update/end stream."""
    poll_counter = 0
    while True:
        poll_counter += 1
        try:
            version = requests.get(f"{FRIGATE_API}/api/version", timeout=3)
            if version.ok:
                _frigate_status.update(connected=True, frigate_version=version.json().get("version", "unknown"))
            events = requests.get(f"{FRIGATE_API}/api/events?label=person&limit=100", timeout=7)
            if events.ok:
                for event in events.json():
                    _ingest_event(event, "rest")
        except Exception as exc:
            _frigate_status["connected"] = False
            print(f"[Poller] {exc}")
        with _tracks_lock:
            now = time.time()
            for track in list(_tracks.values()):
                _try_emit(track)
            for event_id, track in list(_tracks.items()):
                if now - track.last_update > 600:
                    del _tracks[event_id]
            if poll_counter % 300 == 0:
                _cleanup_person_cooldowns(now)
        time.sleep(2)


@_serialized_db
def _init_person_state_from_db():
    conn = _get_db()
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT ON (person_name) person_name,event_type FROM person_events WHERE date=%s AND person_name IS NOT NULL AND person_name NOT LIKE 'Unknown%%' ORDER BY person_name,id DESC", (datetime.now().strftime("%Y-%m-%d"),))
            rows = cur.fetchall()
        with _person_state_lock:
            for name, direction in rows:
                _person_state[name] = {"inside": direction == "ENTRY", "last_update": time.time()}
    except Exception as exc:
        print(f"[DB] hydrate failed: {exc}")


@_serialized_db
def _init_counter_from_db():
    global _unknown_counter
    conn = _get_db()
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT person_name FROM person_events WHERE person_name LIKE 'Unknown#%%' ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
        if row:
            _unknown_counter = int(row[0].split("#", 1)[1])
    except Exception:
        pass


@app.route("/")
def dashboard():
    with _events_lock:
        events = list(_latest_events[:50])
    return render_template("bridge_dashboard.html", events=events, frigate_status=_frigate_status, person_counts=_person_counts, person_summary=[])


@app.route("/api/events")
def api_events():
    with _events_lock:
        return jsonify(_latest_events[:200])


@app.route("/api/stats")
def api_stats():
    with _counts_lock:
        counts = {key: dict(value) for key, value in _person_counts.items()}
    return jsonify({"person_counts": counts, "total_detected": sum(c["total"] for c in counts.values()), "total_inside": sum(1 for state in _person_state.values() if state["inside"]), "total_entries": sum(c["entered"] for c in counts.values()), "total_exits": sum(c["exited"] for c in counts.values())})


@app.route("/api/person_summary")
@_serialized_db
def api_person_summary():
    date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    conn = _get_db()
    if conn is None:
        return jsonify([])
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT person_name,MIN(CASE WHEN event_type='ENTRY' THEN event_time END) AS first_entry,MAX(CASE WHEN event_type='EXIT' THEN event_time END) AS last_exit,COUNT(*) FILTER (WHERE event_type='ENTRY') AS entry_count,COUNT(*) FILTER (WHERE event_type='EXIT') AS exit_count FROM person_events WHERE date=%s GROUP BY person_name ORDER BY person_name", (date,))
            return jsonify(cur.fetchall())
    except Exception as exc:
        print(f"[DB] summary failed: {exc}")
        return jsonify([])


@app.route("/api/tracks")
def api_tracks():
    with _tracks_lock:
        return jsonify([track.to_dict() for track in _tracks.values()])


@app.route("/api/frigate/status")
def api_frigate_status():
    return jsonify(_frigate_status)


@app.route("/video_feed/<camera_id>")
def video_feed(camera_id):
    # The dashboard's live feed proxy remains intentionally independent of
    # identity/event processing.
    urls = {"cam_entry": os.environ.get("CAM_ENTRY_RTSP"), "cam_exit": os.environ.get("CAM_EXIT_RTSP")}
    if camera_id not in urls or not urls[camera_id]:
        return Response(status=404)
    def generate():
        cap = cv2.VideoCapture(urls[camera_id])
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(1)
                    continue
                _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
        finally:
            cap.release()
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/events/<event_id>/snapshot.jpg")
def event_snapshot(event_id):
    try:
        response = requests.get(f"{FRIGATE_API}/api/events/{event_id}/snapshot.jpg", timeout=5)
        if response.ok:
            return Response(response.content, mimetype="image/jpeg")
    except Exception:
        pass
    return Response(status=404)


if __name__ == "__main__":
    _get_db()
    _init_counter_from_db()
    _init_person_state_from_db()
    threading.Thread(target=_mqtt_listener, daemon=True).start()
    threading.Thread(target=_poll_frigate, daemon=True).start()
    print(f"[Bridge] dashboard listening on :{WEB_PORT}")
    app.run(host="0.0.0.0", port=WEB_PORT, debug=False)
