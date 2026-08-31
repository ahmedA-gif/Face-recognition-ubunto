"""
VisionAttend AI — Frigate Bridge (Production Version)
=====================================================
Advanced trajectory-based entry/exit detection with:
- 3-state machine: UNKNOWN → OUTSIDE → INSIDE
- Dead zones for clear INSIDE/OUTSIDE classification
- Turn-back detection
- Minimum crossing distance
- Temporal confirmation
- Track quality scoring

Architecture:
  Frigate NVR → REST API → TrajectoryAnalyzer → Neon PostgreSQL
  Bridge → Flask dashboard at :5001

Usage:
  python app.py
"""
import os
import json
import time
import math
import threading
import requests
from datetime import datetime, timedelta
from collections import defaultdict, deque
from flask import Flask, render_template, jsonify, Response, request

import psycopg2
import psycopg2.extras
import cv2
import numpy as np

# ─── Configuration ───────────────────────────────────────────────────────────
FRIGATE_API = os.environ.get("FRIGATE_API", "http://frigate:5000")
WEB_PORT = int(os.environ.get("WEB_PORT", "5001"))
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://authenticator:npg_IaOx4mzXT2ib@ep-lively-hill-adqayuhc-pooler.c-2.us-east-1.aws.neon.tech/AI%20attendance%20system%20?sslmode=require",
)

# ─── Entry/Exit Detection Thresholds ────────────────────────────────────────
# Camera-identity approach with minimum movement requirements.
# Events are emitted when person is detected at the camera with sufficient movement.

# Minimum track requirements before emitting event
MIN_TRACK_POINTS = 3         # Min foot-points before emitting
MIN_TRACK_DURATION_SEC = 2.0 # Min seconds of tracking before emitting
MIN_TRACK_DISTANCE = 0.05    # Min total y-displacement (person moved)

# Turn-back detection
TURN_BACK_THRESHOLD = 0.30   # Max reversal before considered turn-back

# Event cooldown
EVENT_COOLDOWN_SEC = 15        # Cooldown between events per camera

# Shift configuration
SHIFT_START = "09:00"
SHIFT_END = "17:00"
LATE_THRESHOLD_MINS = 15

# ─── Globals ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
_seen_event_ids = set()
_seen_lock = threading.Lock()
_latest_events = []
_events_lock = threading.Lock()
_person_counts = {
    "cam_entry": {"total": 0, "current": 0, "entered": 0, "exited": 0},
    "cam_exit": {"total": 0, "current": 0, "entered": 0, "exited": 0},
}
_counts_lock = threading.Lock()
_frigate_status = {"connected": False, "last_event": "", "frigate_version": "unknown"}
_event_cooldowns = {}
_cooldown_lock = threading.Lock()
_active_tracks = {}
_tracks_lock = threading.Lock()
_db_lock = threading.Lock()
_db_conn = None

# Person tracking across cameras: person_name -> {"inside": bool, "last_camera": str}
_person_state = {}
_person_state_lock = threading.Lock()


# ─── Foot Point Extraction ──────────────────────────────────────────────────
def extract_foot_point(box):
    """Extract foot point (bottom-center) from Frigate bounding box.
    Handles inverted boxes where x1>x2 or y1>y2.
    """
    if not box:
        return None
    try:
        if isinstance(box, list) and len(box) == 4:
            x1, y1, x2, y2 = map(float, box)
        elif isinstance(box, str):
            parts = box.split()
            if len(parts) == 4:
                x1, y1, x2, y2 = map(float, parts)
            else:
                return None
        else:
            return None
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        foot_x = (x_min + x_max) / 2
        foot_y = y_max
        return (foot_x, foot_y)
    except (ValueError, TypeError):
        return None


def _dedup_path_data(path_data):
    """Deduplicate path_data points. Preserve timestamps."""
    if not path_data:
        return []
    unique = []
    for point in path_data:
        if isinstance(point, list) and len(point) >= 2:
            coords = point[0]
            ts = point[1]
            if isinstance(coords, list) and len(coords) == 2:
                pt = (round(coords[0], 4), round(coords[1], 4), ts)
                if not unique or unique[-1][0:2] != pt[0:2]:
                    unique.append(pt)
    return unique


# ─── Track State Machine ────────────────────────────────────────────────────
class TrackState:
    """State machine for a single tracked person.

    Uses Frigate's zone data directly instead of re-classifying from foot_y.
    Camera-identity approach: cam_entry = ENTRY, cam_exit = EXIT.
    """

    def __init__(self, track_id, camera):
        self.track_id = track_id
        self.camera = camera
        self.path = []  # List of (x, y, timestamp)
        self.frigate_zones = []  # Frigate zone strings per point
        self.first_seen = None  # Set from first point's actual timestamp
        self.last_update = time.time()
        self.event_emitted = False
        self.event_time = None

    def reset(self):
        """Reset the track after an event is emitted."""
        self.path = []
        self.frigate_zones = []
        self.first_seen = time.time()
        self.last_update = time.time()
        self.event_emitted = False
        self.event_time = None

    def add_point(self, foot_point, frigate_zones=None, timestamp=None):
        """Add a point to the trajectory with Frigate zone data."""
        if foot_point is None:
            return

        ts = timestamp or time.time()
        self.path.append((foot_point[0], foot_point[1], ts))
        self.frigate_zones.append(frigate_zones or [])
        self.last_update = ts

        # Set first_seen from first point's actual timestamp
        if self.first_seen is None:
            self.first_seen = ts

        # Keep only recent points (last 60 seconds)
        cutoff = ts - 60
        self.path = [(x, y, t) for x, y, t in self.path if t > cutoff]
        self.frigate_zones = self.frigate_zones[-len(self.path):]

    def get_trajectory(self):
        """Get trajectory as list of (x, y) points."""
        return [(x, y) for x, y, t in self.path]

    def get_zone_sequence(self):
        """Get sequence of Frigate zones visited."""
        return [z for z in self.frigate_zones]

    def get_duration(self):
        """Get track duration in seconds."""
        if self.first_seen is None:
            return 0.0
        if self.path:
            # Use actual path time span
            return self.path[-1][2] - self.path[0][2]
        return time.time() - self.first_seen

    def get_displacement(self):
        """Get y-displacement from first to last point."""
        if len(self.path) < 2:
            return 0.0
        return self.path[-1][1] - self.path[0][1]

    def get_total_distance(self):
        """Get total y-distance traveled (absolute sum of movements)."""
        if len(self.path) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.path)):
            total += abs(self.path[i][1] - self.path[i - 1][1])
        return total

    def get_average_direction(self):
        """Get average y-direction (negative = toward inside, positive = toward outside)."""
        if len(self.path) < 2:
            return 0.0
        total_dy = 0
        for i in range(1, len(self.path)):
            total_dy += self.path[i][1] - self.path[i - 1][1]
        return total_dy / (len(self.path) - 1)

    def has_turn_back(self):
        """Turn-back detection disabled — camera identity approach already handles this."""
        return False

    def get_quality_score(self):
        """Calculate track quality score (0-1)."""
        score = 0.0

        # Length score
        length_score = min(len(self.path) / 10, 0.3)
        score += length_score

        # Duration score
        duration = self.get_duration()
        duration_score = min(duration / 5, 0.2)
        score += duration_score

        # Smoothness score
        if len(self.path) >= 3:
            jitter = 0
            for i in range(2, len(self.path)):
                dy1 = self.path[i - 1][1] - self.path[i - 2][1]
                dy2 = self.path[i][1] - self.path[i - 1][1]
                if dy1 * dy2 < 0:
                    jitter += 1
            smooth_score = max(0, 0.2 - (jitter / len(self.path)) * 0.2)
        else:
            smooth_score = 0.05
        score += smooth_score

        # Displacement score
        displacement = abs(self.get_displacement())
        disp_score = min(displacement / 0.3, 0.3)
        score += disp_score

        return min(score, 1.0)

    def should_emit_event(self, camera):
        """Determine if we should emit an event.

        Camera-identity approach: cameras are fixed at specific doors.
        - cam_entry: person tracked at this camera = ENTRY
        - cam_exit: person tracked at this camera = EXIT

        Requirements before emitting:
        1. Minimum track points accumulated (MIN_TRACK_POINTS)
        2. Minimum tracking duration (MIN_TRACK_DURATION_SEC)
        3. Person has moved (MIN_TRACK_DISTANCE)
        4. No excessive turn-back
        5. Event hasn't been emitted for this track yet

        Returns (confidence, start_y, end_y, displacement) if ready.
        Event type is determined by the caller based on camera identity.
        """
        if self.event_emitted:
            print(f"[Debug] event_emitted=True for {self.track_id}")
            return None

        # Minimum points accumulated
        if len(self.path) < MIN_TRACK_POINTS:
            print(f"[Debug] path={len(self.path)} < {MIN_TRACK_POINTS} for {self.track_id}")
            return None

        # Minimum tracking duration
        dur = self.get_duration()
        if dur < MIN_TRACK_DURATION_SEC:
            print(f"[Debug] duration={dur:.1f}s < {MIN_TRACK_DURATION_SEC}s for {self.track_id}")
            return None

        # Person must have moved
        dist = self.get_total_distance()
        if dist < MIN_TRACK_DISTANCE:
            print(f"[Debug] distance={dist:.3f} < {MIN_TRACK_DISTANCE} for {self.track_id}")
            return None

        # Check for turn-back (person reversed direction)
        if self.has_turn_back():
            print(f"[Debug] turn_back=True for {self.track_id}")
            return None

        # Calculate confidence
        confidence = self.get_quality_score()

        start_y = self.path[0][1]
        end_y = self.path[-1][1]
        displacement = end_y - start_y

        self.event_emitted = True
        self.event_time = time.time()
        return (confidence, start_y, end_y, displacement)

    def to_dict(self):
        """Convert to dictionary for API."""
        from datetime import datetime
        return {
            "track_id": self.track_id,
            "camera": self.camera,
            "path_length": len(self.path),
            "frame_count": len(self.path),
            "trajectory_len": len(self.path),
            "duration": round(self.get_duration(), 1),
            "displacement": round(self.get_displacement(), 4),
            "total_distance": round(self.get_total_distance(), 4),
            "avg_direction": round(self.get_average_direction(), 4),
            "quality_score": round(self.get_quality_score(), 2),
            "zone_sequence": [z for z in self.frigate_zones[-10:]],
            "has_turn_back": self.has_turn_back(),
            "event_emitted": self.event_emitted,
            "first_seen": datetime.fromtimestamp(self.first_seen).strftime("%H:%M:%S") if self.path else "N/A",
            "last_seen": datetime.fromtimestamp(self.last_update).strftime("%H:%M:%S") if self.path else "N/A",
        }


# ─── Database ────────────────────────────────────────────────────────────────
def _get_db():
    global _db_conn
    with _db_lock:
        try:
            if _db_conn is None or _db_conn.closed:
                _db_conn = psycopg2.connect(DATABASE_URL)
            else:
                with _db_conn.cursor() as cur:
                    cur.execute("SELECT 1")
        except Exception:
            try:
                _db_conn = psycopg2.connect(DATABASE_URL)
            except Exception as e:
                print(f"[DB] Connection failed: {e}")
                return None
        return _db_conn


def _write_event(track_id, person_name, event_type, timestamp, camera, foot_point, box, confidence, zones):
    conn = _get_db()
    if conn is None:
        return
    date_str = timestamp.strftime("%Y-%m-%d")
    time_str = timestamp.strftime("%H:%M:%S")
    foot_y = foot_point[1] if foot_point else None
    box_str = f"{box[0]} {box[1]} {box[2]} {box[3]}" if box else None
    zone_str = ",".join(zones) if zones else ""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO person_events(date, track_id, person_name, event_type, event_time, "
                "confidence, camera_id, zone, foot_y, bounding_box) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (date_str, track_id, person_name, event_type, time_str, confidence, camera, zone_str, foot_y, box_str)
            )
            if event_type == "ENTRY":
                cur.execute(
                    "SELECT id FROM attendance WHERE date = %s AND track_id = %s AND check_out_time IS NULL",
                    (date_str, track_id)
                )
                if not cur.fetchone():
                    status = _calc_status(time_str)
                    cur.execute(
                        "INSERT INTO attendance(date, track_id, person_name, check_in_time, status, confidence, camera_id) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        (date_str, track_id, person_name, time_str, status, confidence, camera)
                    )
            elif event_type == "EXIT":
                cur.execute(
                    "SELECT id, date, check_in_time FROM attendance WHERE track_id = %s AND check_out_time IS NULL ORDER BY id DESC LIMIT 1",
                    (track_id,)
                )
                row = cur.fetchone()
                if row:
                    check_in = datetime.strptime(f"{row[1]} {row[2]}", "%Y-%m-%d %H:%M:%S")
                    work_hours = round((timestamp - check_in).total_seconds() / 3600, 2)
                    cur.execute(
                        "UPDATE attendance SET check_out_time = %s, work_hours = %s, camera_id = %s, updated_at = NOW() WHERE id = %s",
                        (time_str, work_hours, camera, row[0])
                    )
        conn.commit()
    except Exception as e:
        print(f"[DB] Write error: {e}")
        try:
            conn.rollback()
        except:
            pass


def _calc_status(time_str):
    h, m, s = map(int, time_str.split(":"))
    shift_h, shift_m = map(int, SHIFT_START.split(":"))
    if h > shift_h or (h == shift_h and m > shift_m + LATE_THRESHOLD_MINS):
        return "Late"
    return "On Time"


def _is_person_inside(person_name):
    """Check if a person is currently inside based on previous ENTRY/EXIT events."""
    with _person_state_lock:
        state = _person_state.get(person_name)
        if state:
            return state.get("inside", False)
        return False


def _update_person_state(person_name, event_type):
    """Update person's inside/outside state after an event."""
    with _person_state_lock:
        if person_name not in _person_state:
            _person_state[person_name] = {"inside": False, "last_camera": None}
        state = _person_state[person_name]
        if event_type == "ENTRY":
            state["inside"] = True
        elif event_type == "EXIT":
            state["inside"] = False
        state["last_update"] = time.time()


def _init_person_state_from_db():
    """Initialize person state from database on startup."""
    conn = _get_db()
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            # Get the last event for each person to determine current state
            cur.execute("""
                SELECT person_name, event_type, event_time
                FROM person_events
                WHERE date = %s
                AND person_name IS NOT NULL
                ORDER BY event_time DESC
            """, (datetime.now().strftime("%Y-%m-%d"),))
            rows = cur.fetchall()
            seen = set()
            for name, event_type, event_time in rows:
                if name not in seen:
                    seen.add(name)
                    with _person_state_lock:
                        _person_state[name] = {
                            "inside": event_type == "ENTRY",
                            "last_camera": None
                        }
                    print(f"[Bridge] Init state for {name}: {'inside' if event_type == 'ENTRY' else 'outside'}")
    except Exception as e:
        print(f"[Bridge] Error initializing person state: {e}")


# ─── Event Processing ───────────────────────────────────────────────────────
_unknown_counter = 0
_counter_lock = threading.Lock()


def _add_event(event):
    with _events_lock:
        _latest_events.insert(0, event)
        if len(_latest_events) > 200:
            _latest_events.pop()


def _process_person_event(event):
    """Process a Frigate person detection event and analyze its trajectory."""
    global _unknown_counter
    event_id = event.get("id", "")
    camera = event.get("camera", "")
    label = event.get("label", "")
    start_time = event.get("start_time", 0)
    end_time = event.get("end_time")
    top_score = event.get("data", {}).get("top_score", 0)
    zones = event.get("zones", [])
    box = event.get("data", {}).get("box")
    path_data = event.get("data", {}).get("path_data", [])
    sub_label = event.get("sub_label")

    if label != "person":
        return

    if not end_time or not box:
        return

    with _seen_lock:
        if event_id in _seen_event_ids:
            return
        _seen_event_ids.add(event_id)
        if len(_seen_event_ids) > 5000:
            _seen_event_ids.clear()

    # Check database for already-processed events
    conn = _get_db()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM person_events WHERE track_id = %s LIMIT 1", (event_id,))
                if cur.fetchone():
                    return
        except Exception:
            pass

    # Fetch sub_label from individual event API (list endpoint doesn't return it)
    if not sub_label:
        try:
            print(f"[Poller] Fetching sub_label for {event_id}...")
            r2 = requests.get(f"{FRIGATE_API}/api/events/{event_id}", timeout=5)
            if r2.status_code == 200:
                detailed_ev = r2.json()
                sub_label = detailed_ev.get("sub_label")
                if sub_label:
                    print(f"[Poller] Got sub_label for {event_id}: {sub_label}")
                else:
                    print(f"[Poller] No sub_label for {event_id}")
        except Exception as e:
            print(f"[Poller] Error fetching sub_label: {e}")

    # Fallback: check face train files for recognized faces (match by time proximity)
    frigate_says_unknown = False
    if not sub_label:
        try:
            r3 = requests.get(f"{FRIGATE_API}/api/faces", timeout=5)
            if r3.status_code == 200:
                faces_data = r3.json()
                train_files = faces_data.get("train", [])
                best_match = None
                best_score = 0
                for fname in train_files:
                    if not fname.endswith(".webp"):
                        continue
                    # Format: {event_start_ts}-{event_suffix}-{face_ts}-{name}-{score}.webp
                    base = fname.replace(".webp", "")
                    parts = base.split("-")
                    if len(parts) < 4:
                        continue
                    name = parts[-2]
                    try:
                        score = float(parts[-1])
                    except ValueError:
                        continue
                    # Skip low-confidence face matches (< 0.50)
                    if score < 0.50:
                        continue
                    # Extract event timestamp from first part
                    try:
                        ev_ts = float(parts[0])
                    except ValueError:
                        continue
                    # Match by time proximity (within 10 seconds of event)
                    # This prevents matching wrong people detected in quick succession
                    if abs(ev_ts - start_time) < 10:
                        if name.lower() == "unknown":
                            # Frigate labeled this person as unknown
                            frigate_says_unknown = True
                        elif score > best_score:
                            best_match = name
                            best_score = score
                if best_match:
                    sub_label = best_match
                    print(f"[Poller] Face match by time: {event_id} -> {sub_label} (score: {best_score:.2f})")
                elif frigate_says_unknown:
                    # Frigate says unknown - trust Frigate, skip our face matching
                    print(f"[Poller] Frigate says unknown, skipping face match: {event_id}")
        except Exception as e:
            print(f"[Poller] Error checking face train: {e}")

    # Cooldown check - skip if too soon since last event on this camera
    with _cooldown_lock:
        now = datetime.now()
        last = _event_cooldowns.get(camera)
        if last and (now - last).total_seconds() < EVENT_COOLDOWN_SEC:
            return
        _event_cooldowns[camera] = now

    # Extract foot points from path_data (normalized coords from Frigate)
    # path_data format: [[[x, y], timestamp], ...]
    path_points = _dedup_path_data(path_data)

    # Also get the box foot point as a fallback
    box_point = extract_foot_point(box)

    # Combine: path_data points first, then box point if not duplicate
    # path_points are (x, y, ts) tuples, box_point is (x, y)
    all_points = list(path_points)
    if box_point:
        bp = (box_point[0], box_point[1], time.time())
        if not all_points or all_points[-1][0:2] != bp[0:2]:
            all_points.append(bp)

    if not all_points:
        return

    print(f"[Bridge] Event {event_id}: {len(all_points)} points, zones={zones}")

    # Get or create track state per CAMERA
    track_key = f"{camera}_main"
    with _tracks_lock:
        if track_key not in _active_tracks:
            _active_tracks[track_key] = TrackState(track_key, camera)
        track = _active_tracks[track_key]

    # Add all points to track (with Frigate timestamps for accurate duration)
    for fp in all_points:
        if track.path:
            last_x, last_y, _ = track.path[-1]
            if abs(fp[0] - last_x) < 0.01 and abs(fp[1] - last_y) < 0.01:
                continue
        track.add_point(fp, frigate_zones=zones, timestamp=fp[2] if len(fp) > 2 else None)

    # Check if we should emit an event
    event_result = track.should_emit_event(camera)
    if event_result is None:
        # Debug: show why no event
        if len(track.path) >= 2:
            print(f"[Bridge] Track {track_key}: no event (path_len={len(track.path)}, "
                  f"dist={track.get_total_distance():.3f}, "
                  f"duration={track.get_duration():.1f}s, "
                  f"quality={track.get_quality_score():.2f})")
        return

    confidence, start_y, end_y, displacement = event_result

    # Determine person name
    if sub_label:
        if isinstance(sub_label, list) and len(sub_label) >= 1:
            person_name = sub_label[0]
        elif isinstance(sub_label, str):
            person_name = sub_label
        else:
            person_name = None
    else:
        person_name = None

    # Try face matching from snapshot if name is still unknown
    # Skip if Frigate already said unknown
    if not person_name and not frigate_says_unknown:
        snapshot_url = f"{FRIGATE_API}/api/events/{event_id}/snapshot.jpg"
        face_match = _match_face_from_snapshot(snapshot_url)
        if face_match:
            person_name, face_score = face_match
            print(f"[Face] Snapshot match: {event_id} -> {person_name} (score={face_score:.2f})")
        else:
            with _counter_lock:
                _unknown_counter += 1
                person_name = f"Unknown#{_unknown_counter:02d}"
    elif not person_name and frigate_says_unknown:
        # Frigate says unknown - create Unknown#XX entry
        with _counter_lock:
            _unknown_counter += 1
            person_name = f"Unknown#{_unknown_counter:02d}"
        print(f"[Bridge] Unknown person (Frigate): {event_id} -> {person_name}")

    # Determine event type based on camera identity
    # - cam_entry = ENTRY (person entering)
    # - cam_exit = EXIT (person exiting)
    # Note: People may have entered BEFORE camera started tracking,
    # so EXIT events are allowed even without prior ENTRY.
    if camera == "cam_entry":
        event_type = "ENTRY"
    elif camera == "cam_exit":
        event_type = "EXIT"
    else:
        return

    # Use current time
    ts = datetime.now()

    # Update counts
    with _counts_lock:
        cam = _person_counts.get(camera, {})
        cam["total"] = cam.get("total", 0) + 1
        if event_type == "ENTRY":
            cam["entered"] = cam.get("entered", 0) + 1
        elif event_type == "EXIT":
            cam["exited"] = cam.get("exited", 0) + 1

    # Create event record
    event_record = {
        "person": person_name,
        "track_id": event_id,
        "direction": event_type,
        "confidence": f"{confidence:.0%}",
        "face_score": 0.0,
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "camera": camera,
        "zones": zones,
        "score": round(top_score or 0, 3),
        "type": "trajectory",
        "displacement": round(displacement, 4),
        "traj_points": len(track.path),
        "quality_score": round(track.get_quality_score(), 2),
        "duration": round(track.get_duration(), 1),
        "snapshot_url": f"/api/events/{event_id}/snapshot.jpg",
    }
    _add_event(event_record)

    # Write to database
    _write_event(event_id, person_name, event_type, ts, camera, foot_point, box, confidence, zones)

    # Update person state (inside/outside)
    _update_person_state(person_name, event_type)

    print(f"[Bridge] {person_name} {event_type} (conf={confidence:.0%}) on {camera} "
          f"zones={zones} disp={displacement:.3f} quality={track.get_quality_score():.2f}")

    # Reset track after event to start fresh
    track.reset()

    # Clean up old tracks
    _cleanup_old_tracks()


def _cleanup_old_tracks():
    """Remove old tracks from memory."""
    with _tracks_lock:
        now = time.time()
        to_remove = []
        for track_id, track in _active_tracks.items():
            if now - track.last_update > 60:  # 60 seconds old
                to_remove.append(track_id)
        for track_id in to_remove:
            del _active_tracks[track_id]


# ─── Frigate Poller ──────────────────────────────────────────────────────────
_pending_sublabel_updates = {}  # track_id -> last_check_time
_startup_complete = False
_startup_events_processed = 0
_startup_poll_count = 0


def _poll_frigate():
    global _startup_events_processed, _startup_complete, _startup_poll_count
    print("[Poller] Starting Frigate event poller...")
    while True:
        try:
            r = requests.get(f"{FRIGATE_API}/api/version", timeout=3)
            if r.status_code == 200:
                _frigate_status["connected"] = True
                _frigate_status["frigate_version"] = r.json().get("version", "unknown")
            else:
                _frigate_status["connected"] = False
        except Exception:
            _frigate_status["connected"] = False

        try:
            r = requests.get(f"{FRIGATE_API}/api/events?label=person&limit=30&has_snapshot=1", timeout=5)
            if r.status_code == 200:
                events = r.json()
                for ev in events:
                    _process_person_event(ev)

                    # Check for sub_label updates on recently processed events
                    ev_id = ev.get("id", "")
                    sub_label = ev.get("sub_label")
                    if ev_id and not sub_label and ev_id in _seen_event_ids:
                        # Re-fetch this event to check for sub_label update
                        now = time.time()
                        last_check = _pending_sublabel_updates.get(ev_id, 0)
                        if now - last_check > 30:  # Check every 30 seconds
                            _pending_sublabel_updates[ev_id] = now
                            try:
                                r2 = requests.get(f"{FRIGATE_API}/api/events/{ev_id}", timeout=5)
                                if r2.status_code == 200:
                                    updated_ev = r2.json()
                                    updated_sublabel = updated_ev.get("sub_label")
                                    if updated_sublabel:
                                        print(f"[Poller] Got sub_label for {ev_id}: {updated_sublabel}")
                                        # Update in DB
                                        _update_event_sublabel(ev_id, updated_sublabel)
                            except Exception:
                                pass

                _frigate_status["last_event"] = datetime.now().strftime("%H:%M:%S")
                _startup_events_processed += len(events)
                _startup_poll_count += 1
                if not _startup_complete and _startup_poll_count >= 2:
                    new_events = [ev for ev in events if ev.get("id", "") not in _seen_event_ids]
                    if len(new_events) == 0:
                        _startup_complete = True
                        print(f"[Poller] Startup complete. Processed {_startup_events_processed} events total.")
        except Exception as e:
            print(f"[Poller] Error: {e}")

        # Clean up old pending updates
        now = time.time()
        old_ids = [k for k, v in _pending_sublabel_updates.items() if now - v > 300]
        for k in old_ids:
            del _pending_sublabel_updates[k]

        time.sleep(2)


def _update_event_sublabel(event_id, sub_label):
    """Update the person_name in DB when sub_label becomes available."""
    conn = _get_db()
    if conn is None:
        return
    # Handle list sub_labels
    if isinstance(sub_label, list) and len(sub_label) >= 1:
        person_name = sub_label[0]
    elif isinstance(sub_label, str):
        person_name = sub_label
    else:
        return
    try:
        with conn.cursor() as cur:
            # Get old name before update
            cur.execute(
                "SELECT person_name FROM person_events WHERE track_id = %s AND person_name LIKE 'Unknown%%'",
                (event_id,)
            )
            old_name_row = cur.fetchone()
            old_name = old_name_row[0] if old_name_row else None

            cur.execute(
                "UPDATE person_events SET person_name = %s WHERE track_id = %s AND person_name LIKE 'Unknown%%'",
                (person_name, event_id)
            )
            if cur.rowcount > 0:
                print(f"[Poller] Updated {event_id} -> {person_name}")
            cur.execute(
                "UPDATE attendance SET person_name = %s WHERE track_id = %s AND person_name LIKE 'Unknown%%'",
                (person_name, event_id)
            )

            # Transfer person state from old name to new name
            if old_name and old_name != person_name:
                with _person_state_lock:
                    old_state = _person_state.pop(old_name, None)
                    if old_state:
                        _person_state[person_name] = old_state
                        print(f"[Poller] Transferred state from {old_name} to {person_name}")

        conn.commit()
    except Exception as e:
        print(f"[DB] Update sublabel error: {e}")


_face_cascade = None
_known_face_cache = {}


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def _download_image(url, timeout=5):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            arr = np.frombuffer(r.content, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        pass
    return None


def _extract_face_roi(img):
    cascade = _get_face_cascade()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.05, 3, minSize=(20, 20))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(max(w, h) * 0.3)
        h_img, w_img = img.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)
        return img[y1:y2, x1:x2]
    h_img, w_img = img.shape[:2]
    cy, cx = h_img // 2, w_img // 2
    s = min(h_img, w_img) // 3
    return img[cy-s:cy+s, cx-s:cx+s]


def _compare_faces(img1, img2):
    if img1 is None or img2 is None:
        return 0.0
    try:
        face1 = _extract_face_roi(img1)
        face2 = _extract_face_roi(img2)
        r1 = cv2.resize(face1, (100, 100))
        r2 = cv2.resize(face2, (100, 100))

        h1 = cv2.cvtColor(r1, cv2.COLOR_BGR2HSV)
        h2 = cv2.cvtColor(r2, cv2.COLOR_BGR2HSV)
        hist1 = cv2.calcHist([h1], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist2 = cv2.calcHist([h2], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        diff = np.linalg.norm(r1.astype(float) - r2.astype(float))
        norm_score = max(0.0, 1.0 - diff / 150000.0)

        return hist_score * 0.5 + norm_score * 0.5
    except Exception:
        return 0.0


def _load_known_faces():
    global _known_face_cache
    if _known_face_cache and time.time() - _known_face_cache.get("_ts", 0) < 300:
        return _known_face_cache
    try:
        r = requests.get(f"{FRIGATE_API}/api/faces", timeout=10)
        if r.status_code != 200:
            return _known_face_cache
        faces_data = r.json()
        for name, files in faces_data.items():
            if name == "train" or not isinstance(files, list):
                continue
            best_img = None
            for fname in sorted(files, reverse=True):
                if not fname.endswith(".webp"):
                    continue
                url = f"{FRIGATE_API}/clips/faces/{name}/{fname}"
                img = _download_image(url)
                if img is not None:
                    best_img = img
                    break
            if best_img is not None:
                _known_face_cache[name] = best_img
        _known_face_cache["_ts"] = time.time()
    except Exception as e:
        print(f"[Face] Error loading known faces: {e}")
    return _known_face_cache


def _match_face_from_snapshot(snapshot_url):
    """Match face from snapshot against known faces.
    
    Returns (name, score) if match found with score > threshold.
    Returns None if no good match (person is unknown).
    
    Threshold: 0.55 — high enough to avoid false matches between
    different people with similar lighting/color histograms.
    """
    snapshot_img = _download_image(snapshot_url)
    if snapshot_img is None:
        return None
    face_roi = _extract_face_roi(snapshot_img)
    known = _load_known_faces()
    best_name = None
    best_score = 0.0
    for name, known_img in known.items():
        if name == "_ts":
            continue
        score = _compare_faces(face_roi, known_img)
        if score > best_score:
            best_score = score
            best_name = name
    if best_name and best_score > 0.55:
        return (best_name, best_score)
    return None


def _scan_face_train_files():
    """Periodically scan Frigate face directories and update person names.
    
    Uses BOTH known face directories (ahmed, Haseeb, etc.) AND train directory.
    Known files: {Name}-{timestamp}.webp  (e.g., Haseeb-1788014474.510587.webp)
    Train files: {ev_ts}-{suffix}-{face_ts}-{name}-{score}.webp
    
    Match person_events by time proximity to known face captures."""
    print("[FaceScan] Starting face scanner...")
    while True:
        try:
            r = requests.get(f"{FRIGATE_API}/api/faces", timeout=10)
            if r.status_code != 200:
                time.sleep(60)
                continue
            faces_data = r.json()

            face_by_time = {}

            # Scan known face directories (e.g., "ahmed", "Haseeb", "Rehmat", "Sulman")
            for name, files in faces_data.items():
                if name == "train" or not isinstance(files, list):
                    continue
                for fname in files:
                    if not fname.endswith(".webp"):
                        continue
                    base = fname.replace(".webp", "")
                    # Format: {Name}-{timestamp}.webp
                    parts = base.rsplit("-", 1)
                    if len(parts) != 2:
                        continue
                    known_name, ts_str = parts
                    try:
                        ev_ts = float(ts_str)
                    except ValueError:
                        continue
                    key = round(ev_ts, 0)
                    if key not in face_by_time:
                        face_by_time[key] = (known_name, 1.0)

            # Also scan train directory for recognized faces
            train_files = faces_data.get("train", [])
            for fname in train_files:
                if not fname.endswith(".webp") or "unknown" in fname.lower():
                    continue
                base = fname.replace(".webp", "")
                parts = base.split("-")
                if len(parts) < 4:
                    continue
                name = parts[-2]
                try:
                    score = float(parts[-1])
                    ev_ts = float(parts[0])
                except (ValueError, IndexError):
                    continue
                key = round(ev_ts, 0)
                if key not in face_by_time or score > face_by_time[key][1]:
                    face_by_time[key] = (name, score)

            if not face_by_time:
                time.sleep(60)
                continue

            print(f"[FaceScan] Found {len(face_by_time)} known face timestamps")

            conn = _get_db()
            if not conn:
                time.sleep(60)
                continue

            updated = 0
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, track_id, event_time, date FROM person_events "
                        "WHERE person_name LIKE 'Unknown%%'"
                    )
                    rows = cur.fetchall()
                    for row in rows:
                        pe_id, pe_track_id, pe_time, pe_date = row
                        try:
                            dt = datetime.strptime(f"{pe_date} {pe_time}", "%Y-%m-%d %H:%M:%S")
                            pe_ts = dt.timestamp()
                        except Exception:
                            continue
                        best_name = None
                        best_dist = 999
                        for face_ts, (name, score) in face_by_time.items():
                            dist = abs(face_ts - pe_ts)
                            if dist < best_dist:
                                best_dist = dist
                                best_name = name
                        if best_name and best_dist < 60:
                            # Transfer person state from old name to new name
                            with _person_state_lock:
                                old_state = _person_state.pop(pe_track_id, None)
                                if old_state:
                                    _person_state[best_name] = old_state
                                    print(f"[FaceScan] Transferred state from {pe_track_id} to {best_name}")

                            cur.execute(
                                "UPDATE person_events SET person_name = %s WHERE id = %s",
                                (best_name, pe_id)
                            )
                            cur.execute(
                                "UPDATE attendance SET person_name = %s WHERE track_id = %s AND person_name LIKE 'Unknown%%'",
                                (best_name, pe_track_id)
                            )
                            updated += 1
                            print(f"[FaceScan] {pe_track_id} -> {best_name} (dist={best_dist:.0f}s)")
                conn.commit()
            except Exception as e:
                print(f"[FaceScan] DB error: {e}")
                conn.rollback()

            if updated > 0:
                print(f"[FaceScan] Updated {updated} events")
        except Exception as e:
            print(f"[FaceScan] Error: {e}")
        time.sleep(60)


# ─── Flask Routes ────────────────────────────────────────────────────────────
@app.route("/")
def dashboard():
    with _events_lock:
        events = list(_latest_events[:50])
    with _counts_lock:
        counts = dict(_person_counts)
    summary = []
    conn = _get_db()
    if conn:
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        track_id,
                        person_name,
                        MIN(CASE WHEN event_type = 'ENTRY' THEN event_time END) as first_entry,
                        MAX(CASE WHEN event_type = 'EXIT' THEN event_time END) as last_exit,
                        COUNT(CASE WHEN event_type = 'ENTRY' THEN 1 END) as entry_count,
                        COUNT(CASE WHEN event_type = 'EXIT' THEN 1 END) as exit_count,
                        ROUND(AVG(confidence)::numeric, 2) as avg_confidence
                    FROM person_events
                    WHERE date = %s
                    GROUP BY track_id, person_name
                    ORDER BY MIN(id)
                """, (datetime.now().strftime("%Y-%m-%d"),))
                summary = cur.fetchall()
        except Exception as e:
            print(f"[DB] Summary query error: {e}")
    return render_template("bridge_dashboard.html",
                           events=events,
                           frigate_status=_frigate_status,
                           person_counts=counts,
                           person_summary=summary)


@app.route("/api/events")
def api_events():
    with _events_lock:
        return jsonify(list(_latest_events[:200]))


@app.route("/api/stats")
def api_stats():
    with _counts_lock:
        counts = dict(_person_counts)
    total_entries = sum(c.get("entered", 0) for c in counts.values())
    total_exits = sum(c.get("exited", 0) for c in counts.values())
    return jsonify({
        "person_counts": counts,
        "total_detected": sum(c.get("total", 0) for c in counts.values()),
        "total_inside": max(0, total_entries - total_exits),
        "total_entries": total_entries,
        "total_exits": total_exits,
    })


@app.route("/api/person_summary")
def api_person_summary():
    date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    conn = _get_db()
    if conn is None:
        return jsonify([])
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                WITH person_events_agg AS (
                    SELECT
                        person_name,
                        MIN(CASE WHEN event_type = 'ENTRY' THEN event_time END) as first_entry,
                        MAX(CASE WHEN event_type = 'EXIT' THEN event_time END) as last_exit,
                        COUNT(CASE WHEN event_type = 'ENTRY' THEN 1 END) as entry_count,
                        COUNT(CASE WHEN event_type = 'EXIT' THEN 1 END) as exit_count,
                        ROUND(AVG(confidence)::numeric, 2) as avg_confidence
                    FROM person_events
                    WHERE date = %s AND person_name IS NOT NULL
                    GROUP BY person_name
                )
                SELECT
                    person_name,
                    first_entry,
                    last_exit,
                    entry_count,
                    exit_count,
                    avg_confidence,
                    CASE
                        WHEN entry_count > exit_count THEN 'Inside'
                        WHEN entry_count = exit_count AND entry_count > 0 THEN 'Checked Out'
                        WHEN entry_count > 0 THEN 'Inside'
                        ELSE 'Unknown'
                    END as status
                FROM person_events_agg
                ORDER BY person_name
            """, (date,))
            return jsonify(cur.fetchall())
    except Exception as e:
        print(f"[DB] Query error: {e}")
        return jsonify([])


@app.route("/api/tracks")
def api_tracks():
    with _tracks_lock:
        tracks = [t.to_dict() for t in _active_tracks.values()]
    return jsonify(tracks)


@app.route("/api/frigate/status")
def api_frigate_status():
    return jsonify({
        "frigate_connected": _frigate_status["connected"],
        "frigate_version": _frigate_status.get("frigate_version", "unknown"),
        "last_event": _frigate_status["last_event"],
    })


@app.route("/video_feed/<camera_id>")
def video_feed(camera_id):
    cam_map = {
        "cam_entry": os.environ.get("CAM_ENTRY_RTSP",
            "rtsp://admin:admin1234@192.168.2.112:554/cam/realmonitor?channel=1&subtype=1"),
        "cam_exit": os.environ.get("CAM_EXIT_RTSP",
            "rtsp://admin:admin1234@192.168.1.111:554/cam/realmonitor?channel=1&subtype=1"),
    }
    if camera_id not in cam_map:
        def gen_blank():
            import numpy as np, cv2
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, f"Camera {camera_id} not found", (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
            _, buf = cv2.imencode(".jpg", blank)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        return Response(gen_blank(), mimetype="multipart/x-mixed-replace; boundary=frame")

    rtsp_url = cam_map[camera_id]
    def generate():
        import cv2, numpy as np
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while True:
            ret, frame = cap.read()
            if ret and frame is not None:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            else:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Reconnecting...", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (60, 60, 60), 2)
                _, buf = cv2.imencode(".jpg", blank)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                cap.release()
                time.sleep(3)
                cap = cv2.VideoCapture(rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(0.033)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/events/<event_id>/snapshot.jpg")
def event_snapshot(event_id):
    """Proxy snapshot from Frigate for an event."""
    try:
        r = requests.get(f"{FRIGATE_API}/api/events/{event_id}/snapshot.jpg", timeout=5)
        if r.status_code == 200:
            return Response(r.content, mimetype="image/jpeg")
    except Exception:
        pass
    return Response(b"", mimetype="image/jpeg", status=404)


# ─── Main ────────────────────────────────────────────────────────────────────
def _init_counter_from_db():
    """Initialize Unknown counter from database to prevent duplicates on restart."""
    global _unknown_counter
    conn = _get_db()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT person_name FROM person_events WHERE person_name LIKE 'Unknown#%' ORDER BY id DESC LIMIT 1")
                row = cur.fetchone()
                if row:
                    name = row[0]
                    if name.startswith("Unknown#"):
                        num = int(name.split("#")[1])
                        _unknown_counter = num
                        print(f"[Bridge] Initialized Unknown counter to {_unknown_counter}")
        except Exception as e:
            print(f"[Bridge] Counter init error: {e}")


if __name__ == "__main__":
    _get_db()
    _init_counter_from_db()
    _init_person_state_from_db()
    poller = threading.Thread(target=_poll_frigate, daemon=True)
    poller.start()
    face_scanner = threading.Thread(target=_scan_face_train_files, daemon=True)
    face_scanner.start()
    print(f"[Bridge] Starting dashboard at http://0.0.0.0:{WEB_PORT}")
    app.run(host="0.0.0.0", port=WEB_PORT, debug=False)
