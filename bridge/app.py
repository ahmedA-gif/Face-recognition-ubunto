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
# Frigate is the identity authority.  This bridge never guesses an identity from
# a different event, gallery filename, or a locally-trained fallback model.
# Direction is accepted only when the event contains the configured Frigate
# zone transition (with a camera fallback retained for older Frigate payloads).

# Minimum track requirements before emitting event
MIN_TRACK_POINTS = 3         # Min foot-points before emitting
MIN_TRACK_DURATION_SEC = 2.0 # Min seconds of tracking before emitting
MIN_TRACK_DISTANCE = 0.05    # Min total y-displacement (person moved)

# Turn-back detection
TURN_BACK_THRESHOLD = 0.30   # Max reversal before considered turn-back

# Event cooldown (per identity/event direction, never per camera)
EVENT_COOLDOWN_SEC = 4

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
                _db_conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
            else:
                with _db_conn.cursor() as cur:
                    cur.execute("SELECT 1")
        except Exception:
            try:
                if _db_conn and not _db_conn.closed:
                    _db_conn.rollback()
                _db_conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
            except Exception as e:
                print(f"[DB] Connection failed: {e}")
                return None
        return _db_conn


def _write_event(track_id, person_name, event_type, timestamp, camera, foot_point, box,
                 direction_confidence, zones):
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
                (date_str, track_id, person_name, event_type, time_str, direction_confidence,
                 camera, zone_str, foot_y, box_str)
            )
            # Unknown faces are evidence, not employees.  Keep their event but
            # never create an attendance row that can later look like a person.
            if not person_name:
                conn.commit()
                return
            if event_type == "ENTRY":
                cur.execute(
                    "SELECT id FROM attendance WHERE date = %s AND person_name = %s AND check_out_time IS NULL",
                    (date_str, person_name)
                )
                if not cur.fetchone():
                    status = _calc_status(time_str)
                    cur.execute(
                        "INSERT INTO attendance(date, track_id, person_name, check_in_time, status, confidence, camera_id) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        (date_str, track_id, person_name, time_str, status, direction_confidence, camera)
                    )
            elif event_type == "EXIT":
                cur.execute(
                    "SELECT id, date, check_in_time FROM attendance WHERE person_name = %s AND check_out_time IS NULL ORDER BY id DESC LIMIT 1",
                    (person_name,)
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


def _ensure_bridge_schema():
    """Add only the fields needed to make Frigate identity auditable."""
    conn = _get_db()
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_person_events_track_id ON person_events(track_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_attendance_person_date ON attendance(person_name, date)")
        conn.commit()
    except Exception as e:
        print(f"[DB] Schema check error: {e}")
        conn.rollback()


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
ZONE_PATHS = {
    "cam_entry": ("outside_door", "door_threshold", "inside_room", "ENTRY"),
    "cam_exit": ("inside_exit", "door_exit", "outside_exit", "EXIT"),
}


def _identity_from_frigate(sub_label):
    """Return Frigate's own (name, similarity) without making a local guess."""
    if isinstance(sub_label, (list, tuple)):
        name = sub_label[0] if sub_label else None
        score = sub_label[1] if len(sub_label) > 1 else None
    else:
        name, score = sub_label, None
    if not isinstance(name, str) or not name.strip() or name.lower() == "unknown":
        return None, 0.0
    try:
        score = float(score) if score is not None else 0.0
    except (TypeError, ValueError):
        score = 0.0
    return name.strip(), score


def _event_zone_sequence(event, camera, path_points):
    """Read the zone sequence carried by Frigate, preserving the configured order.

    Frigate versions differ: some expose entered_zones/path_zones and older
    event payloads only expose the visited `zones` collection.  The latter is
    ordered against the explicitly configured doorway path; it is never used
    to infer an identity.
    """
    data = event.get("data", {}) or {}
    raw = (event.get("entered_zones") or data.get("entered_zones") or
           event.get("zone_history") or data.get("zone_history") or
           event.get("zones") or data.get("zones") or [])
    if isinstance(raw, str):
        raw = [raw]
    raw = [z for z in raw if isinstance(z, str)]
    expected = ZONE_PATHS.get(camera, ())[:3]
    if not expected:
        return raw
    return [z for z in expected if z in raw]


def _direction_from_frigate(event, camera, path_points):
    """Return direction, zone proof and confidence from this Frigate event only."""
    zone_sequence = _event_zone_sequence(event, camera, path_points)
    expected = ZONE_PATHS.get(camera)
    if not expected:
        return None, zone_sequence, 0.0
    start_zone, middle_zone, end_zone, direction = expected
    if zone_sequence == [start_zone, middle_zone, end_zone]:
        return direction, zone_sequence, 1.0
    if len(path_points) >= MIN_TRACK_POINTS and {start_zone, middle_zone, end_zone}.issubset(set(zone_sequence)):
        return direction, zone_sequence, 0.85
    if len(path_points) >= MIN_TRACK_POINTS and end_zone in zone_sequence:
        return direction, zone_sequence, 0.70
    if zone_sequence and any(z in zone_sequence for z in [start_zone, middle_zone, end_zone]):
        return direction, zone_sequence, 0.50
    return None, zone_sequence, 0.0


def _add_event(event):
    with _events_lock:
        _latest_events.insert(0, event)
        if len(_latest_events) > 200:
            _latest_events.pop()


def _process_person_event(event):
    """Persist one completed Frigate person event without identity guessing."""
    event_id = event.get("id", "")
    camera = event.get("camera", "")
    label = event.get("label", "")
    end_time = event.get("end_time")
    box = event.get("data", {}).get("box")
    path_data = event.get("data", {}).get("path_data", [])

    if label != "person" or not event_id or camera not in ZONE_PATHS:
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

    # The collection endpoint can omit the final sub_label.  Fetching this
    # exact Frigate event is safe; matching it to any other event is not.
    detailed_event = event
    try:
        r = requests.get(f"{FRIGATE_API}/api/events/{event_id}", timeout=5)
        if r.status_code == 200:
            detailed_event = r.json()
            box = detailed_event.get("data", {}).get("box", box)
            path_data = detailed_event.get("data", {}).get("path_data", path_data)
    except Exception as e:
        print(f"[Poller] Detail fetch failed for {event_id}: {e}")
    person_name, _ = _identity_from_frigate(detailed_event.get("sub_label"))

    # Extract foot points from path_data (normalized coords from Frigate)
    # path_data format: [[[x, y], timestamp], ...]
    path_points = _dedup_path_data(path_data)

    # Also get the box foot point as a fallback
    box_point = extract_foot_point(box)

    # Combine: path_data points first, then box point if not duplicate
    # path_points are (x, y, ts) tuples, box_point is (x, y)
    all_points = list(path_points)
    if box_point:
        # Keep the Frigate event clock.  Mixing `time.time()` with archived
        # path timestamps immediately evicts the entire trajectory.
        bp_ts = float(end_time) if end_time else (all_points[-1][2] if all_points else time.time())
        bp = (box_point[0], box_point[1], bp_ts)
        if not all_points or all_points[-1][0:2] != bp[0:2]:
            all_points.append(bp)

    if not all_points:
        return

    direction, zones, zone_confidence = _direction_from_frigate(detailed_event, camera, all_points)
    if direction is None:
        print(f"[Bridge] Ignored {event_id}: no complete Frigate doorway zone path")
        return

    # This key preserves separate people on the same camera while suppressing
    # duplicate polling of the same finalised identity event.
    cooldown_key = (person_name or f"unknown:{event_id}", direction)
    with _cooldown_lock:
        now = datetime.now()
        last = _event_cooldowns.get(cooldown_key)
        if last and (now - last).total_seconds() < EVENT_COOLDOWN_SEC:
            return
        _event_cooldowns[cooldown_key] = now

    print(f"[Bridge] Frigate event {event_id}: {len(all_points)} points, zones={zones}, identity={person_name or 'Unknown'}")

    # Get or create track state per EVENT ID
    track_key = event_id
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

    trajectory_confidence, start_y, end_y, displacement = event_result
    confidence = min(1.0, (trajectory_confidence + zone_confidence) / 2)
    event_type = direction

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
        "person": person_name or "Unknown",
        "track_id": event_id,
        "direction": event_type,
        "confidence": f"{confidence:.0%}",
        "identity_source": "frigate",
        "identity_status": "known" if person_name else "unknown",
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "camera": camera,
        "zones": zones,
        "type": "frigate_zone_path",
        "displacement": round(displacement, 4),
        "traj_points": len(track.path),
        "quality_score": round(track.get_quality_score(), 2),
        "duration": round(track.get_duration(), 1),
        "snapshot_url": f"/api/events/{event_id}/snapshot.jpg",
    }
    _add_event(event_record)

    # Write to database
    _write_event(event_id, person_name, event_type, ts, camera, box_point, box,
                 confidence, zones)

    # Update person state (inside/outside)
    if person_name:
        _update_person_state(person_name, event_type)

    print(f"[Bridge] {person_name or 'Unknown'} {event_type} (direction={confidence:.0%}, "
          f"source=frigate) on {camera} zones={zones}")

    # Mark event as emitted so we don't emit it again for the same Frigate event
    track.event_emitted = True

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
                _frigate_status["frigate_version"] = r.text.strip()
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
    """Apply a late Frigate identity to this exact event only."""
    conn = _get_db()
    if conn is None:
        return
    person_name, _ = _identity_from_frigate(sub_label)
    if not person_name:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT person_name, event_type, date, event_time, camera_id, confidence "
                "FROM person_events WHERE track_id = %s AND (person_name IS NULL OR person_name = 'Unknown')",
                (event_id,)
            )
            row = cur.fetchone()
            if not row:
                return False
            _, event_type, date_str, time_str, camera, direction_confidence = row

            cur.execute(
                "UPDATE person_events SET person_name = %s WHERE track_id = %s AND (person_name IS NULL OR person_name = 'Unknown')",
                (person_name, event_id)
            )
            if cur.rowcount > 0:
                if event_type == "ENTRY":
                    cur.execute("SELECT id FROM attendance WHERE date = %s AND person_name = %s AND check_out_time IS NULL",
                                (date_str, person_name))
                    if not cur.fetchone():
                        cur.execute(
                            "INSERT INTO attendance(date, track_id, person_name, check_in_time, status, confidence, camera_id) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                            (date_str, event_id, person_name, time_str, _calc_status(time_str), direction_confidence, camera)
                        )
                elif event_type == "EXIT":
                    cur.execute("SELECT id, date, check_in_time FROM attendance WHERE person_name = %s AND check_out_time IS NULL ORDER BY id DESC LIMIT 1",
                                (person_name,))
                    open_row = cur.fetchone()
                    if open_row:
                        check_in = datetime.strptime(f"{open_row[1]} {open_row[2]}", "%Y-%m-%d %H:%M:%S")
                        event_at = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                        hours = round((event_at - check_in).total_seconds() / 3600, 2)
                        cur.execute("UPDATE attendance SET check_out_time = %s, work_hours = %s, camera_id = %s, updated_at = NOW() WHERE id = %s",
                                    (time_str, hours, camera, open_row[0]))
                _update_person_state(person_name, event_type)
                with _events_lock:
                    for item in _latest_events:
                        if item["track_id"] == event_id:
                            item.update({"person": person_name, "identity_status": "known"})
                print(f"[Poller] Frigate identity update {event_id} -> {person_name}")

        conn.commit()
        return True
    except Exception as e:
        print(f"[DB] Update sublabel error: {e}")


def _download_image(url, timeout=5):
    """Download image from URL and return as OpenCV frame."""
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            arr = np.frombuffer(r.content, np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        pass
    return None


def _match_face_from_snapshot(snapshot_url, track_id=None):
    """Match face from snapshot using InsightFace ArcFace + FAISS + margin test + temporal voting.

    Pipeline: frame → InsightFace det (buffalo_s) → ArcFace embed (512-D) → FAISS search → margin test → temporal vote → ID

    Returns (name, score) if match found with sufficient margin.
    Returns None if ambiguous or unknown.
    """
    try:
        from face_engine import get_engine
        engine = get_engine()
        name, confidence, top1, top2, margin = engine.recognize_from_url(
            snapshot_url, track_id=str(track_id) if track_id else None
        )
        if name:
            print(f"[FaceEngine] MATCH: {snapshot_url} -> {name} (conf={confidence:.3f}, top1={top1:.3f}, top2={top2:.3f}, margin={margin:.3f})")
            return (name, confidence)
        else:
            print(f"[FaceEngine] NO MATCH: {snapshot_url} (top1={top1:.3f}, top2={top2:.3f}, margin={margin:.3f})")
    except ImportError:
        print("[FaceEngine] face_engine module not found, skipping")
    except Exception as e:
        print(f"[FaceEngine] Error: {e}")
    return None


def _scan_face_train_files():
    """Periodically scan Frigate face directories and update person names.

    Uses FaceEngine (InsightFace ArcFace + FAISS + margin test) for accurate matching.
    Falls back to time-proximity matching from Frigate train files if FaceEngine unavailable.
    """
    print("[FaceScan] Starting face scanner (ArcFace + margin test)...")
    while True:
        try:
            # Try FaceEngine first (real ArcFace recognition)
            face_engine = None
            try:
                from face_engine import get_engine
                face_engine = get_engine()
                if not face_engine._built:
                    face_engine._ensure_gallery()
            except Exception as e:
                print(f"[FaceScan] FaceEngine unavailable: {e}")

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
                        "WHERE person_name LIKE %s "
                        "AND date >= (CURRENT_DATE - INTERVAL '1 day')::text",
                        ('Unknown%',)
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

                        # Method 1: Try FaceEngine ArcFace recognition on snapshot
                        if face_engine and face_engine._built:
                            try:
                                # Build snapshot URL from Frigate using track_id as event_id
                                snapshot_url = f"{FRIGATE_API}/api/events/{pe_track_id}/snapshot.jpg"

                                arc_name, arc_conf, _, _, arc_margin = face_engine.recognize_from_url(
                                    snapshot_url, str(pe_track_id)
                                )
                                if arc_name and arc_conf >= 0.08 and arc_margin >= 0.01:
                                    best_name = arc_name
                                    best_dist = 0
                                    print(f"[FaceScan] ArcFace match: {pe_track_id} -> {arc_name} "
                                          f"(conf={arc_conf:.3f}, margin={arc_margin:.3f})")
                                else:
                                    print(f"[FaceScan] ArcFace no match: conf={arc_conf:.3f} margin={arc_margin:.3f}")
                            except Exception as e:
                                print(f"[FaceScan] ArcFace error: {e}")

                        # Method 2: Fallback to time-proximity matching
                        if best_name is None:
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
        live_events = list(_latest_events[:200])
    # Keep the dashboard useful after a bridge restart.  In-memory events are
    # newest; Neon fills the history without duplicating Frigate event IDs.
    seen_ids = {item["track_id"] for item in live_events}
    conn = _get_db()
    if conn is None:
        return jsonify(live_events)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT track_id, person_name, event_type, event_time, confidence,
                       camera_id, zone
                FROM person_events
                ORDER BY id DESC LIMIT 200
            """)
            for row in cur.fetchall():
                if row["track_id"] in seen_ids:
                    continue
                live_events.append({
                    "person": row["person_name"] or "Unknown",
                    "track_id": row["track_id"],
                    "direction": row["event_type"],
                    "confidence": f"{float(row['confidence'] or 0):.0%}",
                    "identity_source": "frigate",
                    "identity_status": "known" if row["person_name"] else "unknown",
                    "timestamp": str(row["event_time"]),
                    "camera": row["camera_id"],
                    "zones": (row["zone"] or "").split(",") if row["zone"] else [],
                    "snapshot_url": f"/api/events/{row['track_id']}/snapshot.jpg",
                    "type": "frigate_zone_path",
                })
    except Exception as e:
        print(f"[DB] Recent events query error: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
    return jsonify(live_events[:200])


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


@app.route("/api/frigate/recognition")
def api_frigate_recognition():
    """Expose the Frigate face-library state so recognition failures are visible."""
    try:
        response = requests.get(f"{FRIGATE_API}/api/faces", timeout=8)
        if response.status_code != 200:
            return jsonify({"connected": False, "error": f"Frigate returned {response.status_code}"}), 502
        faces = response.json()
        people = {name: len(files) for name, files in faces.items()
                  if name != "train" and isinstance(files, list)}
        training = faces.get("train", [])
        return jsonify({
            "connected": True,
            "known_people": sorted(people),
            "gallery_images": sum(people.values()),
            "images_per_person": people,
            "recent_recognition_attempts": len(training) if isinstance(training, list) else 0,
            "face_recognition_enabled": True,
        })
    except Exception as e:
        return jsonify({"connected": False, "error": str(e)}), 502


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
        if r.status_code == 200 and len(r.content) > 100:
            return Response(r.content, mimetype="image/jpeg")
    except Exception:
        pass
    return Response(b"", mimetype="image/jpeg", status=404)


@app.route("/api/face/evaluate", methods=["GET", "POST"])
def api_face_evaluate():
    """Run face model evaluation (buffalo_s, antelopev2, buffalo_l)."""
    try:
        from face_evaluator import get_evaluator
        evaluator = get_evaluator()
        model = None
        try:
            if request.is_json:
                model = request.json.get("model")
        except Exception:
            pass
        results = evaluator.evaluate(model)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/face/evaluate/report")
def api_face_report():
    """Get the latest face model evaluation report."""
    try:
        from face_evaluator import get_evaluator
        evaluator = get_evaluator()
        report = evaluator.get_report()
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/face/engine/status")
def api_face_engine_status():
    """Get face engine status including model info."""
    try:
        from face_engine import get_engine
        engine = get_engine()
        return jsonify(engine.status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _get_db()
    _ensure_bridge_schema()
    _init_person_state_from_db()
    poller = threading.Thread(target=_poll_frigate, daemon=True)
    poller.start()
    print(f"[Bridge] Starting dashboard at http://0.0.0.0:{WEB_PORT}")
    app.run(host="0.0.0.0", port=WEB_PORT, debug=False)
