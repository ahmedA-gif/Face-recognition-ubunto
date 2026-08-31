"""Regression tests for the Frigate-only attendance decision path."""
import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("frigate_bridge", Path(__file__).parents[1] / "app.py")
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)
REAL_WRITE_EVENT = bridge._write_event


def event(event_id, zones, start=1000.0, end=None, sub_label=None, camera="cam_entry", y=300):
    return {
        "id": event_id, "camera": camera, "label": "person", "start_time": start,
        "end_time": end, "zones": zones, "sub_label": sub_label,
        "data": {"box": [10, y - 60, 50, y], "top_score": 0.9},
    }


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    bridge._tracks.clear()
    bridge._person_cooldowns.clear()
    bridge._person_state.clear()
    bridge._latest_events.clear()
    bridge._unknown_counter = 0
    monkeypatch.setattr(bridge, "_write_event", lambda *args: True)


def cross(event_id="one", name="Ahmed", camera="cam_entry"):
    zones = ("outside_door", "door_threshold", "inside_room") if camera == "cam_entry" else ("inside_exit", "door_exit", "outside_exit")
    for i, zone in enumerate(zones):
        bridge._ingest_event(event(event_id, [zone], start=1000, end=1002 if i == 2 else None, sub_label=[name, .91], camera=camera, y=100 + i * 100))


def test_zone_entry():
    cross()
    assert bridge._tracks["one"].direction == "ENTRY"
    assert bridge._latest_events[0]["person"] == "Ahmed"


def test_no_entry_on_presence():
    bridge._ingest_event(event("one", ["outside_door"], start=1000, end=1003, sub_label=["Ahmed", .91]))
    assert not bridge._tracks["one"].emitted


def test_peek_reverse():
    for i, zone in enumerate(("outside_door", "door_threshold", "outside_door")):
        bridge._ingest_event(event("one", [zone], start=1000, end=1003 if i == 2 else None, sub_label=["Ahmed", .91], y=100 + i * 20))
    assert not bridge._tracks["one"].emitted


def test_two_people_same_camera_do_not_share_a_track():
    bridge._ingest_event(event("a", ["outside_door"], sub_label=["Ahmed", .91]))
    bridge._ingest_event(event("b", ["outside_door"], sub_label=["Haseeb", .91]))
    assert set(bridge._tracks) == {"a", "b"}
    assert bridge._tracks["a"].entered_zones == ["outside_door"]
    assert bridge._tracks["b"].entered_zones == ["outside_door"]


def test_no_time_proximity_swap_and_no_snapshot_matcher():
    # Names only ever come from each event's own Frigate sub_label.
    cross("a", "Ahmed")
    cross("b", "Haseeb")
    assert bridge._tracks["a"].name == "Ahmed"
    assert bridge._tracks["b"].name == "Haseeb"
    assert not hasattr(bridge, "_match_face_from_snapshot")
    assert not hasattr(bridge, "_scan_face_train_files")


def test_sublabel_late_update_only_updates_its_track(monkeypatch):
    calls = []
    class Cursor:
        rowcount = 1
        def execute(self, sql, params):
            calls.append((sql, params))
        def fetchone(self):
            return ("Unknown#01", "ENTRY")
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class Connection:
        def cursor(self): return Cursor()
        def commit(self): pass
        def rollback(self): pass
    monkeypatch.setattr(bridge, "_get_db", lambda: Connection())
    assert bridge._update_event_sublabel("event-a", ["Ahmed", .91])
    assert all(params[1] == "event-a" for _, params in calls if len(params) > 1)
    assert any("attendance" in sql for sql, _ in calls)


def test_low_score_sublabel_is_unknown():
    assert bridge._name_from_sublabel(["Ahmed", .74]) is None
    assert bridge._name_from_sublabel("Ahmed") == "Ahmed"


def test_unordered_frigate_zone_payload_is_hierarchy_stable():
    track = bridge.TrackState("one", "cam_entry")
    track.update(event("one", ["door_threshold", "outside_door"], sub_label=["Ahmed", .91]))
    assert track.entered_zones == ["outside_door", "door_threshold"]


def test_unknown_spatial_cooldown_key_survives_track_reacquisition():
    first = bridge.TrackState("first-id", "cam_entry")
    second = bridge.TrackState("second-id", "cam_entry")
    first.last_box = [64, 96, 128, 192]
    second.last_box = [66, 98, 130, 194]
    assert bridge._unknown_cooldown_key(first, "ENTRY") == bridge._unknown_cooldown_key(second, "ENTRY")


def test_cooldown_eviction_removes_only_stale_entries():
    bridge._person_cooldowns.update({("old", "ENTRY"): 10, ("new", "ENTRY"): 100})
    bridge._cleanup_person_cooldowns(now=130)
    assert ("old", "ENTRY") not in bridge._person_cooldowns
    assert ("new", "ENTRY") in bridge._person_cooldowns


def test_zone_crossing_requires_duration_and_distance():
    track = bridge.TrackState("one", "cam_entry")
    track.first_seen, track.end_time = 1000.0, 1001.0
    track.entered_zones = ["outside_door", "door_threshold", "inside_room"]
    track.points = [(1.0, 0.10, 1000.0), (1.0, 0.15, 1001.0)]
    assert track.ready_direction() is None  # duration is below 1.5 seconds
    track.end_time = 1002.0
    assert track.ready_direction() == "ENTRY"
    track.points[-1] = (1.0, 0.11, 1002.0)
    assert track.ready_direction() is None  # displacement is below 0.03


def test_late_status_threshold():
    assert bridge._calc_status("09:15:00") == "On Time"
    assert bridge._calc_status("09:16:00") == "Late"


def test_attendance_exit_matches_name_not_track_id(monkeypatch):
    queries = []
    class Cursor:
        def execute(self, sql, params=None):
            queries.append((sql, params))
        def fetchone(self):
            # No prior person_event, then an open attendance row for Ahmed.
            return None if len(queries) == 1 else (4, "1970-01-01", "00:16:00")
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class Connection:
        def cursor(self): return Cursor()
        def commit(self): pass
        def rollback(self): pass
    track = bridge.TrackState("different-frigate-event", "cam_exit")
    track.end_time, track.last_box, track.last_score = 1002, [1, 1, 10, 100], .9
    track.entered_zones = ["inside_exit", "door_exit", "outside_exit"]
    monkeypatch.setattr(bridge, "_get_db", lambda: Connection())
    monkeypatch.setattr(bridge, "_write_event", REAL_WRITE_EVENT)
    assert bridge._write_event(track, "Ahmed", "EXIT")
    checkout_lookup = [params for sql, params in queries if "attendance WHERE person_name" in sql]
    assert checkout_lookup == [("Ahmed",)]
