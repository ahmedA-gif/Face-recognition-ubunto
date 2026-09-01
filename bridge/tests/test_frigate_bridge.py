"""Regression tests for the Frigate-first identity and doorway bridge."""
import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("frigate_bridge", Path(__file__).parents[1] / "app.py")
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


def event(event_id, name="Ahmed", score=0.91, camera="cam_entry", zones=None):
    if zones is None:
        zones = ["outside_door", "door_threshold", "inside_room"]
    return {
        "id": event_id, "camera": camera, "label": "person", "start_time": 1000,
        "end_time": 1002, "zones": zones, "sub_label": [name, score] if name else None,
        "data": {"box": [10, 20, 50, 200],
                 "path_data": [[[0.2, 0.2], 1000], [[0.2, 0.4], 1001], [[0.2, 0.6], 1002]]},
    }


class Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    bridge._active_tracks.clear()
    bridge._seen_event_ids.clear()
    bridge._event_cooldowns.clear()
    bridge._person_state.clear()
    bridge._latest_events.clear()
    monkeypatch.setattr(bridge, "_get_db", lambda: None)


def test_identity_is_only_the_frigate_sublabel():
    assert bridge._identity_from_frigate(["Rehmat", 0.93]) == ("Rehmat", 0.93)
    assert bridge._identity_from_frigate("Haseeb") == ("Haseeb", 0.0)
    assert bridge._identity_from_frigate(["Unknown", 0.99]) == (None, 0.0)


def test_entry_direction_requires_frigate_door_zones():
    direction, zones, confidence = bridge._direction_from_frigate(event("one"), "cam_entry", [(0, 0, 1000)] * 3)
    assert (direction, zones, confidence) == ("ENTRY", ["outside_door", "door_threshold", "inside_room"], 1.0)


def test_unknown_is_stable_not_a_fake_number(monkeypatch):
    payload = event("unknown", name=None)
    monkeypatch.setattr(bridge.requests, "get", lambda *args, **kwargs: Response(payload))
    bridge._process_person_event(payload)
    assert bridge._latest_events[0]["person"] == "Unknown"
    assert bridge._latest_events[0]["identity_status"] == "unknown"


def test_known_name_score_and_zone_label_reach_dashboard(monkeypatch):
    payload = event("rehmat", name="Rehmat", score=0.94)
    monkeypatch.setattr(bridge.requests, "get", lambda *args, **kwargs: Response(payload))
    bridge._process_person_event(payload)
    record = bridge._latest_events[0]
    assert (record["person"], record["face_score"], record["identity_source"]) == ("Rehmat", 0.94, "frigate")
    assert record["direction"] == "ENTRY"
    assert record["zones"] == ["outside_door", "door_threshold", "inside_room"]


def test_two_people_same_camera_are_not_suppressed(monkeypatch):
    first, second = event("rehmat", "Rehmat"), event("haseeb", "Haseeb")
    monkeypatch.setattr(bridge.requests, "get", lambda url, **kwargs: Response(first if url.endswith("rehmat") else second))
    bridge._process_person_event(first)
    bridge._process_person_event(second)
    assert {row["person"] for row in bridge._latest_events} == {"Rehmat", "Haseeb"}
