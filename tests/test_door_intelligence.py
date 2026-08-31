"""Door Intelligence Engine — polygon FSM unit tests.

Synthetic bbox sequences drive the engine through OUTSIDE / DOOR / INSIDE and
assert the event rules from the production plan:

- clean walk-through  → exactly one correctly-directed ENTRY
- peek / step back    → no event
- stand-at-door       → no event
- ten simultaneous    → ten distinct events, no cross-attribution
- lock after ENTRY    → no duplicate ENTRY while standing inside
- clean exit          → exactly one EXIT
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.events.door_intelligence import DoorIntelligenceEngine, TrackState
from src.events.store import Event, EventsStore
from src.tracking.bytetrack import Track

# Normalized zones scaled to a 100x100 pixel frame.
ZONES = {
    "OUTSIDE": [[0.15, 0.05], [0.85, 0.05], [0.85, 0.48], [0.15, 0.48]],
    "DOOR": [[0.18, 0.48], [0.82, 0.48], [0.82, 0.62], [0.18, 0.62]],
    "INSIDE": [[0.05, 0.62], [0.95, 0.62], [0.95, 0.98], [0.05, 0.98]],
}

# Foot y positions that land in each region of the 100x100 frame.
OUTSIDE_Y = 20
DOOR_Y = 56
INSIDE_Y = 80


def _track(tid: int, y_bottom: float, hits: int = 3, gid: str = "") -> Track:
    """Person 10px wide, 40px tall; feet (bottom-center) at y=y_bottom."""
    t = Track(
        track_id=tid,
        xyxy=np.array([45.0, y_bottom - 40, 55.0, y_bottom], dtype=np.float64),
        conf=0.9,
        hits=hits,
    )
    if gid:
        t.meta["global_id"] = gid
        t.person_name = gid
    return t


def _engine(tmp_path: Path, **kw):
    store = EventsStore(str(tmp_path / "events.db"))
    params = dict(
        zones=ZONES,
        camera_id="test_cam",
        min_track_frames=3,
        min_dwell_door_sec=0.0,
        min_inside_frames=3,
        min_outside_frames=3,
        min_motion_speed=0.5,
        motion_toward_inside_dot=0.25,
        motion_outward_dot=0.25,
        peek_max_inside_sec=10.0,
        uturn_max_door_sec=10.0,
    )
    params.update(kw)
    return DoorIntelligenceEngine(**params), store


def _walk_y(start: float, step: float, n: int) -> list[float]:
    return [round(start + step * i, 2) for i in range(n)]


class TestWalkThrough:
    def test_clean_entry_fires_exactly_once(self, tmp_path: Path):
        eng, store = _engine(tmp_path)
        events: list[Event] = []
        for y in _walk_y(20, 6, 11):  # 20 → 80 through outside→door→inside
            events += eng.update([_track(1, y, gid="Guest#001")], (100, 100, 3), store)
        assert len(events) == 1
        assert events[0].direction == "entry"
        assert events[0].person == "Guest#001"
        # Keep standing inside → no second ENTRY (lock holds).
        for _ in range(10):
            events += eng.update([_track(1, 80, gid="Guest#001")], (100, 100, 3), store)
        assert len(events) == 1
        store.close()

    def test_clean_exit_fires_exactly_once(self, tmp_path: Path):
        eng, store = _engine(tmp_path)
        events: list[Event] = []
        # Warm up deep inside.
        for _ in range(4):
            events += eng.update([_track(1, INSIDE_Y, gid="Guest#001")], (100, 100, 3), store)
        # Walk out through the door to outside (y 80 → 20).
        for y in _walk_y(80, -6, 11):
            events += eng.update([_track(1, y, gid="Guest#001")], (100, 100, 3), store)
        assert len(events) == 1
        assert events[0].direction == "exit"
        assert events[0].person == "Guest#001"
        store.close()

    def test_peek_and_step_back_no_event(self, tmp_path: Path):
        eng, store = _engine(tmp_path)
        events: list[Event] = []
        # Approach from outside into the corridor, then back out.
        for y in (20, 26, 32, 38, 44, 50, 56, 50, 44, 38, 32, 26, 20):
            events += eng.update([_track(1, y)], (100, 100, 3), store)
        # Hover in the corridor for a while.
        for _ in range(15):
            events += eng.update([_track(1, DOOR_Y)], (100, 100, 3), store)
        assert len(events) == 0
        store.close()

    def test_stand_at_door_no_event(self, tmp_path: Path):
        eng, store = _engine(tmp_path)
        events: list[Event] = []
        for _ in range(30):
            events += eng.update([_track(1, DOOR_Y)], (100, 100, 3), store)
        assert len(events) == 0
        store.close()

    def test_enter_peek_out_and_come_back_no_double(self, tmp_path: Path):
        eng, store = _engine(tmp_path)
        events: list[Event] = []
        for y in _walk_y(20, 6, 11):  # entry
            events += eng.update([_track(1, y)], (100, 100, 3), store)
        assert len(events) == 1
        # Walk to the corridor and back inside — still no second event.
        for y in (74, 68, 62, 56, 62, 68, 74, 80):
            events += eng.update([_track(1, y)], (100, 100, 3), store)
        assert len(events) == 1
        store.close()

    def test_entry_then_exit_both_fire(self, tmp_path: Path):
        eng, store = _engine(tmp_path)
        events: list[Event] = []
        for y in _walk_y(20, 6, 11):  # ENTRY
            events += eng.update([_track(1, y)], (100, 100, 3), store)
        for _ in range(4):
            events += eng.update([_track(1, 80)], (100, 100, 3), store)
        for y in _walk_y(80, -6, 11):  # EXIT
            events += eng.update([_track(1, y)], (100, 100, 3), store)
        assert [e.direction for e in events] == ["entry", "exit"]
        store.close()


class TestMultiPerson:
    def test_ten_simultaneous_entries_no_cross_attribution(self, tmp_path: Path):
        eng, store = _engine(tmp_path)
        events: list[Event] = []
        gids = [f"Guest#{i:03d}" for i in range(1, 11)]
        for y in _walk_y(20, 6, 11):
            tracks = [_track(tid, y, gid=gids[tid - 1]) for tid in range(1, 11)]
            events += eng.update(tracks, (100, 100, 3), store)
        entries = [e for e in events if e.direction == "entry"]
        assert len(entries) == 10
        assert {e.person for e in entries} == set(gids)
        assert len({e.person for e in entries}) == 10  # no duplicates
        store.close()

    def test_mixed_directions_independent(self, tmp_path: Path):
        eng, store = _engine(tmp_path)
        events: list[Event] = []
        # Person A enters while Person B exits, in the same frames.
        for i, y in enumerate(_walk_y(20, 6, 11)):
            a = _track(1, y, gid="Guest#001")          # outside → inside
            b_y = 80 - i * 6
            b = _track(2, max(20.0, b_y), gid="Guest#002")  # inside → outside
            events += eng.update([a, b], (100, 100, 3), store)
        dirs = sorted(e.direction for e in events)
        assert dirs == ["entry", "exit"]
        by_person = {e.person: e.direction for e in events}
        assert by_person["Guest#001"] == "entry"
        assert by_person["Guest#002"] == "exit"
        store.close()


class TestStateMetadata:
    def test_zone_and_state_on_track(self, tmp_path: Path):
        eng, store = _engine(tmp_path)
        for y in _walk_y(20, 6, 11):
            t = _track(1, y)
            eng.update([t], (100, 100, 3), store)
        assert t.meta["zone"] == "INSIDE"
        assert t.meta["fsm_state"] in {s.name for s in TrackState}
        store.close()

    def test_fsm_path_recorded(self, tmp_path: Path):
        eng, store = _engine(tmp_path)
        events: list[Event] = []
        for y in _walk_y(20, 6, 11):
            events += eng.update([_track(1, y)], (100, 100, 3), store)
        assert len(events) == 1
        path = events[0].fsm_path
        assert path[0] == "OUTSIDE"
        assert "DOOR" in path
        assert path[-1] == "INSIDE"
        store.close()

    def test_event_has_uuid(self, tmp_path: Path):
        eng, store = _engine(tmp_path)
        events: list[Event] = []
        for y in _walk_y(20, 6, 11):
            events += eng.update([_track(1, y)], (100, 100, 3), store)
        assert len(events) == 1
        assert events[0].event_id  # non-empty UUID
        store.close()


class TestPurgeExit:
    """When a track is purged while LOCKED_AFTER_ENTRY, EXIT fires automatically."""

    def test_purge_fires_exit_for_locked_track(self, tmp_path: Path):
        eng, store = _engine(tmp_path, purge_after_sec=0.05)
        events: list[Event] = []
        # Walk in.
        for y in _walk_y(20, 6, 11):
            events += eng.update([_track(1, y, gid="Guest#001")], (100, 100, 3), store)
        assert len(events) == 1
        assert events[0].direction == "entry"
        # Wait for purge timeout.
        import time
        time.sleep(0.1)
        # Next update with no tracks triggers purge of stale track.
        events += eng.update([], (100, 100, 3), store)
        # EXIT should have been fired on purge.
        exit_events = [e for e in events if e.direction == "exit"]
        assert len(exit_events) == 1
        assert exit_events[0].person == "Guest#001"
        store.close()

    def test_reentry_after_purge_exit(self, tmp_path: Path):
        eng, store = _engine(tmp_path, purge_after_sec=0.05)
        events: list[Event] = []
        # Walk in.
        for y in _walk_y(20, 6, 11):
            events += eng.update([_track(1, y, gid="Guest#001")], (100, 100, 3), store)
        assert len(events) == 1
        # Wait for purge.
        import time
        time.sleep(0.1)
        events += eng.update([], (100, 100, 3), store)
        assert len(events) == 2  # entry + purge exit
        # Walk in again — should produce a second ENTRY.
        for y in _walk_y(20, 6, 11):
            events += eng.update([_track(1, y, gid="Guest#001")], (100, 100, 3), store)
        entries = [e for e in events if e.direction == "entry"]
        assert len(entries) == 2
        store.close()

    def test_full_entry_exit_entry_cycle(self, tmp_path: Path):
        eng, store = _engine(tmp_path)
        events: list[Event] = []
        # Entry.
        for y in _walk_y(20, 6, 11):
            events += eng.update([_track(1, y, gid="Guest#001")], (100, 100, 3), store)
        assert len(events) == 1
        # Walk back out.
        for y in _walk_y(80, -6, 11):
            events += eng.update([_track(1, y, gid="Guest#001")], (100, 100, 3), store)
        assert len(events) == 2
        assert events[1].direction == "exit"
        # Re-enter — should fire second ENTRY.
        for y in _walk_y(20, 6, 11):
            events += eng.update([_track(1, y, gid="Guest#001")], (100, 100, 3), store)
        entries = [e for e in events if e.direction == "entry"]
        assert len(entries) == 2
        store.close()
