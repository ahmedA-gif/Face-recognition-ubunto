"""Tests for reasoning + attendance + boundary behaviour with unknown persons.

Covers the critical CCTV flow: an unrecognised person (Guest identity) must
get a check-in on entry and a check-out on exit — with no gallery enrolled.
"""

from __future__ import annotations

import tempfile
import os
from pathlib import Path

import numpy as np
import pytest

from src.attendance.db import AttendanceDB
from src.attendance.manager import AttendanceManager
from src.events.dynamic_boundary import DynamicBoundaryEngine
from src.events.entry_exit import EntryExitEngine
from src.events.store import Event, EventsStore
from src.reasoning.spatial_temporal import SpatialTemporalReasoning
from src.tracking.bytetrack import Track


class TestEntryExitHysteresis:
    """Centroid jitter near the line must NOT produce duplicate events."""

    LINE = {"x1": 0.5, "y1": 0.0, "x2": 0.5, "y2": 1.0}  # vertical, px-space x=50

    def _track(self, x: float, hits: int = 20) -> Track:
        return Track(track_id=1, xyxy=np.array([x, 10, x + 4, 40], dtype=np.float64), conf=0.8, hits=hits)

    @pytest.fixture()
    def eng(self, tmp_path: Path) -> tuple[EntryExitEngine, EventsStore]:
        store = EventsStore(str(tmp_path / "events.db"))
        engine = EntryExitEngine(
            line_norm=self.LINE,
            entry_direction="A_to_B",
            debounce_sec=0.0,
            min_track_frames=3,
            hysteresis_px=8.0,
        )
        engine.update([self._track(20, hits=3)], (100, 100, 3), store)  # warm A zone
        yield engine, store
        store.close()

    def test_single_clean_crossing(self, eng):
        engine, store = eng
        events: list[Event] = []
        events += engine.update([self._track(80)], (100, 100, 3), store)
        assert len(events) == 1
        assert events[0].direction == "entry"

    def test_jitter_inside_band_no_events(self, eng):
        engine, store = eng
        events: list[Event] = []
        for x in (52, 48, 53, 47, 52, 46, 51):  # oscillate right at the line
            events += engine.update([self._track(x)], (100, 100, 3), store)
        assert len(events) == 0

    def test_shallow_overshoot_no_event(self, eng):
        engine, store = eng
        events: list[Event] = []
        events += engine.update([self._track(55)], (100, 100, 3), store)
        assert len(events) == 0  # 55-50=5px < margin 8 → still sticky A

    def test_deep_cross_then_back_fires_twice(self, eng):
        engine, store = eng
        events: list[Event] = []
        events += engine.update([self._track(80)], (100, 100, 3), store)
        events += engine.update([self._track(20)], (100, 100, 3), store)
        assert [e.direction for e in events] == ["entry", "exit"]

    def test_line_swap_resets_zones(self, eng):
        engine, store = eng
        events: list[Event] = []
        engine.update([self._track(80)], (100, 100, 3), store)  # A→B fired
        engine.set_line({"x1": 0.8, "y1": 0.0, "x2": 0.8, "y2": 1.0})
        events += engine.update([self._track(80)], (100, 100, 3), store)
        events += engine.update([self._track(85)], (100, 100, 3), store)
        assert len(events) == 0  # zone was reset; 80-78=2px inside band


class TestHorizontalDoor:
    """Horizontal door line (camera inside, door at top): B_to_A = entry."""

    # Door segment: x in [0.2, 0.8] * 100px, at y=0.55 * 100px = 55.
    LINE = {"x1": 0.2, "y1": 0.55, "x2": 0.8, "y2": 0.55}
    ENTRY = "B_to_A"

    def _track(self, foot_x: float, foot_y: float, hits: int = 20, tid: int = 1) -> Track:
        # 6px-wide, 30px-tall box whose feet are at (foot_x, foot_y).
        return Track(
            track_id=tid,
            xyxy=np.array([foot_x - 3, foot_y - 30, foot_x + 3, foot_y], dtype=np.float64),
            conf=0.8,
            hits=hits,
        )

    @pytest.fixture()
    def eng(self, tmp_path: Path):
        store = EventsStore(str(tmp_path / "events.db"))
        engine = EntryExitEngine(
            line_norm=self.LINE,
            entry_direction=self.ENTRY,
            debounce_sec=0.0,
            min_track_frames=5,
            hysteresis_px=8.0,
            use_foot_point=True,
            segment_pad=0.12,
            require_committed_zone=True,
        )
        yield engine, store
        store.close()

    def test_above_to_below_is_entry(self, eng):
        engine, store = eng
        events: list[Event] = []
        # Warm up deep on the outside (B) side: feet at y=40.
        for _ in range(5):
            events += engine.update([self._track(50, 40, hits=5)], (100, 100, 3), store)
        # Cross through the door into the room (A side): feet at y=70.
        events += engine.update([self._track(50, 70, hits=6)], (100, 100, 3), store)
        assert len(events) == 1
        assert events[0].direction == "entry"

    def test_below_to_above_is_exit(self, eng):
        engine, store = eng
        events: list[Event] = []
        for _ in range(5):
            events += engine.update([self._track(50, 70, hits=5)], (100, 100, 3), store)  # warm inside (A)
        events += engine.update([self._track(50, 40, hits=6)], (100, 100, 3), store)  # leave → B
        assert len(events) == 1
        assert events[0].direction == "exit"

    def test_crossing_outside_door_segment_no_event(self, eng):
        engine, store = eng
        events: list[Event] = []
        # Warm up outside the door (left of segment): foot_x=10 (projection < -pad).
        for _ in range(5):
            events += engine.update([self._track(10, 40, hits=5)], (100, 100, 3), store)
        # "Cross" to the inside side but still left of the door → no event.
        events += engine.update([self._track(10, 70, hits=6)], (100, 100, 3), store)
        events += engine.update([self._track(10, 75, hits=7)], (100, 100, 3), store)
        assert len(events) == 0

    def test_foot_point_fires_when_centroid_has_not_crossed(self, tmp_path: Path):
        """A person whose torso is outside but feet already inside counts as entry
        with foot-point geometry, but not with body-centroid geometry."""
        store = EventsStore(str(tmp_path / "foot.db"))
        foot_eng = EntryExitEngine(
            line_norm=self.LINE, entry_direction=self.ENTRY, debounce_sec=0.0,
            min_track_frames=5, hysteresis_px=8.0, use_foot_point=True,
        )
        ctr_eng = EntryExitEngine(
            line_norm=self.LINE, entry_direction=self.ENTRY, debounce_sec=0.0,
            min_track_frames=5, hysteresis_px=8.0, use_foot_point=False,
        )

        def box(foot_y: float) -> Track:
            # 8px-wide box, 30px tall. Feet at y=foot_y, top at foot_y-30.
            return Track(track_id=1, xyxy=np.array([46, foot_y - 30, 54, foot_y], dtype=np.float64), conf=0.8, hits=5)

        # Warm up: feet deep outside (B), y=40 → centroid y=25.
        for _ in range(5):
            foot_eng.update([box(40)], (100, 100, 3), store)
            ctr_eng.update([box(40)], (100, 100, 3), store)

        # Straddle: feet at y=70 (inside, A) but centroid at y=55 → still on the line.
        f_events = foot_eng.update([box(70)], (100, 100, 3), store)
        c_events = ctr_eng.update([box(70)], (100, 100, 3), store)

        assert len(f_events) == 1
        assert f_events[0].direction == "entry"
        assert len(c_events) == 0
        store.close()


class TestWindowBias:
    def test_no_flip_by_default_evening_entry_stays_entry(self):
        rs = SpatialTemporalReasoning()  # window_bias defaults to False
        e = Event(id=1, date="2026-08-02", time="19:58:27", person="Guest#004", direction="entry", track_id=4)
        v = rs.verify([e], boundary_conf=0.0, boundary_learned=False)[0]
        assert v.action == "accept"
        assert v.direction == "entry"

    def test_no_flip_by_default_morning_exit_stays_exit(self):
        rs = SpatialTemporalReasoning()
        e = Event(id=2, date="2026-08-02", time="09:30:00", person="Guest#001", direction="exit", track_id=1)
        v = rs.verify([e], boundary_conf=0.0, boundary_learned=False)[0]
        assert v.action == "accept"
        assert v.direction == "exit"

    def test_flip_when_window_bias_enabled(self):
        rs = SpatialTemporalReasoning(window_bias=True)
        e = Event(id=3, date="2026-08-02", time="19:58:27", person="Guest#004", direction="entry", track_id=4)
        v = rs.verify([e], boundary_conf=0.0, boundary_learned=False)[0]
        assert v.action == "flip"
        assert v.direction == "exit"

    def test_no_flip_after_boundary_learned(self):
        rs = SpatialTemporalReasoning(window_bias=True)
        e = Event(id=4, date="2026-08-02", time="19:58:27", person="Guest#004", direction="entry", track_id=4)
        v = rs.verify([e], boundary_conf=1.0, boundary_learned=True)[0]
        assert v.action == "accept"
        assert v.direction == "entry"

    def test_uturn_still_rejected_with_defaults(self):
        rs = SpatialTemporalReasoning()
        # Same direction within uturn_sec → duplicate rejection
        rs.verify([Event(id=10, date="2026-08-02", time="12:00:00", person="Guest#001", direction="entry", track_id=7)], 1.0, True)
        v = rs.verify([Event(id=11, date="2026-08-02", time="12:00:01", person="Guest#001", direction="entry", track_id=7)], 1.0, True)[0]
        assert v.action == "reject"
        assert v.void_previous_id == 10


class TestUnknownAttendance:
    """Unenrolled persons (Guest#NNN) must still get check-in / check-out."""

    @pytest.fixture()
    def manager(self, tmp_path: Path):
        db = AttendanceDB(str(tmp_path / "attendance.db"))
        m = AttendanceManager(db, shift_start="09:00", shift_end="17:00", debounce_minutes=0)
        yield m
        db.close()

    def test_unknown_check_in_and_out(self, manager: AttendanceManager):
        manager.process_events(
            [Event(id=1, date="2026-08-02", time="19:58:27", person="Guest#004", direction="entry", track_id=4)]
        )
        manager.process_events(
            [Event(id=2, date="2026-08-02", time="20:30:00", person="Guest#004", direction="exit", track_id=9)]
        )
        rows = manager.summary("2026-08-02")
        assert len(rows) == 1
        r = rows[0]
        assert r["person_id"] == "Guest#004"
        assert r["check_in_time"] == "19:58:27"
        assert r["check_out_time"] == "20:30:00"
        assert r["work_hours"] == pytest.approx(31.55 / 60, abs=0.01)

    def test_evening_entry_produces_checkin(self, manager: AttendanceManager):
        manager.process_events(
            [Event(id=3, date="2026-08-02", time="19:58:27", person="Guest#004", direction="entry", track_id=4)]
        )
        rows = manager.summary("2026-08-02")
        assert len(rows) == 1
        assert rows[0]["check_in_time"] == "19:58:27"
        assert rows[0]["status"] == "Late"

    def test_fragmented_tracks_same_person_one_record(self, manager: AttendanceManager):
        # 3 raw track_ids (occlusion fragments) → ONE attendance record
        for tid, t in [(11, "10:00:00"), (12, "10:00:05"), (13, "10:00:09")]:
            manager.process_events(
                [Event(id=tid, date="2026-08-02", time=t, person="Guest#007", direction="entry", track_id=tid)]
            )
        rows = manager.summary("2026-08-02")
        assert len(rows) == 1
        assert rows[0]["check_in_time"] == "10:00:00"


class TestAttendanceHydration:
    """State must survive process restarts (hydrated from DB)."""

    def _entry_outcome(self, m: AttendanceManager, person: str, t: str, tid: int):
        return m.process_events([Event(id=tid, date="2026-08-02", time=t, person=person, direction="entry", track_id=tid)])

    def test_exit_after_restart_closes_open_checkin(self, tmp_path: Path):
        db_path = str(tmp_path / "attendance.db")
        db1 = AttendanceDB(db_path)
        m1 = AttendanceManager(db1, shift_start="09:00", shift_end="17:00", debounce_minutes=0)
        m1.process_events(
            [Event(id=1, date="2026-08-02", time="09:30:00", person="Haseeb", direction="entry", track_id=1)]
        )
        db1.close()

        # Simulate a process restart: brand-new manager reading the same DB.
        db2 = AttendanceDB(db_path)
        m2 = AttendanceManager(db2, shift_start="09:00", shift_end="17:00", debounce_minutes=0)
        out = m2.process_events(
            [Event(id=2, date="2026-08-02", time="17:00:00", person="Haseeb", direction="exit", track_id=2)]
        )
        assert out and out[0]["action"] == "check_out"
        rows = m2.summary("2026-08-02")
        assert rows[0]["check_in_time"] == "09:30:00"
        assert rows[0]["check_out_time"] == "17:00:00"
        db2.close()

    def test_exit_without_prior_entry_still_ignored(self, tmp_path: Path):
        db_path = str(tmp_path / "attendance.db")
        db = AttendanceDB(db_path)
        m = AttendanceManager(db, debounce_minutes=0)
        out = m.process_events(
            [Event(id=1, date="2026-08-02", time="17:00:00", person="Ali", direction="exit", track_id=1)]
        )
        assert out == []
        assert m.summary("2026-08-02") == []
        db.close()

    def test_reentry_after_checkout_accepted(self, tmp_path: Path):
        db = AttendanceDB(str(tmp_path / "attendance.db"))
        m = AttendanceManager(db, shift_start="09:00", shift_end="17:00", debounce_minutes=0)
        m.process_events([Event(id=1, date="2026-08-02", time="09:00:00", person="Haseeb", direction="entry", track_id=1)])
        out1 = m.process_events([Event(id=2, date="2026-08-02", time="12:00:00", person="Haseeb", direction="exit", track_id=2)])
        assert out1[0]["action"] == "check_out"
        assert m.summary("2026-08-02")[0]["check_out_time"] == "12:00:00"

        # Person returns the same day → checkout cleared, state back to CHECKED_IN.
        out2 = m.process_events([Event(id=3, date="2026-08-02", time="14:00:00", person="Haseeb", direction="entry", track_id=3)])
        assert out2[0]["action"] == "re_entry"
        assert m.summary("2026-08-02")[0]["check_out_time"] is None

        # And a final exit closes the day again.
        out3 = m.process_events([Event(id=4, date="2026-08-02", time="17:00:00", person="Haseeb", direction="exit", track_id=4)])
        assert out3[0]["action"] == "check_out"
        assert m.summary("2026-08-02")[0]["check_out_time"] == "17:00:00"
        db.close()


class TestBoundaryLearning:
    def test_learns_vertical_line_from_flow(self):
        bnd = DynamicBoundaryEngine(
            min_tracks_for_learning=10, adaptation_rate=0.5, smoothing=False, min_speed_norm=0.02, recompute_every=3
        )
        for i in range(14):  # dominant flow: left → right
            for x in np.linspace(0.15, 0.85, 6):
                bnd.feed(f"P{i:02d}", x, 0.5)
        for i in range(3):   # minority flow: right → left
            for x in np.linspace(0.85, 0.15, 6):
                bnd.feed(f"L{i:02d}", x, 0.5)

        assert bnd.learned
        assert bnd.confidence == pytest.approx(1.0)
        # dominant flow left→right means entry = A_to_B and line is near-vertical
        assert bnd.entry_direction == "A_to_B"
        assert abs(bnd.line_norm["x1"] - bnd.line_norm["x2"]) < 0.02
        assert 0.2 <= bnd.line_norm["x1"] <= 0.8

    def test_keeps_seed_until_enough_tracks(self):
        bnd = DynamicBoundaryEngine(
            min_tracks_for_learning=10, adaptation_rate=0.5, smoothing=False, min_speed_norm=0.02, recompute_every=3,
            seed_line={"x1": 0.45, "y1": 0.1, "x2": 0.45, "y2": 0.9},
        )
        for x in np.linspace(0.15, 0.85, 6):
            bnd.feed("A", x, 0.5)
        assert not bnd.learned
        assert bnd.line_norm == {"x1": 0.45, "y1": 0.1, "x2": 0.45, "y2": 0.9}
        assert bnd.confidence < 1.0
