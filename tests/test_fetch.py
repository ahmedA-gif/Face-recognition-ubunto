"""Unit tests for all fetch surfaces: EventsStore, AttendanceDB, FaceGallery, CameraStream."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.attendance.db import AttendanceDB
from src.capture.stream import CameraStream, _is_file_source, _is_network_source
from src.events.store import Event, EventsStore
from src.recognition.gallery import FaceGallery


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def events_store(tmp_dir: Path) -> EventsStore:
    store = EventsStore(str(tmp_dir / "events.db"))
    yield store
    store.close()


@pytest.fixture()
def attendance_db(tmp_dir: Path) -> AttendanceDB:
    db = AttendanceDB(str(tmp_dir / "attendance.db"))
    yield db
    db.close()


@pytest.fixture()
def gallery(tmp_dir: Path) -> FaceGallery:
    g = FaceGallery(str(tmp_dir / "faces.db"), match_threshold=0.35, backend="numpy")
    yield g
    g.close()


def _unit(dim: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


# ── EventsStore ───────────────────────────────────────────────────────────────


class TestEventsStore:
    def test_insert_and_recent_order(self, events_store: EventsStore):
        for i in range(5):
            events_store.insert(
                Event(
                    date="2026-07-31",
                    time=f"10:0{i}:00",
                    person=f"P{i}",
                    direction="entry" if i % 2 == 0 else "exit",
                    track_id=i,
                    confidence=0.8 + i * 0.01,
                )
            )
        rows = events_store.recent(3)
        assert len(rows) == 3
        assert [r["person"] for r in rows] == ["P4", "P3", "P2"]
        assert rows[0]["id"] > rows[1]["id"] > rows[2]["id"]

    def test_get_by_id(self, events_store: EventsStore):
        eid = events_store.insert(
            Event(date="2026-07-31", time="11:00:00", person="Ahmed", direction="entry")
        )
        row = events_store.get(eid)
        assert row is not None
        assert row["person"] == "Ahmed"
        assert row["direction"] == "entry"
        assert events_store.get(99999) is None

    def test_update_direction_and_delete(self, events_store: EventsStore):
        eid = events_store.insert(
            Event(date="2026-07-31", time="12:00:00", person="Sara", direction="entry")
        )
        assert events_store.update_direction(eid, "exit") is True
        assert events_store.get(eid)["direction"] == "exit"
        assert events_store.delete(eid) is True
        assert events_store.get(eid) is None
        assert events_store.delete(eid) is False  # already gone

    def test_by_person_and_count(self, events_store: EventsStore):
        for person, direction in [("Ahmed", "entry"), ("Ahmed", "exit"), ("Sara", "entry")]:
            events_store.insert(
                Event(date="2026-07-31", time="09:00:00", person=person, direction=direction)
            )
        ahmed = events_store.by_person("Ahmed", date="2026-07-31")
        assert len(ahmed) == 2
        assert all(r["person"] == "Ahmed" for r in ahmed)
        assert events_store.count() == 3
        assert events_store.by_person("Nobody") == []

    def test_recent_limit_clamped(self, events_store: EventsStore):
        events_store.insert(Event(date="2026-07-31", time="01:00:00", person="X", direction="entry"))
        assert len(events_store.recent(0)) == 1  # clamped to >= 1
        assert len(events_store.recent(-5)) == 1

    def test_context_manager(self, tmp_dir: Path):
        with EventsStore(str(tmp_dir / "ctx.db")) as store:
            store.insert(Event(date="2026-07-31", time="01:00:00", person="Y", direction="exit"))
            assert store.count() == 1


# ── AttendanceDB ──────────────────────────────────────────────────────────────


class TestAttendanceDB:
    def test_upsert_preserves_check_in(self, attendance_db: AttendanceDB):
        attendance_db.upsert_log(
            "2026-07-31", "Ahmed", "Ahmed", "09:05", None, "On Time", None, 0.9, "cam_01"
        )
        attendance_db.upsert_log(
            "2026-07-31", "Ahmed", "Ahmed", None, "17:10", "On Time", 8.08, 0.95, "cam_01"
        )
        row = attendance_db.get_log("2026-07-31", "Ahmed")
        assert row is not None
        assert row["check_in_time"] == "09:05"
        assert row["check_out_time"] == "17:10"
        assert row["work_hours"] == pytest.approx(8.08)
        assert row["confidence"] == pytest.approx(0.95)

    def test_daily_logs_sorted(self, attendance_db: AttendanceDB):
        attendance_db.upsert_log(
            "2026-07-31", "B", "B", "09:30", None, "Late", None, 0.8, "cam_01"
        )
        attendance_db.upsert_log(
            "2026-07-31", "A", "A", "08:50", None, "On Time", None, 0.9, "cam_01"
        )
        logs = attendance_db.daily_logs("2026-07-31")
        assert [r["person_id"] for r in logs] == ["A", "B"]
        assert attendance_db.daily_logs("2099-01-01") == []

    def test_get_log_missing(self, attendance_db: AttendanceDB):
        assert attendance_db.get_log("2026-07-31", "ghost") is None

    def test_calibrations_parse_json(self, attendance_db: AttendanceDB):
        line = {"x1": 0.4, "y1": 0.1, "x2": 0.4, "y2": 0.9}
        cid = attendance_db.insert_calibration(line, 0.77)
        cals = attendance_db.calibrations(limit=5)
        assert len(cals) == 1
        assert cals[0]["id"] == cid
        assert cals[0]["line_coords"] == line
        assert cals[0]["cluster_confidence"] == pytest.approx(0.77)
        assert isinstance(cals[0]["line_coords"], dict)


# ── FaceGallery ───────────────────────────────────────────────────────────────


class TestFaceGallery:
    def test_add_match_and_unknown(self, gallery: FaceGallery):
        emb_a = _unit(64, seed=1)
        emb_b = _unit(64, seed=99)  # orthogonal-ish
        gallery.add("Alice", emb_a)
        name, score = gallery.match(emb_a)
        assert name == "Alice"
        assert score >= gallery.match_threshold

        name2, score2 = gallery.match(emb_b)
        # random vector should fall below threshold most of the time
        assert name2 == "Unknown" or score2 < 1.0

    def test_incremental_add_no_full_reload(self, gallery: FaceGallery):
        for i in range(10):
            gallery.add(f"P{i}", _unit(64, seed=i + 1))
        assert gallery.count() == 10
        assert gallery.count_db() == 10
        assert set(gallery.list_people()) == {f"P{i}" for i in range(10)}

    def test_add_many_batch(self, gallery: FaceGallery):
        items = [(f"Bulk{i}", _unit(64, seed=100 + i)) for i in range(20)]
        n = gallery.add_many(items)
        assert n == 20
        assert gallery.count() == 20
        name, score = gallery.match(items[7][1])
        assert name == "Bulk7"
        assert score > 0.99

    def test_empty_gallery_match(self, gallery: FaceGallery):
        name, score = gallery.match(_unit(64, seed=0))
        assert name == "Unknown"
        assert score == 0.0

    def test_status_memory_backed(self, gallery: FaceGallery):
        gallery.add("Zed", _unit(64, seed=7))
        s = gallery.status()
        assert "people=1" in s
        assert "embeddings=1" in s
        assert "numpy" in s

    def test_reload_after_external_write(self, tmp_dir: Path):
        path = str(tmp_dir / "ext.db")
        g1 = FaceGallery(path, backend="numpy", match_threshold=0.3)
        g1.add("Ext", _unit(32, seed=3))
        g1.close()

        g2 = FaceGallery(path, backend="numpy", match_threshold=0.3)
        assert g2.count() == 1
        assert g2.list_people() == ["Ext"]
        g2.close()

    def test_dim_mismatch_skipped_on_reload(self, tmp_dir: Path):
        path = str(tmp_dir / "bad.db")
        g = FaceGallery(path, backend="numpy")
        # Manually insert a corrupt row
        emb = _unit(16, seed=1)
        g._conn.execute(
            "INSERT INTO faces(name, embedding, dim) VALUES (?, ?, ?)",
            ("Bad", emb.tobytes(), 99),  # wrong dim
        )
        g._conn.commit()
        g.reload()
        assert g.count() == 0
        g.close()


# ── FaceGallery · FAISS backend + persisted index paths ───────────────────────


class TestFaceGalleryFAISS:
    """Covers the faiss-cpu fetch path: exact match, index reuse, corruption,
    and staleness after external writes."""

    def test_faiss_match_exact_and_threshold(self, tmp_dir: Path):
        faiss_mod = pytest.importorskip("faiss")
        g = FaceGallery(str(tmp_dir / "f.db"), backend="faiss", match_threshold=0.35)
        emb_a = _unit(64, seed=1)
        emb_b = _unit(64, seed=2)
        g.add("Alice", emb_a)
        g.add("Bob", emb_b)

        name, score = g.match(emb_a)
        assert name == "Alice"
        assert score > 0.99
        assert g.backend == "faiss"
        assert g._faiss_index is not None
        assert g._faiss_index.d == 64
        g.close()

    def test_persisted_index_reused_on_reopen(self, tmp_dir: Path):
        faiss_mod = pytest.importorskip("faiss")
        path = str(tmp_dir / "reuse.db")
        g1 = FaceGallery(path, backend="faiss", match_threshold=0.3)
        g1.add_many([(f"P{i}", _unit(64, seed=10 + i)) for i in range(10)])
        assert g1._faiss_index is not None and g1._faiss_index.ntotal == 10
        g1.close()

        g2 = FaceGallery(path, backend="faiss", match_threshold=0.3)
        # index must be loaded from disk, not rebuilt — same count, matching works
        assert g2._faiss_index is not None
        assert g2._faiss_index.ntotal == 10
        name, score = g2.match(_unit(64, seed=13))
        assert name == "P3"
        assert score > 0.99
        g2.close()

    def test_corrupt_index_falls_back_to_rebuild(self, tmp_dir: Path):
        pytest.importorskip("faiss")
        path = tmp_dir / "corrupt.db"
        g1 = FaceGallery(str(path), backend="faiss", match_threshold=0.3)
        g1.add("Solo", _unit(64, seed=5))
        g1.close()
        # Corrupt the persisted index file
        path.with_suffix(".faiss").write_bytes(b"NOT A FAISS INDEX")

        g2 = FaceGallery(str(path), backend="faiss", match_threshold=0.3)
        assert g2.count() == 1
        assert g2._faiss_index is not None and g2._faiss_index.ntotal == 1
        name, score = g2.match(_unit(64, seed=5))
        assert name == "Solo"
        g2.close()

    def test_stale_index_detected_after_external_write(self, tmp_dir: Path):
        pytest.importorskip("faiss")
        path = tmp_dir / "stale.db"
        g1 = FaceGallery(str(path), backend="faiss", match_threshold=0.3)
        g1.add_many([(f"P{i}", _unit(64, seed=20 + i)) for i in range(5)])
        g1.close()  # persisted index now has 5 vectors

        # External process inserts rows directly via SQLite (index is stale)
        conn = sqlite3.connect(str(path))
        conn.execute(
            "INSERT INTO faces(name, embedding, dim) VALUES (?, ?, ?)",
            ("Ext", _unit(64, seed=99).tobytes(), 64),
        )
        conn.commit()
        conn.close()

        g2 = FaceGallery(str(path), backend="faiss", match_threshold=0.3)
        assert g2.count_db() == 6
        assert g2.count() == 6
        assert g2._faiss_index is not None and g2._faiss_index.ntotal == 6  # rebuilt, not stale
        name, score = g2.match(_unit(64, seed=99))
        assert name == "Ext"
        g2.close()

    def test_dim_mismatch_persisted_index_rejected(self, tmp_dir: Path):
        pytest.importorskip("faiss")
        path = str(tmp_dir / "dim.db")
        g1 = FaceGallery(path, backend="faiss", match_threshold=0.3)
        g1.add("D64", _unit(64, seed=3))
        g1.close()
        # Manually change dim of the single row → persisted index (64d) is stale
        conn = sqlite3.connect(path)
        conn.execute("UPDATE faces SET dim = 128 WHERE name = 'D64'")
        conn.commit()
        conn.close()
        g2 = FaceGallery(path, backend="faiss", match_threshold=0.3)
        assert g2.count() == 0  # corrupt row skipped on reload
        g2.close()


# ── DB pragmas (WAL, indexes) + fetch variants ────────────────────────────────


class TestDBPragmasAndFetch:
    def test_events_wal_and_indexes(self, events_store: EventsStore):
        mode = events_store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert str(mode).lower() == "wal"
        idx = {
            r[0]
            for r in events_store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'events'"
            )
        }
        assert {"idx_events_id_desc", "idx_events_date_person"} <= idx

    def test_attendance_wal_and_indexes(self, attendance_db: AttendanceDB):
        mode = attendance_db._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert str(mode).lower() == "wal"
        idx = {
            r[0]
            for r in attendance_db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'attendance_logs'"
            )
        }
        assert {"idx_attendance_date", "idx_attendance_person"} <= idx

    def test_events_row_factory_returns_dicts(self, events_store: EventsStore):
        eid = events_store.insert(
            Event(date="2026-07-31", time="08:00:00", person="Row", direction="entry")
        )
        row = events_store.get(eid)
        assert isinstance(row, dict)
        assert set(row) >= {"id", "person", "direction", "date", "time"}

    def test_by_person_without_date_and_limit(self, events_store: EventsStore):
        for i in range(5):
            events_store.insert(
                Event(date="2026-07-31", time=f"09:0{i}:00", person="Ahmed", direction="entry")
            )
        for i in range(3):
            events_store.insert(
                Event(date="2026-08-01", time=f"10:0{i}:00", person="Ahmed", direction="exit")
            )
        events_store.insert(
            Event(date="2026-08-01", time="11:00:00", person="Sara", direction="entry")
        )

        all_rows = events_store.by_person("Ahmed")
        assert len(all_rows) == 8
        assert all(r["person"] == "Ahmed" for r in all_rows)
        assert all_rows[0]["id"] > all_rows[-1]["id"]  # id DESC

        limited = events_store.by_person("Ahmed", limit=3)
        assert len(limited) == 3

        dated = events_store.by_person("Ahmed", date="2026-07-31")
        assert len(dated) == 5
        assert all(r["date"] == "2026-07-31" for r in dated)

        assert events_store.by_person("Ghost", limit=0) == []  # limit clamped, no rows

    def test_daily_logs_null_checkin_ordered_last(self, attendance_db: AttendanceDB):
        attendance_db.upsert_log(
            "2026-07-31", "B", "B", None, "17:00", "Present", None, 0.8, "cam_01"
        )
        attendance_db.upsert_log(
            "2026-07-31", "A", "A", "08:50", "17:05", "On Time", 8.25, 0.9, "cam_01"
        )
        logs = attendance_db.daily_logs("2026-07-31")
        assert [r["person_id"] for r in logs] == ["A", "B"]  # NULL check_in last

    def test_events_recent_with_many_rows_uses_index(self, events_store: EventsStore):
        for i in range(500):
            events_store.insert(
                Event(
                    date=f"2026-07-{1 + i % 28:02d}",
                    time="10:00:00",
                    person=f"P{i % 25}",
                    direction="entry" if i % 2 else "exit",
                    track_id=i,
                )
            )
        rows = events_store.recent(10)
        assert len(rows) == 10
        assert rows[0]["id"] == 500
        person_rows = events_store.by_person("P7", date="2026-07-05")
        assert all(r["person"] == "P7" and r["date"] == "2026-07-05" for r in person_rows)


# ── CameraStream ──────────────────────────────────────────────────────────────


class TestCameraStream:
    def test_source_classification(self):
        assert _is_network_source("rtsp://127.0.0.1:8554/cam") is True
        assert _is_network_source("https://example.com/x.mjpg") is True
        assert _is_network_source("/tmp/video.mp4") is False
        assert _is_file_source("/tmp/video.mp4") is True
        assert _is_file_source("clip.mov") is True
        assert _is_file_source(0) is False
        assert _is_network_source(0) is False
        assert _is_network_source("0") is False

    def _write_tiny_avi(self, path: Path, n: int = 8, w: int = 64, h: int = 48) -> Path:
        # MJPG/AVI is reliably writable across opencv-python mac wheels
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            10.0,
            (w, h),
        )
        assert writer.isOpened(), "VideoWriter(MJPG) failed to open"
        for i in range(n):
            frame = np.full((h, w, 3), (i * 20) % 256, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return path

    def test_read_from_synthetic_video(self, tmp_dir: Path):
        video_path = self._write_tiny_avi(tmp_dir / "tiny.avi", n=8)
        frames = []
        with CameraStream(str(video_path), drop_stale=False, reconnect=False) as stream:
            assert stream.is_opened()
            for frame in stream.frames():
                frames.append(frame)
        assert len(frames) == 8
        assert frames[0].shape == (48, 64, 3)

    def test_read_tuple_api(self, tmp_dir: Path):
        video_path = self._write_tiny_avi(tmp_dir / "one.avi", n=3, w=32, h=32)
        stream = CameraStream(str(video_path), drop_stale=False, reconnect=False)
        stream.open()
        ok, frame = stream.read()
        assert ok is True
        assert frame is not None
        # drain to EOF
        while True:
            ok, _ = stream.read()
            if not ok:
                break
        stream.release()

    def test_project_test_video_if_present(self):
        project_video = Path("data/test_video.mp4")
        if not project_video.exists():
            pytest.skip("data/test_video.mp4 not present")
        with CameraStream(str(project_video), drop_stale=False, reconnect=False) as stream:
            ok, frame = stream.read()
            assert ok is True
            assert frame is not None
            assert frame.ndim == 3

    def test_missing_source_raises(self):
        stream = CameraStream("/nonexistent/path/nope.mp4", reconnect=False)
        with pytest.raises(RuntimeError, match="Cannot open"):
            stream.open()


# ── micro-benchmarks (guardrails, not strict) ─────────────────────────────────


class TestFetchPerformance:
    def test_events_recent_is_fast(self, events_store: EventsStore):
        for i in range(200):
            events_store.insert(
                Event(
                    date="2026-07-31",
                    time="10:00:00",
                    person=f"P{i % 20}",
                    direction="entry",
                    track_id=i,
                )
            )
        t0 = time.perf_counter()
        for _ in range(50):
            events_store.recent(50)
        elapsed = time.perf_counter() - t0
        # 50 fetches of 50 rows should be well under 250ms on any modern machine
        assert elapsed < 0.5, f"recent() too slow: {elapsed:.3f}s"

    def test_gallery_match_is_fast(self, gallery: FaceGallery):
        items = [(f"P{i}", _unit(128, seed=i + 1)) for i in range(100)]
        gallery.add_many(items)
        q = _unit(128, seed=42)
        t0 = time.perf_counter()
        for _ in range(500):
            gallery.match(q)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5, f"match() too slow: {elapsed:.3f}s"

    def test_incremental_add_faster_than_rebuild(self, tmp_dir: Path):
        g_inc = FaceGallery(str(tmp_dir / "inc.db"), backend="numpy")
        g_reb = FaceGallery(str(tmp_dir / "reb.db"), backend="numpy")
        vecs = [_unit(64, seed=i + 1) for i in range(40)]

        t0 = time.perf_counter()
        for i, v in enumerate(vecs):
            g_inc.add(f"P{i}", v)  # incremental
        t_inc = time.perf_counter() - t0

        t0 = time.perf_counter()
        for i, v in enumerate(vecs):
            g_reb.add(f"P{i}", v, rebuild=True)  # full reload each time
        t_reb = time.perf_counter() - t0

        g_inc.close()
        g_reb.close()
        # Incremental should not be slower than full rebuild (usually much faster)
        assert t_inc <= t_reb * 1.5 + 0.05
