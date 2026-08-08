from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from src.attendance.db import AttendanceDB
from src.attendance.manager import AttendanceManager
from src.capture.stream import CameraStream
from src.detection.person_yolo import PersonDetector
from src.events.door_intelligence import DoorIntelligenceEngine
from src.events.dynamic_boundary import DynamicBoundaryEngine
from src.events.entry_exit import EntryExitEngine
from src.events.redis_publisher import EventPublisher
from src.events.store import Event, EventsStore
from src.overlay.draw import OverlayRenderer
from src.reasoning.spatial_temporal import SpatialTemporalReasoning
from src.recognition.face_engine import FaceEngine
from src.recognition.gallery import FaceGallery
from src.tracking.bytetrack import ByteTracker, Track
from src.tracking.identity_fusion import IdentityFusionEngine
from src.utils.assign import attach_faces_to_tracks
from src.utils.config import load_zones


@dataclass
class PipelineResult:
    frames_processed: int
    events_count: int
    source: str
    output_path: str = ""


def normalize_source(source: Any) -> str | int:
    """Treat numeric strings as webcam indices, leave file paths and RTSP URLs untouched."""
    if isinstance(source, int):
        return source
    if isinstance(source, Path):
        return str(source)
    if source is None:
        return 0
    if isinstance(source, str):
        stripped = source.strip()
        if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
            return int(stripped)
        return stripped
    return source


def _make_writer(output_path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {output_path}")
    return writer


def _build_components(cfg: dict[str, Any]):
    m = cfg["models"]
    pipe = cfg["pipeline"]
    ee = cfg["entry_exit"]
    ev = cfg["events"]
    ov = cfg["overlay"]

    detector = PersonDetector(
        weights=m.get("yolo_onnx", m["yolo_weights"]),
        conf=m["yolo_conf"],
        iou=m["yolo_iou"],
        imgsz=m["yolo_imgsz"],
        device=pipe["device"],
        person_class_id=m["person_class_id"],
    )
    tracker = ByteTracker()
    face_engine = FaceEngine(root=m["face_root"], pack=m["face_pack"], det_size=tuple(m["face_det_size"]))
    gallery = FaceGallery(
        db_path=ev["faces_db_path"],
        match_threshold=m["face_match_threshold"],
        backend="faiss",
    )
    print(f"[Pipeline] {gallery.status()}")
    store = EventsStore(db_path=ev["db_path"])
    entry_exit = EntryExitEngine(
        line_norm=ee["line"],
        entry_direction=ee["entry_direction"],
        debounce_sec=ee["debounce_sec"],
        min_track_frames=ee["min_track_frames"],
        hysteresis_px=ee.get("hysteresis_px", 14.0),
        use_foot_point=bool(ee.get("use_foot_point", True)),
        segment_pad=float(ee.get("segment_pad", 0.12)),
        require_committed_zone=bool(ee.get("require_committed_zone", True)),
    )
    overlay = OverlayRenderer(
        pulse_frames=ov["pulse_frames"],
        hud=ov["hud"],
        show_boundary=bool(ov.get("show_boundary", False)),
    )

    # ── Door Intelligence Engine (polygon FSM) — primary when calibrated ──
    di = cfg.get("door_intelligence", {})
    using_door_engine = bool(di.get("enabled", False))
    if using_door_engine:
        zones = dict(di.get("zones") or {}) or load_zones(
            di.get("zones_path", "config/zones.yaml"), di.get("camera_id")
        )
        if not zones:
            print(
                "[DoorIntelligence] enabled but no calibrated polygons found "
                "(config/zones.yaml) — falling back to the line engine."
            )
            using_door_engine = False
        else:
            entry_exit = DoorIntelligenceEngine(
                zones=zones,
                camera_id=di.get("camera_id", "cam_01"),
                probe=di.get("probe", "foot"),
                min_track_frames=di.get("min_track_frames", 5),
                min_dwell_door_sec=di.get("min_dwell_door_sec", 0.15),
                min_inside_frames=di.get("min_inside_frames", 3),
                min_outside_frames=di.get("min_outside_frames", 3),
                lock_after_event=di.get("lock_after_event", True),
                motion_toward_inside_dot=di.get("motion_toward_inside_dot", 0.25),
                motion_outward_dot=di.get("motion_outward_dot", 0.25),
                peek_max_inside_sec=di.get("peek_max_inside_sec", 1.5),
                uturn_max_door_sec=di.get("uturn_max_door_sec", 3.0),
                min_event_confidence=di.get("min_event_confidence", 0.70),
                purge_after_sec=di.get("purge_after_sec", 10.0),
            )
    print(
        f"[DoorIntelligence] {'ACTIVE (polygon FSM)' if using_door_engine else 'disabled (line engine)'}"
    )

    # ── new intelligence modules (config-gated) ────────────────────────────
    if_in = cfg.get("identity_fusion", {})
    if if_in.get("enabled", True):
        fusion = IdentityFusionEngine(
            max_stitch_dist_px=if_in.get("max_stitch_dist_px", 60),
            max_stitch_time_sec=if_in.get("max_stitch_time_sec", 2.0),
            embedding_match_threshold=if_in.get("embedding_match_threshold", m["face_match_threshold"]),
            max_pool_embeddings=if_in.get("max_pool_embeddings", 8),
        )
    else:
        fusion = None
    print(f"[IdentityFusion] {'active' if fusion else 'disabled'}")

    ab = cfg.get("auto_boundary", {})
    if ab.get("enabled", True):
        boundary = DynamicBoundaryEngine(
            min_tracks_for_learning=ab.get("min_tracks_for_learning", 150),
            adaptation_rate=ab.get("adaptation_rate", 0.05),
            smoothing=ab.get("smoothing", True),
            seed_line=ee["line"],
        )
    else:
        boundary = None
    print(f"[Boundary] {'auto-learning (seed line active until ' + str(ab.get('min_tracks_for_learning', 150)) + ' tracks)' if boundary else 'disabled (manual line only)'}")

    rs = cfg.get("reasoning", {})
    reasoner = SpatialTemporalReasoning(
        morning_window=tuple(rs.get("morning_window", ["07:00", "11:00"])),
        evening_window=tuple(rs.get("evening_window", ["16:00", "20:00"])),
        uturn_sec=rs.get("uturn_sec", 3.0),
        tailgate_sec=rs.get("tailgate_sec", 2.0),
        window_bias=rs.get("window_bias", False),
        enabled=rs.get("enabled", True),
    )

    at = cfg.get("attendance", {})
    att_db: Optional[AttendanceDB] = None
    attendance: Optional[AttendanceManager] = None
    if at.get("enabled", True):
        att_db = AttendanceDB(db_path=at.get("db_path", "data/db/attendance.db"))
        attendance = AttendanceManager(
            db=att_db,
            shift_start=at.get("shift_start", "09:00"),
            shift_end=at.get("shift_end", "17:00"),
            late_threshold_mins=at.get("late_threshold_mins", 15),
            early_exit_mins=at.get("early_exit_mins", 15),
            debounce_minutes=at.get("debounce_minutes", 2.0),
        )
    print(f"[Attendance] {'active → ' + at.get('db_path', 'data/db/attendance.db') if attendance else 'disabled'}")

    # ── Redis Streams live publisher (graceful JSONL fallback) ────────────
    rd = cfg.get("redis", {})
    publisher = EventPublisher(
        url=rd.get("url", "redis://localhost:6379/0"),
        stream=rd.get("stream", "attendance:events"),
        maxlen=rd.get("maxlen", 10000),
        enabled=bool(rd.get("enabled", True)),
        fallback_path=rd.get("fallback_path", "data/db/events_redis_fallback.jsonl"),
    )

    return (detector, tracker, face_engine, gallery, store, entry_exit, overlay,
            fusion, boundary, reasoner, att_db, attendance,
            publisher, using_door_engine)


def _fix_counts(engine: EntryExitEngine) -> None:
    """Keep the IN/OUT/present counters consistent after reasoner adjustments."""
    engine.counts["entry"] = max(0, engine.counts.get("entry", 0))
    engine.counts["exit"] = max(0, engine.counts.get("exit", 0))
    engine.counts["present"] = engine.counts["entry"] - engine.counts["exit"]


def run_pipeline(
    cfg: dict[str, Any],
    *,
    source_override: Any = None,
    output_path: str | Path | None = None,
    display: bool = True,
    max_frames: int | None = None,
    skip_frames: int | None = None,
    face_every_n: int | None = None,
) -> PipelineResult:
    cam_cfg = cfg["camera"]
    pipe = cfg["pipeline"]

    source = normalize_source(source_override if source_override is not None else cam_cfg["source"])
    (detector, tracker, face_engine, gallery, store, event_engine, overlay,
     fusion, boundary, reasoner, att_db, attendance,
     publisher, using_door_engine) = _build_components(cfg)

    if skip_frames is None:
        skip_frames = int(pipe.get("skip_frames", 1))
    if face_every_n is None:
        face_every_n = int(pipe.get("face_every_n", 1))

    writer: Optional[cv2.VideoWriter] = None
    frames_processed = 0
    total_events = 0
    last_dets = []
    last_faces = []
    seen_ids: set[int] = set()
    boundary_applied = False
    last_boundary_progress = -1
    last_fused = -1
    output_file = Path(output_path) if output_path else None

    # Fast people cross the line before face recognition runs. Their events
    # are held in pending_events for a short grace period; if face
    # recognition names the track within that window, the event's person is
    # corrected before it reaches the attendance/CSV pipeline.
    NAME_GRACE_SEC = 2.0
    pending_events: Dict[int, Event] = {}
    pending_deadline: Dict[int, float] = {}

    def _handle_event(e: Event) -> None:
        """Run reasoning → attendance → publish (Redis) for a single (finalised) event."""
        verdicts = reasoner.verify(
            [e],
            boundary.confidence if boundary else 0.0,
            boundary.learned if boundary else False,
        )
        v = verdicts[0]
        if v.action == "reject":
            store.delete(v.event.id)
            event_engine.counts[v.event.direction] = event_engine.counts.get(v.event.direction, 0) - 1
            if v.void_previous_id:
                prev = store.get(v.void_previous_id)
                store.delete(v.void_previous_id)
                if prev is not None:
                    prev_dir = prev.get("direction")
                    if prev_dir:
                        event_engine.counts[prev_dir] = event_engine.counts.get(prev_dir, 0) - 1
                print(
                    f"[REASON] rejected {v.event.person} {v.event.direction} "
                    f"({v.note}) voided event #{v.void_previous_id}"
                )
            else:
                print(f"[REASON] rejected {v.event.person} {v.event.direction} ({v.note})")
            _fix_counts(event_engine)
            return
        if v.action == "flip":
            old_dir = v.event.direction
            store.update_direction(v.event.id, v.direction)
            event_engine.counts[old_dir] = event_engine.counts.get(old_dir, 0) - 1
            event_engine.counts[v.direction] = event_engine.counts.get(v.direction, 0) + 1
            v.event.direction = v.direction
            print(f"[REASON] flipped {v.event.person} {old_dir}→{v.direction} ({v.note})")
        _fix_counts(event_engine)
        accepted = [v.event]
        outcomes = attendance.process_events(accepted) if attendance is not None else []
        outcome_by_key = {(o["person"], o["time"]): o for o in outcomes}
        e2 = v.event
        print(f"[EVENT] {e2.date} {e2.time} | {e2.person} | {e2.direction}")
        publisher.publish(e2)
        for alert in reasoner.alerts:
            print(f"[ALERT] {alert}")

    def _is_placeholder(name: str) -> bool:
        return not name or name.startswith(("Guest#", "Unknown", "ID:"))

    def _resolve_pending(tracks: List[Track]) -> None:
        """Flush pending events whose track got a real face name; expire others."""
        now_t = time.time()
        for tid, e in list(pending_events.items()):
            t = next((t for t in tracks if t.track_id == tid), None)
            if t is None:
                t = next((t for t in tracks if t.meta.get("global_id") == e.person), None)
            name = t.person_name if t is not None else ""
            if t is not None and not _is_placeholder(name):
                e.person = name
                store.update_person(e.id, name)
                pending_events.pop(tid)
                _handle_event(e)
                overlay.pulse_boundary()
            elif t is None or now_t >= pending_deadline.get(tid, 0.0):
                pending_events.pop(tid)
                _handle_event(e)
                overlay.pulse_boundary()

    try:
        with CameraStream(
            source=source,
            buffer_size=cam_cfg.get("buffer_size", 1),
            width=cam_cfg.get("width"),
            height=cam_cfg.get("height"),
        ) as stream:
            while True:
                ok, frame = stream.read()
                if not ok or frame is None:
                    if frames_processed == 0:
                        print("[Pipeline] ERROR: Could not read first frame — check source path / camera.")
                    else:
                        print(f"[Pipeline] Stream ended after {frames_processed} frames.")
                    break

                frames_processed += 1
                if max_frames is not None and frames_processed > max_frames:
                    break

                if using_door_engine and not overlay._zones_px:
                    # Draw the calibrated polygons once frame size is known.
                    fh, fw = frame.shape[:2]
                    overlay.set_zones(
                        {
                            name: [(int(x * fw), int(y * fh)) for x, y in poly]
                            for name, poly in event_engine.zones.items()
                        }
                    )

                run_det = (frames_processed - 1) % max(1, skip_frames) == 0
                run_face = (frames_processed - 1) % max(1, face_every_n) == 0

                if run_det:
                    last_dets = detector.detect(frame)
                tracks = tracker.update(last_dets)

                for t in tracks:
                    if t.track_id not in seen_ids:
                        overlay.note_new_track(t.track_id)
                        seen_ids.add(t.track_id)

                if run_face:
                    last_faces = face_engine.detect_and_embed(frame, min_face_px=cfg["models"]["min_face_px"])
                # Always re-match last known faces to current tracks (even on
                # non-face frames) so person_name is populated before entry/exit.
                if last_faces:
                    attach_faces_to_tracks(tracks, last_faces, gallery.match)
                    for f in last_faces:
                        if f.name != "Unknown":
                            print(f"[Face] {f.name}  score={f.match_score:.2f}")

                # ── Identity Fusion & Re-ID ───────────────────────────────
                if fusion is not None:
                    fusion.update(tracks)

                # ── Auto boundary: feed trajectories, apply when learned ──
                # (skipped entirely while the polygon Door engine is active)
                if boundary is not None and not using_door_engine:
                    fh, fw = frame.shape[:2]
                    for t in tracks:
                        gid = t.meta.get("global_id") if fusion is not None else None
                        if gid:
                            boundary.feed(gid, t.centroid[0] / fw, t.centroid[1] / fh)
                    if boundary.learned and not boundary_applied:
                        event_engine.set_line(boundary.line_norm)
                        event_engine.entry_direction = boundary.entry_direction
                        overlay.set_boundary(boundary.line_norm, label="BOUNDARY")
                        boundary_applied = True
                        if att_db is not None:
                            att_db.insert_calibration(boundary.line_norm, boundary.confidence)
                        print(
                            f"[Boundary] LEARNED line={boundary.line_norm} "
                            f"entry={boundary.entry_direction} conf={boundary.confidence:.2f}"
                        )
                    elif not boundary_applied:
                        # Show the current (seed/EMA) line while calibrating
                        n, _ = boundary.progress()
                        if n >= 5:
                            overlay.set_boundary(boundary.line_norm, label="ZONE:CALIBRATING")
                    n, total = boundary.progress()
                    if n != last_boundary_progress and n > 0 and n % 10 == 0:
                        print(f"[Boundary] learning… {n}/{total} trajectory vectors")
                        last_boundary_progress = n
                elif not boundary_applied and overlay.show_boundary and not using_door_engine:
                    # Auto-learning disabled → draw the static configured line
                    overlay.set_boundary(event_engine.line_norm, label="BOUNDARY")
                    boundary_applied = True

                # ── Entry/exit events → reasoning → attendance → Redis ────
                events = event_engine.update(tracks, frame.shape, store)
                if events:
                    total_events += len(events)
                    now_t = time.time()
                    for e in events:
                        if _is_placeholder(e.person):
                            pending_events[e.track_id] = e
                            pending_deadline[e.track_id] = now_t + NAME_GRACE_SEC
                        else:
                            _handle_event(e)
                            overlay.pulse_boundary()

                _resolve_pending(tracks)

                if fusion is not None:
                    stats = fusion.stats()
                    total_fused = stats["stitches"] + stats["face_merges"]
                    if total_fused != last_fused and total_fused > 0:
                        print(
                            f"[IdentityFusion] tracks→identities={len(fusion.track_identity)} "
                            f"stitches={stats['stitches']} face_merges={stats['face_merges']}"
                        )
                        last_fused = total_fused

                vis = overlay.draw(frame, tracks, last_faces, event_engine.counts)

                if output_file is not None and writer is None:
                    fps = float(stream.cap.get(cv2.CAP_PROP_FPS)) if stream.cap is not None else 0.0
                    if not fps or fps != fps or fps < 1.0:
                        fps = 25.0
                    h, w = vis.shape[:2]
                    writer = _make_writer(output_file, fps, (w, h))

                if writer is not None:
                    writer.write(vis)

                if display:
                    cv2.imshow("Person Face Events (CPU)", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
    finally:
        for tid, e in list(pending_events.items()):
            pending_events.pop(tid)
            _handle_event(e)
            overlay.pulse_boundary()
        publisher.close()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    # ── End-of-run attendance summary ──────────────────────────────────────
    if attendance is not None:
        logs = attendance.summary()
        if logs:
            print("\n── Attendance Summary ─────────────────────────────")
            for row in logs:
                print(
                    f"  {row['date']} | {row['person_id']:<14} | in {row['check_in_time'] or '-':8} | "
                    f"out {row['check_out_time'] or 'present':8} | {row['status']:<16} | "
                    f"{(str(row['work_hours']) + 'h') if row['work_hours'] is not None else '—'}"
                )
            print("────────────────────────────────────────────────────\n")
        else:
            print("[Attendance] no records for today (no verified crossings)")

    return PipelineResult(
        frames_processed=frames_processed,
        events_count=total_events,
        source=str(source),
        output_path=str(output_file) if output_file else "",
    )
