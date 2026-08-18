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
from src.events.crossing_line import CrossingLineEngine
from src.events.door_intelligence import DoorIntelligenceEngine
from src.events.dynamic_boundary import DynamicBoundaryEngine
from src.events.entry_exit import EntryExitEngine
from src.events.entry_exit_v2 import EntryExitEngineV2
from src.events.redis_publisher import EventPublisher
from src.events.store import Event, EventsStore
from src.overlay.draw import OverlayRenderer
from src.reasoning.spatial_temporal import SpatialTemporalReasoning
from src.recognition.face_engine import FaceEngine
from src.recognition.gallery import FaceGallery
from src.recognition.person_reid import PersonReIDEngine
from src.tracking.bytetrack import ByteTracker, Track
from src.tracking.identity_fusion import IdentityFusionEngine
from src.utils.assign import attach_faces_to_tracks, appearance_signature
from src.utils.config import load_zones


def _line_dict_to_tuple(line):
    """Convert {'x1':..,'y1':..,'x2':..,'y2':..} dict to (x1,y1,x2,y2) tuple."""
    if isinstance(line, (list, tuple)):
        return tuple(line)
    if isinstance(line, dict):
        return (float(line["x1"]), float(line["y1"]), float(line["x2"]), float(line["y2"]))
    return (0.25, 0.75, 0.75, 0.75)


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

    # Get backend from config or default to onnx
    backend = pipe.get("backend", "onnx")
    # Check environment variable for backend override
    import os
    env_backend = os.environ.get("VMS_BACKEND", "").lower()
    if env_backend:
        backend = env_backend
    
    detector = PersonDetector(
        weights=m.get("yolo_onnx", m["yolo_weights"]),
        conf=m["yolo_conf"],
        iou=m["yolo_iou"],
        imgsz=m["yolo_imgsz"],
        device=pipe["device"],
        person_class_id=m["person_class_id"],
        backend=backend,
    )
    tracker = ByteTracker()
    face_providers = None
    if str(pipe["device"]).startswith(("cuda", "gpu")):
        face_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    face_engine = FaceEngine(root=m["face_root"], pack=m["face_pack"], det_size=tuple(m["face_det_size"]), providers=face_providers)
    gallery = FaceGallery(
        db_path=ev["faces_db_path"],
        match_threshold=m["face_match_threshold"],
        backend="faiss",
    )
    print(f"[Pipeline] {gallery.status()}")
    store = EventsStore(db_path=ev["db_path"])
    
    # Use EntryExitEngineV2 as the primary engine (fixed version)
    # DoorIntelligence can be enabled as an alternative
    entry_exit = EntryExitEngineV2(
        line_norm=ee["line"],
        buffer_threshold=ee.get("buffer_threshold", 10.0),
        debounce_sec=ee.get("debounce_sec", 1.5),
        min_track_frames=ee.get("min_track_frames", 5),
        min_deep_frames=ee.get("min_deep_frames", 3),
        min_displacement=ee.get("min_displacement", 20.0),
        use_foot_point=bool(ee.get("use_foot_point", True)),
        segment_pad=float(ee.get("segment_pad", 0.12)),
        camera_id=ee.get("camera_id", "cam_01"),
    )
    overlay = OverlayRenderer(
        pulse_frames=ov["pulse_frames"],
        hud=ov["hud"],
        show_boundary=bool(ov.get("show_boundary", True)),
    )

    # ── Door Intelligence Engine (polygon FSM) — optional ──
    di = cfg.get("door_intelligence", {})
    using_door_engine = bool(di.get("enabled", False))

    # ── CrossingLine Engine (ObjectCounter-inspired) — preferred ──
    cl = cfg.get("crossing_line") or ee.get("crossing_line") or {}
    using_crossing_line = bool(cl.get("enabled", False))

    if using_crossing_line:
        entry_exit = CrossingLineEngine(
            line_norm=_line_dict_to_tuple(cl.get("line", ee.get("line", (0.25, 0.75, 0.75, 0.75)))),
            entry_direction=cl.get("entry_direction", ee.get("entry_direction", "upward")),
            camera_id=cl.get("camera_id", ee.get("camera_id", "cam_01")),
            min_track_frames=cl.get("min_track_frames", ee.get("min_track_frames", 3)),
            cooldown_sec=cl.get("cooldown_sec", 2.0),
            min_crossing_gap_sec=cl.get("min_crossing_gap_sec", 1.0),
        )
    elif using_door_engine:
        zones = dict(di.get("zones") or {}) or load_zones(
            di.get("zones_path", "config/zones.yaml"), di.get("camera_id")
        )
        if not zones:
            print(
                "[DoorIntelligence] enabled but no calibrated polygons found "
                "(config/zones.yaml) — falling back to EntryExitEngineV2 (line-based)."
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
                lock_after_event=di.get("lock_after_event", False),
                motion_toward_inside_dot=di.get("motion_toward_inside_dot", 0.25),
                motion_outward_dot=di.get("motion_outward_dot", 0.25),
                peek_max_inside_sec=di.get("peek_max_inside_sec", 1.5),
                uturn_max_door_sec=di.get("uturn_max_door_sec", 3.0),
                min_event_confidence=di.get("min_event_confidence", 0.70),
                purge_after_sec=di.get("purge_after_sec", 10.0),
                min_motion_speed=di.get("min_motion_speed", 0.6),
            )
    if using_crossing_line:
        engine_name = "CrossingLine (ObjectCounter-style)"
    elif using_door_engine:
        engine_name = "DoorIntelligence (polygon FSM)"
    else:
        engine_name = "EntryExitEngineV2 (line-based)"
    print(f"[EntryExit] Using {engine_name}")

    # ── new intelligence modules (config-gated) ────────────────────────────
    if_in = cfg.get("identity_fusion", {})

    # Optional clothing-invariant person-ReID (scripted/traced torch model).
    # Reads weights from identity_fusion.person_reid_weights OR
    # models.person_reid_weights; disabled (HSV fallback kept) otherwise.
    reid = None
    reid_weights = if_in.get("person_reid_weights") or m.get("person_reid_weights")
    if reid_weights:
        try:
            reid = PersonReIDEngine(
                weights=reid_weights,
                device=pipe.get("device", "cpu"),
                input_size=tuple(if_in.get("reid_input_size", [256, 128])),
            )
            print(f"[PersonReID] ACTIVE model={reid_weights} device={pipe.get('device', 'cpu')}")
        except Exception as exc:  # noqa: BLE001
            print(f"[PersonReID] DISABLED — could not load {reid_weights}: {exc}")

    if if_in.get("enabled", True):
        fusion = IdentityFusionEngine(
            max_stitch_dist_px=if_in.get("max_stitch_dist_px", 60),
            max_stitch_time_sec=if_in.get("max_stitch_time_sec", 2.0),
            embedding_match_threshold=if_in.get("embedding_match_threshold", m["face_match_threshold"]),
            appearance_match_threshold=if_in.get("appearance_match_threshold", 0.85),
            reid_match_threshold=if_in.get("reid_match_threshold", 0.82),
            max_pool_embeddings=if_in.get("max_pool_embeddings", 8),
            state_path=if_in.get("state_path", "data/db/identity_state.json"),
        )
    else:
        fusion = None
    print(f"[IdentityFusion] {'active' if fusion else 'disabled'}"
          f" (face_reid={if_in.get('embedding_match_threshold', m['face_match_threshold'])}, "
          f"appearance_reid={if_in.get('appearance_match_threshold', 0.85)}, "
          f"person_reid={if_in.get('reid_match_threshold', 0.82) if reid else 'off'})")

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
            publisher, using_door_engine, using_crossing_line, reid)


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

    # ── GPU auto-switch: check VRAM before building components ─────────
    from src.hardware.gpu_monitor import gpu_monitor
    gpu_cfg = cfg.get("gpu", {})
    gpu_monitor.vram_threshold_gb = gpu_cfg.get("vram_threshold_gb", 2.5)
    if gpu_monitor.try_use_gpu():
        print(f"[GPUMonitor] GPU OK — using CUDA")
    else:
        pipe["device"] = "cpu"
        print(f"[GPUMonitor] GPU unavailable or VRAM exceeded — using CPU")

    (detector, tracker, face_engine, gallery, store, event_engine, overlay,
     fusion, boundary, reasoner, att_db, attendance,
     publisher, using_door_engine, using_crossing_line, reid) = _build_components(cfg)
    using_polygon_or_line = using_door_engine or using_crossing_line

    if skip_frames is None:
        skip_frames = int(pipe.get("skip_frames", 1))
    if face_every_n is None:
        face_every_n = int(pipe.get("face_every_n", 1))
    reid_every_n = int(cfg.get("identity_fusion", {}).get("reid_every_n", 1))

    writer: Optional[cv2.VideoWriter] = None
    frames_processed = 0
    total_events = 0
    last_dets = []
    last_faces = []
    seen_ids: set[int] = set()
    _printed_tracks: set = set()
    _dbg_prev_state: dict = {}  # track_id -> (zone, fsm, dir)
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
        evidence = f" {e2.fsm_path}" if getattr(e2, "fsm_path", None) else ""
        face_note = f" | face={e2.person}" if e2.person and not _is_placeholder(e2.person) else ""
        print(
            f"[EVENT] {e2.date} {e2.time} | {e2.person} | {e2.direction}{evidence}"
            f" | conf={e2.confidence:.2f}{face_note}"
        )
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

                # Periodic GPU VRAM check — auto-switch to CPU if exceeded
                if frames_processed % 150 == 0 and not gpu_monitor.try_use_gpu():
                    if not str(pipe["device"]).startswith("cpu"):
                        pipe["device"] = "cpu"
                        print(f"[GPUMonitor] VRAM exceeded during run — switching to CPU")

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
                run_reid = (frames_processed - 1) % max(1, reid_every_n) == 0

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
                            print(f"[Face] {f.name}  score={f.match_score:.2f}  det={f.det_conf:.2f}" if hasattr(f, 'det_conf') else f"[Face] {f.name}  score={f.match_score:.2f}")

                # ── Identity Fusion & Re-ID ───────────────────────────────
                if fusion is not None:
                    for t in tracks:
                        if t.meta.get("appearance") is None:
                            t.meta["appearance"] = appearance_signature(frame, t.xyxy)
                    if reid is not None and run_reid:
                        # Clothing-invariant person-ReID per track (when the
                        # model is enabled); HSV stays as the backstop.
                        for t in tracks:
                            t.meta["person_reid"] = reid.extract(frame, t.xyxy)
                    fusion.update(tracks)

                # ── Auto boundary: feed trajectories, apply when learned ──
                # (skipped entirely while the polygon Door engine is active)
                if boundary is not None and not using_polygon_or_line:
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
                elif not boundary_applied and overlay.show_boundary and not using_polygon_or_line:
                    # Auto-learning disabled → draw the static configured line
                    overlay.set_boundary(event_engine.line_norm, label="BOUNDARY")
                    boundary_applied = True

                # ── Entry/exit events → reasoning → attendance → Redis ────
                events = event_engine.update(tracks, frame.shape, store)

                # ── Debug: print active tracks on NEW or state change ──────
                if tracks:
                    for t in tracks:
                        zone = t.meta.get("zone", "?")
                        fsm = t.meta.get("fsm_state", "-")
                        direction = t.meta.get("direction", "-")
                        state_key = (zone, fsm, direction)
                        is_new = t.track_id not in _printed_tracks
                        changed = _dbg_prev_state.get(t.track_id) != state_key
                        _dbg_prev_state[t.track_id] = state_key
                        if is_new or changed:
                            from src.utils.geometry import foot_point as _fp
                            _fp = _fp(t.xyxy)
                            _fx = _fp[0] / frame.shape[1]
                            _fy = _fp[1] / frame.shape[0]
                            _printed_tracks.add(t.track_id)
                            tag = "NEW" if is_new else "upd"
                            print(
                                f"[Track] id={t.track_id} {tag} "
                                f"person={t.person_name!r} conf={t.conf:.2f} "
                                f"feet=({_fx:.2f},{_fy:.2f}) zone={zone} fsm={fsm} "
                                f"dir={direction}"
                            )

                if events:
                    total_events += len(events)
                    now_t = time.time()
                    # Save snapshots for each event
                    _snap_dir = Path(cfg.get("events", {}).get("snapshots_dir", "data/snapshots"))
                    _snap_dir.mkdir(parents=True, exist_ok=True)
                    for e in events:
                        try:
                            snap_name = f"{e.date}_{e.time.replace(':', '-')}_{e.track_id}_{e.direction}.jpg"
                            snap_path = _snap_dir / snap_name
                            cv2.imwrite(str(snap_path), frame)
                            e.snapshot_path = str(snap_path)
                        except Exception:
                            pass
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
                    total_fused = stats["stitches"] + stats["face_merges"] + stats["reid_merges"]
                    if total_fused != last_fused and total_fused > 0:
                        print(
                            f"[IdentityFusion] tracks→identities={len(fusion.track_identity)} "
                            f"stitches={stats['stitches']} face_merges={stats['face_merges']}"
                            f" reid_merges={stats['reid_merges']}"
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
        if fusion is not None:
            fusion.close()
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
