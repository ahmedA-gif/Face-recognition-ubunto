#!/usr/bin/env python3
"""Attendance worker — consumes confirmed events from the Redis Stream.

Separate process from the video pipeline. Reads ``attendance:events`` with
XREADGROUP, validates each payload, dedupes by ``event_id``, persists to the
SQLite EventsStore (dev/default) and pushes through the AttendanceManager so
check-in/check-out stays consistent even if this worker crashes or Postgres is
added later. The AI pipeline never waits on this process — Redis buffers.

Usage:
    .venv/bin/python3 scripts/attendance_worker.py
    .venv/bin/python3 scripts/attendance_worker.py --stream attendance:events --db data/db/events.db
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attendance.db import AttendanceDB
from src.attendance.manager import AttendanceManager
from src.events.store import Event, EventsStore
from src.utils.config import load_settings

GROUP = "attendance-worker"
CONSUMER = "worker-1"


def _connect(url: str):
    try:
        import redis
    except ImportError as exc:
        sys.exit(f"redis module required: pip install redis ({exc})")
    client = redis.Redis.from_url(url, decode_responses=True)
    client.ping()
    return client


def _parse(payload: dict) -> Event:
    data = payload.get("data")
    obj = data if isinstance(data, dict) else json.loads(data)
    return Event(
        event_id=obj.get("event_id", ""),
        date=obj.get("date", ""),
        time=obj.get("time", ""),
        person=obj.get("employee_id") or obj.get("global_id") or "Unknown",
        direction="entry" if (obj.get("event") or "").upper() == "ENTRY" else "exit",
        track_id=int(obj.get("track_id") or 0),
        camera_id=obj.get("camera_id", "cam_01"),
        confidence=float(obj.get("confidence") or 0.0),
        global_id=obj.get("global_id", ""),
        fsm_path=list(obj.get("fsm_path") or []),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Consume Redis Stream events → SQLite + attendance.")
    ap.add_argument("--stream", default=None, help="Stream name (default: config redis.stream).")
    ap.add_argument("--url", default=None, help="Redis URL (default: config redis.url).")
    ap.add_argument("--db", default=None, help="Events SQLite path (default: config events.db_path).")
    ap.add_argument("--block-ms", type=int, default=2000, help="XREADGROUP block milliseconds.")
    args = ap.parse_args()

    cfg = load_settings()
    rd = cfg.get("redis", {})
    stream = args.stream or rd.get("stream", "attendance:events")
    url = args.url or rd.get("url", "redis://localhost:6379/0")
    db_path = args.db or cfg["events"]["db_path"]

    client = _connect(url)
    try:
        client.xgroup_create(stream, GROUP, id="0", mkstream=True)
    except Exception:  # group already exists
        pass

    store = EventsStore(db_path)
    att_db = AttendanceDB(cfg.get("attendance", {}).get("db_path", "data/db/attendance.db"))
    manager = AttendanceManager(
        db=att_db,
        shift_start=cfg.get("attendance", {}).get("shift_start", "09:00"),
        shift_end=cfg.get("attendance", {}).get("shift_end", "17:00"),
        late_threshold_mins=cfg.get("attendance", {}).get("late_threshold_mins", 15),
        early_exit_mins=cfg.get("attendance", {}).get("early_exit_mins", 15),
        debounce_minutes=cfg.get("attendance", {}).get("debounce_minutes", 2.0),
    )
    seen: set = set()

    print(f"[Worker] consuming stream '{stream}' from {url} → {db_path}")
    print(f"[Worker] Ctrl+C to stop.")
    try:
        while True:
            entries = client.xreadgroup(
                GROUP, CONSUMER, {stream: ">"}, count=32, block=args.block_ms
            )
            if not entries:
                continue
            for _stream, messages in entries:
                for message_id, payload in messages:
                    try:
                        event = _parse(payload)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[Worker] SKIP malformed payload {message_id}: {exc}")
                        client.xack(stream, GROUP, message_id)
                        continue
                    if event.event_id and event.event_id in seen:
                        client.xack(stream, GROUP, message_id)
                        continue
                    if event.event_id:
                        seen.add(event.event_id)

                    # Dedup against SQLite before writing (idempotent consume).
                    if event.event_id and store.get_by_event_id(event.event_id):
                        client.xack(stream, GROUP, message_id)
                        continue

                    event.id = store.insert(event)
                    manager.process_events([event])
                    print(
                        f"[Worker] {event.direction.upper():5} {event.person:<14} "
                        f"{event.date} {event.time}  ({event.fsm_path})"
                    )
                    client.xack(stream, GROUP, message_id)
    except KeyboardInterrupt:
        print("\n[Worker] stopped.")
    finally:
        store.close()
        att_db.close()


if __name__ == "__main__":
    main()
