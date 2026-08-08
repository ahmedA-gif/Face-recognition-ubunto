"""Redis Streams publisher for confirmed entry/exit events.

The hot path is ``XADD attendance:events`` (~1 ms), so the video loop never
blocks on a database. If Redis is unavailable or the module is not installed
(dev box), the publisher degrades gracefully: it logs a warning once and
appends the JSON payload to a local JSONL file. The SQLite EventsStore
written by the pipeline remains the durable local log in both modes.

Payload contract (consumed by ``scripts/attendance_worker.py``):

    {
      "event_id":   "<uuid>",
      "camera_id":  "cam_01",
      "track_id":   <int>,
      "global_id":  "<global_person_id>",
      "employee_id": "<face name | null>",
      "event":      "ENTRY" | "EXIT",
      "confidence": <float>,
      "fsm_path":   ["OUTSIDE", "DOOR", "INSIDE"],
      "timestamp":  "%Y-%m-%dT%H:%M:%SZ",
      "date":       "%Y-%m-%d",
      "time":       "%H:%M:%S"
    }
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from src.events.store import Event

logger = logging.getLogger("redis_publisher")

_JSON_FIELDS = (
    "event_id",
    "camera_id",
    "track_id",
    "global_id",
    "employee_id",
    "event",
    "confidence",
    "fsm_path",
    "timestamp",
    "date",
    "time",
)


def _encode(event: Event, camera_id: Optional[str]) -> dict:
    direction = (event.direction or "").upper()
    event_type = "ENTRY" if direction in ("entry", "ENTRY") else "EXIT"
    return {
        "event_id": event.event_id or str(uuid.uuid4()),
        "camera_id": camera_id or event.camera_id,
        "track_id": int(event.track_id),
        "global_id": event.global_id or event.person,
        "employee_id": event.person if not event.person.startswith(("Unknown", "Guest#", "ID:")) else None,
        "event": event_type,
        "confidence": round(float(event.confidence or 0.0), 4),
        "fsm_path": list(event.fsm_path) if event.fsm_path else [],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "date": event.date,
        "time": event.time,
    }


class EventPublisher:
    """XADD events to a Redis stream with a JSONL fallback."""

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        stream: str = "attendance:events",
        maxlen: int = 10000,
        enabled: bool = True,
        fallback_path: Optional[str] = None,
    ) -> None:
        self.url = url
        self.stream = stream
        self.maxlen = int(maxlen)
        self.enabled = bool(enabled)
        self.fallback_path = Path(fallback_path) if fallback_path else None
        self._client = None
        self._warned = False
        self.published = 0

        if self.enabled:
            self._connect()

    # ── public API ─────────────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        return self._client is not None

    def publish(self, event: Event) -> bool:
        """Publish one event to the Redis stream. Never raises."""
        payload = _encode(event, None)
        if self.active:
            try:
                self._client.xadd(self.stream, {"data": json.dumps(payload)}, maxlen=self.maxlen)
                self.published += 1
                return True
            except Exception as exc:  # noqa: BLE001 — degrade, never crash AI loop
                self._warn_once(f"Redis publish failed ({exc}) — falling back to JSONL.")
                self._client = None
        self._fallback(payload)
        return False

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    # ── internals ──────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        try:
            import redis  # lazy import — module is optional on dev boxes
        except ImportError:
            self._warn_once("redis module not installed (pip install redis) — using JSONL fallback.")
            return
        try:
            client = redis.Redis.from_url(self.url, decode_responses=True)
            client.ping()
            self._client = client
            print(f"[RedisPublisher] active → {self.url} stream={self.stream}")
        except Exception as exc:  # noqa: BLE001
            self._warn_once(f"Redis unreachable ({exc}) — using JSONL fallback.")

    def _fallback(self, payload: dict) -> None:
        if self.fallback_path is None:
            return
        try:
            self.fallback_path.parent.mkdir(parents=True, exist_ok=True)
            with self.fallback_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def _warn_once(self, message: str) -> None:
        if not self._warned:
            self._warned = True
            print(f"[RedisPublisher] WARNING: {message}")
            logger.warning(message)
