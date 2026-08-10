#!/usr/bin/env bash
# run_attendance.sh — 24/7 attendance wrapper (camera network + go2rtc + pipeline)
#
# Used by the systemd unit "attendance.service". Runs headless.
# - start_go2rtc.sh: finds the live camera interface (carrier=1), assigns
#   192.168.2.100/24, and starts the go2rtc restreamer (idempotent).
# - run_video.py --no-display: processes the live CCTV stream and writes
#   verified entry/exit events to SQLite + Redis Streams in real time.
set -e
cd "$(dirname "$0")/.."

./scripts/start_go2rtc.sh

exec sudo -u rahmat .venv/bin/python3 -u scripts/run_video.py --no-display
