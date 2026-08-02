#!/usr/bin/env bash
# Start go2rtc restreamer in the background.
# Web UI: http://127.0.0.1:1984
set -e

cd "$(dirname "$0")/.."
mkdir -p data

# Ensure direct-connection ethernet has an IP on the camera subnet
if ! ifconfig en6 2>/dev/null | grep -q "inet 192.168.2\."; then
  echo "Setting en6 to 192.168.2.100 (camera subnet)…"
  sudo ifconfig en6 192.168.2.100 netmask 255.255.255.0 up
fi

if pgrep -x go2rtc >/dev/null 2>&1; then
  echo "go2rtc already running"
  exit 0
fi

nohup ./go2rtc -c go2rtc.yaml > data/go2rtc.log 2>&1 &
echo "go2rtc started (PID $!). Web UI: http://127.0.0.1:1984"
