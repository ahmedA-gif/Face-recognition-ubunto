#!/usr/bin/env bash
# Start go2rtc restreamer in the background.
# Web UI: http://127.0.0.1:1984
set -e

cd "$(dirname "$0")/.."
mkdir -p data

# Create go2rtc.yaml from example if not exists
if [ ! -f go2rtc.yaml ]; then
  if [ -f config/go2rtc.yaml.example ]; then
    cp config/go2rtc.yaml.example go2rtc.yaml
    echo "Created go2rtc.yaml from example — edit it with your camera credentials"
    exit 0
  fi
fi

# Detect camera interface (Linux: eth0/enp0s3, macOS: en6/en7)
CAM_IF=$(ip -o link show | awk -F': ' '!/lo/{print $2; exit}' 2>/dev/null || echo "eth0")

# Ensure direct-connection ethernet has an IP on the camera subnet
if ! ip addr show "$CAM_IF" 2>/dev/null | grep -q "192.168.2."; then
  echo "Setting $CAM_IF to 192.168.2.100 (camera subnet)…"
  sudo ip addr add 192.168.2.100/24 dev "$CAM_IF" 2>/dev/null || true
  sudo ip link set "$CAM_IF" up
fi

if pgrep -x go2rtc >/dev/null 2>&1; then
  echo "go2rtc already running"
  exit 0
fi

nohup ./go2rtc -c go2rtc.yaml > data/go2rtc.log 2>&1 &
echo "go2rtc started (PID $!). Web UI: http://127.0.0.1:1984"
