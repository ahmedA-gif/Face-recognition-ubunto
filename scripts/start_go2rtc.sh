#!/usr/bin/env bash
# Start go2rtc restreamer in the background.
# Web UI: http://127.0.0.1:1984
set -e

cd "$(dirname "$0")/.."
mkdir -p data

# Download go2rtc for Linux if not installed
if [ ! -f go2rtc ] || ! ./go2rtc --version >/dev/null 2>&1; then
  echo "Downloading go2rtc for Linux..."
  ARCH=$(uname -m)
  case "$ARCH" in
    x86_64)  URL="https://github.com/AlexxIT/go2rtc/releases/download/v1.9.14/go2rtc_linux_amd64" ;;
    aarch64) URL="https://github.com/AlexxIT/go2rtc/releases/download/v1.9.14/go2rtc_linux_arm64" ;;
    armv7l)  URL="https://github.com/AlexxIT/go2rtc/releases/download/v1.9.14/go2rtc_linux_armv7" ;;
    *)       echo "Unsupported arch: $ARCH"; exit 1 ;;
  esac
  curl -L -o go2rtc "$URL"
  chmod +x go2rtc
  echo "go2rtc installed"
fi

# Create go2rtc.yaml from example if not exists
if [ ! -f go2rtc.yaml ]; then
  if [ -f config/go2rtc.yaml.example ]; then
    cp config/go2rtc.yaml.example go2rtc.yaml
  else
    echo "No go2rtc.yaml found. Copy config/go2rtc.yaml to project root."
    exit 1
  fi
fi

# Find the USB ethernet adapter connected to camera and assign IP
# Priority: 1) enx* (USB-Ethernet, e.g. enxc8a362908ac9)  2) any non-wifi NIC
CAM_IF=""
for iface in $(ip -o link show | awk -F': ' '!/lo/{print $2}'); do
  case "$iface" in
    wl*|ww*|docker*|veth*|br-*|virbr*|tun*|tap*) continue ;;
  esac
  if [ -z "$CAM_IF" ]; then
    CAM_IF="$iface"
  fi
  case "$iface" in
    enx*|usb*) CAM_IF="$iface"; break ;;   # prefer USB ethernet adapter
  esac
done
if [ -z "$CAM_IF" ]; then
  echo "ERROR: no wired interface found. Is the camera's USB-Ethernet adapter plugged in?"
  ip link show
  exit 1
fi
echo "Using camera interface: $CAM_IF"

# Check if any interface has 192.168.2.x — if not, assign to USB ethernet
if ! ip addr show | grep -q "192.168.2."; then
  echo "Adding 192.168.2.100 to $CAM_IF for camera access..."
  sudo ip addr add 192.168.2.100/24 dev "$CAM_IF" 2>/dev/null || true
  sudo ip link set "$CAM_IF" up
  sleep 2
fi

if pgrep -x go2rtc >/dev/null 2>&1; then
  echo "go2rtc already running"
  exit 0
fi

nohup ./go2rtc -c go2rtc.yaml > data/go2rtc.log 2>&1 &
sleep 2

if pgrep -x go2rtc >/dev/null 2>&1; then
  echo "go2rtc started (PID $!). Web UI: http://127.0.0.1:1984"
else
  echo "go2rtc failed. Check data/go2rtc.log"
  cat data/go2rtc.log 2>/dev/null | tail -20
  exit 1
fi
