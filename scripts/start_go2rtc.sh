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
    echo "Created go2rtc.yaml — edit it with your camera credentials"
    exit 0
  fi
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
  echo "go2rtc failed to start. Check data/go2rtc.log"
  cat data/go2rtc.log 2>/dev/null | tail -20
  exit 1
fi
