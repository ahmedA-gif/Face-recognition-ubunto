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
  if [ "$ARCH" = "x86_64" ]; then
    curl -L -o go2rtc.zip "https://github.com/AlexxIT/go2rtc/releases/latest/download/go2rtc_linux_amd64.zip"
  elif [ "$ARCH" = "aarch64" ]; then
    curl -L -o go2rtc.zip "https://github.com/AlexxIT/go2rtc/releases/latest/download/go2rtc_linux_arm64.zip"
  else
    curl -L -o go2rtc.zip "https://github.com/AlexxIT/go2rtc/releases/latest/download/go2rtc_linux_armv7.zip"
  fi
  unzip -o go2rtc.zip go2rtc
  rm -f go2rtc.zip
  chmod +x go2rtc
  echo "go2rtc installed: $(./go2rtc --version 2>&1 || echo 'ok')"
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

# Verify it started
if pgrep -x go2rtc >/dev/null 2>&1; then
  echo "go2rtc started (PID $!). Web UI: http://127.0.0.1:1984"
else
  echo "go2rtc failed to start. Check data/go2rtc.log"
  cat data/go2rtc.log 2>/dev/null | tail -20
  exit 1
fi
