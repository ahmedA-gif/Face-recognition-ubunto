#!/usr/bin/env bash
# One-shot camera test — run ONLY after 30 min of no attempts.
set -e

CAMERA="rtsp://admin:admin1234@192.168.2.112:554/cam/realmonitor?channel=1&subtype=1"

echo "Testing camera connection..."
OUT=$(ffprobe -v error -rtsp_transport tcp -timeout 10000000 \
  "$CAMERA" \
  -show_entries stream=codec_name,width,height \
  -of default=noprint_wrappers=1 2>&1) || true

if echo "$OUT" | grep -q "codec_name"; then
  echo "SUCCESS — camera connected!"
  echo "$OUT"
  echo ""
  echo "Now run: ./scripts/start_go2rtc.sh"
else
  echo "FAILED:"
  echo "$OUT"
  echo ""
  echo "Still locked or wrong password — wait another 30 min, then retry."
fi
