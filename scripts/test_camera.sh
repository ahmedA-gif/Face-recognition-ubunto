#!/usr/bin/env bash
# One-shot camera test — check link first, then RTSP once.
# Do not spam auth: Dahua can lock after failed logins.
set -e

CAMERA="rtsp://admin:admin1234@192.168.2.112:554/cam/realmonitor?channel=1&subtype=1"
CAM_IP="192.168.2.112"

echo "── Pre-check: physical link + route ──"
HAS_CARRIER=0
for iface in $(ip -o link show | awk -F': ' '{print $2}'); do
  case "$iface" in wl*|ww*|docker*|lo|veth*|br-*) continue ;; esac
  c=$(cat "/sys/class/net/$iface/carrier" 2>/dev/null || echo 0)
  if [ "$c" = "1" ]; then
    echo "  $iface carrier=1"
    HAS_CARRIER=1
  fi
done
if [ "$HAS_CARRIER" != "1" ]; then
  echo "FAILED: no wired interface has carrier=1 (NO physical Ethernet link)."
  echo "  This is NOT a password lockout. Fix the cable/power first:"
  echo "    bash scripts/fix_camera_link.sh"
  exit 1
fi
if ! ip route get "$CAM_IP" 2>/dev/null | grep -q .; then
  echo "FAILED: no route to $CAM_IP — assign 192.168.2.100/24 on the USB-Eth interface."
  echo "    bash scripts/fix_camera_link.sh"
  exit 1
fi
echo "  route: $(ip route get "$CAM_IP" 2>/dev/null | head -1)"
if ! ping -c 1 -W 2 "$CAM_IP" >/dev/null 2>&1; then
  echo "FAILED: $CAM_IP does not answer ping (link up but wrong IP or camera off)."
  echo "    bash scripts/fix_camera_link.sh"
  exit 1
fi
echo "  ping $CAM_IP OK"

echo "Testing camera RTSP..."
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
  case "$OUT" in
    *"Cannot assign requested address"*|*"Network is unreachable"*|*"No route"*)
      echo "Network path broken — run: bash scripts/fix_camera_link.sh"
      ;;
    *"401"*|*"Unauthorized"*|*"403"*|*"Auth"*)
      echo "Auth failed — wrong password, or Dahua lockout (wait 30 min, retry ONCE)."
      ;;
    *)
      echo "If auth-related: wait 30 min with zero attempts, then retry once."
      echo "If link-related: bash scripts/fix_camera_link.sh"
      ;;
  esac
fi
