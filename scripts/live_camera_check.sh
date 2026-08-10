#!/usr/bin/env bash
# live_camera_check.sh — real-time dongle link + camera ping monitor
# Usage: bash scripts/live_camera_check.sh   (enter sudo password once)
set -u

CAM_IP="192.168.2.112"
LOCAL_IP="192.168.2.100"

echo "Watching for USB dongle link (Ctrl+C to stop) ..."
echo "Camera target: $CAM_IP"

sudo -v

while true; do
  IF=""
  for i in $(ip -o link show | awk -F": " "/enx|usb/{print \$2}"); do
    C=$(cat /sys/class/net/$i/carrier 2>/dev/null || echo 0)
    if [ "$C" = "1" ]; then IF="$i"; break; fi
  done
  if [ -z "$IF" ]; then
    echo "[$(date +%T)] no dongle link (carrier=0) - re-seat cable/dongle..."
    sleep 1
    continue
  fi
  sudo ip addr add "$LOCAL_IP/24" dev "$IF" 2>/dev/null
  sudo ip link set "$IF" up
  if ping -c 1 -W 2 "$CAM_IP" >/dev/null 2>&1; then
    RTT=$(ping -c 1 "$CAM_IP" | grep -oE "time=.*" | head -1)
    echo "[$(date +%T)] LINK UP on $IF  -> CAMERA RESPONDS ($RTT)"
    break
  else
    echo "[$(date +%T)] carrier=1 on $IF but camera $CAM_IP NOT answering"
    sleep 2
  fi
done
