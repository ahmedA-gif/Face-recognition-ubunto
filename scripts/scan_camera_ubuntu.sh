#!/usr/bin/env bash
# scan_camera_ubuntu.sh — find the Dahua camera's real IP on a direct USB-Ethernet link
# Camera MAC: 30:dd:aa:98:7b:f4
set -e

CAMERA_MAC="30:dd:aa:98:7b:f4"
LOCAL_IP="192.168.2.100"

echo "=== Step 1: find the live wired interface ==="
CAM_IF=""
for iface in $(ip -o link show | awk -F': ' '!/lo/{print $2}'); do
  case "$iface" in
    wl*|ww*|docker*|veth*|br-*|virbr*|tun*|tap*) continue ;;
  esac
  carrier=$(cat "/sys/class/net/$iface/carrier" 2>/dev/null || echo 0)
  [ "$carrier" != "1" ] && continue
  CAM_IF="$iface"
  break
done
if [ -z "$CAM_IF" ]; then
  echo "ERROR: no wired interface has carrier=1. Cable not plugged or camera not powered."
  ip link show
  exit 1
fi
echo "  Using interface: $CAM_IF (carrier=1)"

echo "=== Step 2: assign camera-subnet IP ==="
sudo ip addr add "$LOCAL_IP/24" dev "$CAM_IF" 2>/dev/null || true
sudo ip link set "$CAM_IF" up
sleep 2

echo "=== Step 3: scan subnets for the camera MAC ==="
for subnet in 192.168.2 192.168.1 192.168.0 10.1.1 192.168.3 192.168.100; do
  echo "  Scanning $subnet.0/24 ..."
  for i in $(seq 1 254); do
    ping -c 1 -W 1 "$subnet.$i" > /dev/null 2>&1 &
  done
  wait
  sleep 1
  MATCH=$(arp -n | grep -i "$CAMERA_MAC")
  if [ -n "$MATCH" ]; then
    echo "  >>> CAMERA FOUND: $MATCH"
    echo "CAMERA_FOUND=yes"
    break
  fi
done

echo "=== Step 4: full ARP table on $CAM_IF ==="
ip neigh show dev "$CAM_IF"

echo "=== Step 5: probe RTSP on 192.168.2.112 and .112 alternatives ==="
for ip in 192.168.2.112 192.168.1.108 192.168.0.112; do
  if nc -z -w 2 "$ip" 554 2>/dev/null; then
    echo "  RTSP OPEN on $ip — camera lives here"
  else
    echo "  RTSP closed on $ip"
  fi
done

echo "=== Done ==="
