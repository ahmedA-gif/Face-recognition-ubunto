#!/usr/bin/env bash
# setup_camera.sh — configure the network + verify camera access (Ubuntu/macOS)
#
# Steps performed:
#   1. Find the wired/USB-Ethernet interface (prefers enx* / en* USB adapters)
#   2. Assign 192.168.2.100/24 on that interface (camera subnet)
#   3. Ping the camera to confirm the link
#   4. Probe RTSP port 554
#   5. If reachable, start go2rtc and run the pipeline
set -e

CAM_IP="${CAM_IP:-192.168.2.112}"
LOCAL_IP="${LOCAL_IP:-192.168.2.100}"

echo "═ Camera access setup ═"
echo "  Camera : $CAM_IP"
echo "  Local  : $LOCAL_IP"

if command -v ip >/dev/null 2>&1; then
  # ── Linux (Ubuntu) ────────────────────────────────────────────────────────
  echo "── Detecting wired interface (Ubuntu) ──"
  CAM_IF=""
  for iface in $(ip -o link show | awk -F': ' '!/lo/{print $2}'); do
    case "$iface" in
      wl*|ww*|docker*|veth*|br-*|virbr*|tun*|tap*) continue ;;
    esac
    carrier=$(cat "/sys/class/net/$iface/carrier" 2>/dev/null || echo 0)
    [ "$carrier" != "1" ] && continue
    case "$iface" in
      enx*|usb*) CAM_IF="$iface"; break ;;
      *) [ -z "$CAM_IF" ] && CAM_IF="$iface" ;;
    esac
  done
  if [ -z "$CAM_IF" ]; then
    echo "  ERROR: no wired interface with a live cable (carrier=1) found."
    echo "         Is the camera's Ethernet adapter plugged in and the camera powered?"
    ip link show
    exit 1
  fi
  echo "  Interface: $CAM_IF (carrier=1)"
  # Remove stale copy of the camera IP from dead interfaces (e.g. eno1 NO-CARRIER)
  for iface in $(ip -o addr show | awk -F': ' "/${LOCAL_IP%/*}/{print \$2}"); do
    if [ "$iface" != "$CAM_IF" ]; then
      echo "  Removing stale $LOCAL_IP from $iface (no cable link)..."
      sudo ip addr del "$LOCAL_IP/24" dev "$iface" 2>/dev/null || true
    fi
  done
  if ! ip addr show dev "$CAM_IF" | grep -q "$LOCAL_IP"; then
    echo "  Assigning $LOCAL_IP/24 on $CAM_IF …"
    sudo ip addr add "$LOCAL_IP/24" dev "$CAM_IF" 2>/dev/null || true
    sudo ip link set "$CAM_IF" up
  else
    echo "  $LOCAL_IP already set on $CAM_IF"
  fi
elif command -v ifconfig >/dev/null 2>&1; then
  # ── macOS ─────────────────────────────────────────────────────────────────
  echo "── Detecting wired interface (macOS) ──"
  CAM_IF=""
  for iface in $(networksetup -listallhardwareports 2>/dev/null | grep "Device:" | awk '{print $2}'); do
    case "$iface" in
      en0|en1|en2|en3|en4|en5) continue ;;   # skip WiFi/Thunderbolt built-ins
    esac
    CAM_IF="$iface"
    break
  done
  [ -z "$CAM_IF" ] && CAM_IF="en6"
  echo "  Interface: $CAM_IF"
  if ! ifconfig "$CAM_IF" 2>/dev/null | grep -q "$LOCAL_IP"; then
    echo "  Assigning $LOCAL_IP on $CAM_IF …"
    sudo ifconfig "$CAM_IF" "$LOCAL_IP" netmask 255.255.255.0 up
  else
    echo "  $LOCAL_IP already set on $CAM_IF"
  fi
else
  echo "ERROR: neither 'ip' nor 'ifconfig' available."
  exit 1
fi

echo "── Testing camera $CAM_IP ──"
if ping -c 3 -W 2 "$CAM_IP" >/dev/null 2>&1; then
  echo "  ✓ Camera responds to ping"
else
  echo "  ✗ No ping reply from $CAM_IP"
fi

echo "── Checking physical link state ──"
if command -v ip >/dev/null 2>&1; then
  LINK_STATE=$(ip link show "$CAM_IF" 2>/dev/null | grep -o "state [A-Z]*")
  CARRIER=$(cat /sys/class/net/"$CAM_IF"/carrier 2>/dev/null || echo "?")
  echo "  $CAM_IF: $LINK_STATE (carrier=$CARRIER)"
  if [ "$CARRIER" = "0" ]; then
    echo "  ✗ NO CABLE LINK — ethernet cable is not detected."
    echo "    • Plug the cable into the camera AND the adapter"
    echo "    • Make sure the camera is powered on (LED lit)"
    echo "    • Try a different cable / port"
  fi
else
  ifconfig "$CAM_IF" 2>/dev/null | grep -E "status|media"
fi

echo "── Scanning 192.168.2.x for the camera ──"
if command -v nmap >/dev/null 2>&1; then
  nmap -sn 192.168.2.0/24 2>/dev/null | grep -E "report|Host is up" || echo "  (nothing found)"
else
  echo "  (nmap not installed — run: sudo apt install nmap)"
fi

echo "── Checking RTSP port 554 ──"
if nc -z -w 3 "$CAM_IP" 554 2>/dev/null; then
  echo "  ✓ RTSP port 554 OPEN"
else
  echo "  ✗ RTSP port 554 closed — camera not reachable on $CAM_IP"
  echo "    If the link is UP but no device answers, the camera IP changed —"
  echo "    scan other subnets:  nmap -sn 192.168.0.0/24 192.168.1.0/24"
fi

echo "── Done ──"
echo "Next steps:"
echo "  ./scripts/start_go2rtc.sh"
echo "  python3 scripts/run_video.py"
