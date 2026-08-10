#!/usr/bin/env bash
# fix_camera_link.sh — diagnose + fix direct PC↔camera Ethernet link (Ubuntu)
#
# Root cause of "carrier 0" / NO-CARRIER:
#   The USB-Ethernet adapter is detected by Linux, but the PHY sees NO link partner.
#   That is a PHYSICAL problem (cable / camera power / port), not a password problem.
#   IP / RTSP / ffprobe will ALL fail until carrier becomes 1.
#
# Topology expected:
#   PC USB port ── USB-Ethernet dongle (ASIX AX88179) ── RJ45 cable ── Camera (192.168.2.112)
#   WiFi stays on 192.168.0.x for internet; camera is a separate /24 on the dongle.
#
# Usage:
#   bash scripts/fix_camera_link.sh
#   WAIT_SECS=120 bash scripts/fix_camera_link.sh   # wait longer for you to re-seat cable
#   CAM_IP=192.168.2.112 LOCAL_IP=192.168.2.100 bash scripts/fix_camera_link.sh

set -euo pipefail

CAM_IP="${CAM_IP:-192.168.2.112}"
LOCAL_IP="${LOCAL_IP:-192.168.2.100}"
WAIT_SECS="${WAIT_SECS:-90}"
USER="${RTSP_USER:-admin}"
PASS="${RTSP_PASS:-admin1234}"
RTSP_URL="rtsp://${USER}:${PASS}@${CAM_IP}:554/cam/realmonitor?channel=1&subtype=1"

red()    { printf '\033[1;31m%s\033[0m\n' "$*"; }
green()  { printf '\033[1;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[1;33m%s\033[0m\n' "$*"; }
bold()   { printf '\033[1m%s\033[0m\n' "$*"; }

need_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

carrier_of() {
  cat "/sys/class/net/$1/carrier" 2>/dev/null || echo 0
}

# Prefer USB-Ethernet (enx*) even if currently NO-CARRIER — that is the camera dongle.
find_usb_eth() {
  local iface
  for iface in $(ip -o link show | awk -F': ' '{print $2}'); do
    case "$iface" in
      enx*|usb*) echo "$iface"; return 0 ;;
    esac
  done
  return 1
}

# Any wired iface with live carrier (fallback)
find_live_wired() {
  local iface c
  for iface in $(ip -o link show | awk -F': ' '{print $2}'); do
    case "$iface" in
      lo|wl*|ww*|docker*|veth*|br-*|virbr*|tun*|tap*) continue ;;
    esac
    c=$(carrier_of "$iface")
    [ "$c" = "1" ] && { echo "$iface"; return 0; }
  done
  return 1
}

print_phys_checklist() {
  yellow "════════════════════════════════════════════════════════"
  yellow "  carrier=0  →  NO physical Ethernet link"
  yellow "  Software (IP / password / RTSP) cannot fix this."
  yellow "════════════════════════════════════════════════════════"
  cat <<'EOF'

  Checklist (do these now, in order):

  1. Camera POWERED ON
     - PoE injector / 12V PSU plugged in
     - IR / status LED on the camera lit

  2. Ethernet cable
     - One end fully clicked into the CAMERA RJ45
     - Other end fully clicked into the USB-Ethernet DONGLE (not the PC LAN port)
     - Try a different known-good Cat5e/Cat6 cable

  3. USB dongle
     - Plug into a USB 3.0 port (blue) on the PC back panel
     - Avoid hubs / front-panel ports if flaky
     - LED on the dongle should light when cable+camera are good

  4. Do NOT use built-in eno1 unless the cable is actually in that port
     - eno1 also shows NO-CARRIER right now (nothing plugged there)

  5. After re-seating, wait ~5s and re-run this script

EOF
}

bold "═══ Camera link fix ═══"
echo "  Camera IP : $CAM_IP"
echo "  PC IP     : $LOCAL_IP/24  (on USB-Ethernet only)"
echo "  Wait      : ${WAIT_SECS}s for carrier"
echo

# ── Step 1: USB adapter present? ────────────────────────────────────────────
echo "── Step 1: USB-Ethernet adapter ──"
if ! lsusb 2>/dev/null | grep -qiE 'ASIX|0b95:1790|Ethernet'; then
  red "  ✗ No ASIX / USB-Ethernet device in lsusb"
  echo "    Plug the USB-Ethernet dongle into the PC, then re-run."
  lsusb
  exit 1
fi
lsusb | grep -iE 'ASIX|0b95|Ethernet' || true

CAM_IF="$(find_usb_eth || true)"
if [ -z "${CAM_IF:-}" ]; then
  red "  ✗ Dongle in lsusb but no enx* interface yet"
  echo "    Unplug dongle 5s, replug USB3 port, wait 3s, re-run."
  ip link show
  exit 1
fi
green "  ✓ Interface: $CAM_IF"
echo "    driver : $(ethtool -i "$CAM_IF" 2>/dev/null | awk '/driver:/{print $2}')"
echo "    state  : $(cat /sys/class/net/"$CAM_IF"/operstate 2>/dev/null)  carrier=$(carrier_of "$CAM_IF")"
ip link show "$CAM_IF"
echo

# ── Step 2: wait for carrier ────────────────────────────────────────────────
echo "── Step 2: physical link (carrier must become 1) ──"
if [ "$(carrier_of "$CAM_IF")" != "1" ]; then
  print_phys_checklist
  yellow "  Waiting up to ${WAIT_SECS}s for carrier on $CAM_IF …"
  yellow "  (re-seat cable / power camera now)"
  SECS=0
  while [ "$SECS" -lt "$WAIT_SECS" ]; do
    # Interface name can change if dongle is re-plugged — re-detect
    NEW_IF="$(find_usb_eth || true)"
    [ -n "${NEW_IF:-}" ] && CAM_IF="$NEW_IF"
    if [ "$(carrier_of "$CAM_IF")" = "1" ]; then
      green "  ✓ carrier=1 on $CAM_IF after ${SECS}s"
      break
    fi
    # Also accept any other live wired NIC (e.g. cable moved to eno1)
    LIVE="$(find_live_wired || true)"
    if [ -n "${LIVE:-}" ]; then
      CAM_IF="$LIVE"
      green "  ✓ carrier=1 on $CAM_IF after ${SECS}s"
      break
    fi
    printf "\r  … %3ds  %s carrier=%s   " "$SECS" "$CAM_IF" "$(carrier_of "$CAM_IF")"
    sleep 1
    SECS=$((SECS + 1))
  done
  echo
fi

if [ "$(carrier_of "$CAM_IF")" != "1" ]; then
  red "  ✗ Still NO-CARRIER on $CAM_IF after ${WAIT_SECS}s"
  echo
  echo "  What this means:"
  echo "    • Linux sees the USB dongle (good)"
  echo "    • The dongle does NOT see another Ethernet device on the RJ45 side"
  echo "    • Camera off, bad/unplugged cable, or wrong port — not a software bug"
  echo
  echo "  Verify with:"
  echo "    cat /sys/class/net/$CAM_IF/carrier     # must print 1"
  echo "    ethtool $CAM_IF | grep 'Link detected' # must be yes"
  echo
  echo "  Kernel USB log (recent):"
  journalctl -k --since "10 min ago" --no-pager 2>/dev/null \
    | grep -iE 'asix|ax881|cdc_ncm|enxc8|usb.*disconnect|error -110' | tail -15 || true
  exit 1
fi

green "  Link UP: $CAM_IF  (carrier=1)"
ethtool "$CAM_IF" 2>/dev/null | grep -E 'Speed|Duplex|Link detected' || true
echo

# ── Step 3: assign camera-subnet IP (do not touch WiFi default route) ───────
echo "── Step 3: assign $LOCAL_IP/24 on $CAM_IF ──"

# Drop stale copies of this address from other interfaces
for iface in $(ip -o addr show | awk -F': ' "/${LOCAL_IP//./\\.}/{print \$2}" | awk '{print $1}'); do
  if [ "$iface" != "$CAM_IF" ]; then
    yellow "  Removing stale $LOCAL_IP from $iface"
    need_sudo ip addr del "$LOCAL_IP/24" dev "$iface" 2>/dev/null || true
  fi
done

# Prefer NetworkManager so the address survives
if command -v nmcli >/dev/null 2>&1; then
  CON=$(nmcli -t -f NAME,DEVICE connection show --active 2>/dev/null | awk -F: -v d="$CAM_IF" '$2==d{print $1; exit}')
  if [ -z "${CON:-}" ]; then
    # Match existing profile by interface or create one
    CON=$(nmcli -t -f NAME,DEVICE connection show 2>/dev/null | awk -F: -v d="$CAM_IF" '$2==d || $1==d {print $1; exit}')
  fi
  if [ -z "${CON:-}" ]; then
    CON="camera-link"
    need_sudo nmcli connection add type ethernet ifname "$CAM_IF" con-name "$CON" \
      ipv4.method manual ipv4.addresses "$LOCAL_IP/24" ipv4.never-default yes \
      ipv6.method disabled connection.autoconnect yes 2>/dev/null \
      || need_sudo nmcli connection add type ethernet ifname "$CAM_IF" con-name "$CON"
  fi
  need_sudo nmcli connection modify "$CON" \
    connection.interface-name "$CAM_IF" \
    connection.autoconnect yes \
    ipv4.method manual \
    ipv4.addresses "$LOCAL_IP/24" \
    ipv4.gateway "" \
    ipv4.never-default yes \
    ipv4.ignore-auto-dns yes \
    ipv6.method disabled
  need_sudo nmcli connection up "$CON" ifname "$CAM_IF" 2>/dev/null \
    || need_sudo nmcli device connect "$CAM_IF" 2>/dev/null \
    || true
fi

# Always ensure address with iproute2 (works even if NM profile is weird)
need_sudo ip link set "$CAM_IF" up
if ! ip addr show dev "$CAM_IF" | grep -q "$LOCAL_IP"; then
  need_sudo ip addr add "$LOCAL_IP/24" dev "$CAM_IF" 2>/dev/null || true
fi
# Host route to camera (belt and suspenders)
need_sudo ip route replace "$CAM_IP/32" dev "$CAM_IF" 2>/dev/null || true

sleep 1
ip addr show dev "$CAM_IF"
echo "  routes:"
ip route | grep -E "192\.168\.2|$CAM_IF" || true
echo

# ── Step 4: reachability ────────────────────────────────────────────────────
echo "── Step 4: ping camera $CAM_IP ──"
if ping -c 4 -W 2 "$CAM_IP"; then
  green "  ✓ Camera answers ping"
else
  red "  ✗ No ping reply (link is UP — so wrong IP, camera firewall, or different subnet)"
  echo "  Scanning common camera subnets for any host (this takes ~30s)…"
  for subnet in 192.168.2 192.168.1 192.168.0 10.1.1; do
    echo "    $subnet.0/24 …"
    # parallel ping burst
    for i in $(seq 1 254); do
      ping -c 1 -W 1 "$subnet.$i" >/dev/null 2>&1 &
    done
    wait
    FOUND=$(ip neigh show dev "$CAM_IF" 2>/dev/null | grep -v FAILED | grep -v INCOMPLETE || true)
    if [ -n "$FOUND" ]; then
      yellow "  Neighbors on $CAM_IF:"
      echo "$FOUND"
    fi
  done
  echo "  If you see the camera MAC (often starts with a Dahua OUI), use that IP."
  exit 1
fi
echo

# ── Step 5: RTSP ────────────────────────────────────────────────────────────
echo "── Step 5: RTSP port 554 ──"
if nc -z -w 3 "$CAM_IP" 554 2>/dev/null; then
  green "  ✓ TCP 554 open"
else
  red "  ✗ TCP 554 closed — camera up but RTSP not listening / wrong IP"
  exit 1
fi

if command -v ffprobe >/dev/null 2>&1; then
  echo "  Probing stream (one attempt)…"
  if ffprobe -v error -rtsp_transport tcp -timeout 8000000 \
      -i "$RTSP_URL" -show_entries stream=codec_name -of csv=p=0 2>/tmp/ffprobe_cam.err; then
    green "  ✓ RTSP OK — credentials work"
  else
    yellow "  ✗ RTSP open failed (lockout / wrong password / path)"
    cat /tmp/ffprobe_cam.err 2>/dev/null | tail -5
    echo "  URL used: rtsp://${USER}:****@${CAM_IP}:554/cam/realmonitor?channel=1&subtype=1"
    echo "  If lockout: wait 30 min with ZERO attempts, then retry once."
    exit 1
  fi
else
  yellow "  (ffprobe not installed — skip stream test)"
fi

echo
green "═══ Camera link is healthy ═══"
echo "  Interface : $CAM_IF"
echo "  PC        : $LOCAL_IP/24"
echo "  Camera    : $CAM_IP"
echo
echo "Next:"
echo "  ./scripts/start_go2rtc.sh"
echo "  python3 scripts/run_video.py"
echo "  # or: ./scripts/test_camera.sh"
