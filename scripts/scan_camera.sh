#!/usr/bin/env bash
# Scan common Dahua subnets for the camera MAC (direct connection)
CAMERA_MAC="30:dd:aa:98:7b:f4"

echo "=== Adding secondary IPs to en6 ==="
sudo ifconfig en6 alias 10.1.1.100 netmask 255.255.255.0 2>&1
sudo ifconfig en6 alias 192.168.0.100 netmask 255.255.255.0 2>&1
sudo ifconfig en6 alias 192.168.1.100 netmask 255.255.255.0 2>&1

for subnet in "192.168.1" "10.1.1" "192.168.0"; do
  echo "=== Scanning $subnet.x ==="
  for i in $(seq 1 254); do
    ping -c 1 -t 1 $subnet.$i > /dev/null 2>&1 &
  done
  wait
  sleep 1
  MATCH=$(arp -a | grep -i "$CAMERA_MAC")
  if [ -n "$MATCH" ]; then
    echo ">>> CAMERA FOUND: $MATCH"
  else
    echo "    camera MAC not in ARP on $subnet.x"
  fi
done

echo ""
echo "=== All devices on en6 (via arp -i en6) ==="
arp -a -i en6
