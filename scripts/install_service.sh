#!/usr/bin/env bash
# install_service.sh — install "attendance" systemd service (auto-starts at boot).
set -e

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_SRC="$PROJ_DIR/scripts/attendance.service"
UNIT_DEST="/etc/systemd/system/attendance.service"

if [ ! -f "$PROJ_DIR/.venv/bin/python3" ]; then
  echo "ERROR: no .venv found at $PROJ_DIR/.venv"
  exit 1
fi

chmod +x "$PROJ_DIR/scripts/run_attendance.sh"

echo "Installing attendance.service for project at: $PROJ_DIR"
sed "s|__PROJECT_DIR__|$PROJ_DIR|g" "$UNIT_SRC" | sudo tee "$UNIT_DEST" > /dev/null

sudo systemctl daemon-reload
sudo systemctl enable attendance.service
sudo systemctl restart attendance.service

echo ""
echo "Service installed and started:"
echo "  sudo systemctl status attendance.service"
echo "  journalctl -u attendance.service -f   # live logs"
echo "  redis-cli XREVRANGE attendance:events + - COUNT 10   # live entry/exit stream"
