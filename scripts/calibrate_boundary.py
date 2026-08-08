"""Calibrate the entry/exit boundary: capture a live frame with the current
line drawn so you can see where it sits relative to the real door.

Usage:
    .venv/bin/python3 scripts/calibrate_boundary.py [y1] [y2]
    e.g. .venv/bin/python3 scripts/calibrate_boundary.py 0.80 0.80

Prints the on-screen door position so you can dial in the exact y.
"""
import sys, cv2
sys.path.insert(0, "/home/rahmat/Face-recognition-ubunto")
from src.utils.config import load_settings

cfg = load_settings()
line = dict(cfg["entry_exit"]["line"])
if len(sys.argv) >= 3:
    line["y1"] = float(sys.argv[1])
    line["y2"] = float(sys.argv[2])
if len(sys.argv) >= 5:
    line["x1"] = float(sys.argv[3])
    line["x2"] = float(sys.argv[4])

cap = cv2.VideoCapture(cfg["camera"]["source"])
for _ in range(15):
    ok, f = cap.read()
    if ok and f is not None:
        break
cap.release()
H, W = f.shape[:2]
x1, y1 = int(line["x1"] * W), int(line["y1"] * H)
x2, y2 = int(line["x2"] * W), int(line["y2"] * H)
cv2.line(f, (x1, y1), (x2, y2), (0, 0, 255), 3)
cv2.putText(f, f"y={line['y1']}  x={line['x1']}-{line['x2']}", (x1 + 5, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
out = "/home/rahmat/Desktop/calibration_line.png"
cv2.imwrite(out, f)
print(f"saved {out}  ({W}x{H})  line normalized y={line['y1']}, x={line['x1']}..{line['x2']}")
