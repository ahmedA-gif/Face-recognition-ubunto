from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "config" / "settings.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # Resolve relative paths against project root
    for key in ("yolo_weights", "yolo_onnx", "tracker_config"):
        if key in data.get("models", {}):
            p = Path(data["models"][key])
            if not p.is_absolute():
                data["models"][key] = str(ROOT / p)
    for section, keys in (
        ("models", ("face_root",)),
        ("events", ("db_path", "faces_db_path", "snapshots_dir")),
        ("attendance", ("db_path",)),
        ("gallery", ("images_dir",)),
    ):
        for key in keys:
            if key in data.get(section, {}):
                p = Path(data[section][key])
                if not p.is_absolute():
                    data[section][key] = str(ROOT / p)
    data["_root"] = str(ROOT)
    return data
