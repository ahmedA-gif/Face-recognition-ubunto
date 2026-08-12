from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.utils.geometry import clean_polygon

ROOT = Path(__file__).resolve().parents[2]

# Aliases used in config/zones.yaml → canonical region names used by the
# Door Intelligence Engine.
_ZONE_ALIASES = {
    "outside": "OUTSIDE",
    "out": "OUTSIDE",
    "door": "DOOR",
    "door_corridor": "DOOR",
    "corridor": "DOOR",
    "inside": "INSIDE",
    "in": "INSIDE",
}


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
        ("models", ("face_root", "person_reid_weights")),
        ("events", ("db_path", "faces_db_path", "snapshots_dir")),
        ("attendance", ("db_path",)),
        ("gallery", ("images_dir",)),
        ("identity_fusion", ("person_reid_weights",)),
    ):
        for key in keys:
            if key in data.get(section, {}) and data[section][key]:
                p = Path(data[section][key])
                if not p.is_absolute():
                    data[section][key] = str(ROOT / p)
    data["_root"] = str(ROOT)
    return data


def load_zones(
    path: str | Path | None = None,
    camera_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Load per-camera region polygons from a zones YAML file.

    File layout (config/zones.yaml)::

        camera_1:
          zones:
            outside: [[x,y], ...]
            door_corridor: [[x,y], ...]
            inside: [[x,y], ...]

    Returns the polygon dict for ``camera_id`` (default: the first camera)
    with keys normalized to the engine's region names OUTSIDE / DOOR / INSIDE.
    Empty dict if the file or camera is missing.
    """
    zones_path = Path(path) if path else ROOT / "config" / "zones.yaml"
    if not zones_path.exists():
        return {}
    with open(zones_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    cameras = data.get("cameras") or data
    if camera_id and camera_id in cameras:
        section = cameras[camera_id]
    else:
        # Fall back to the first camera section with a "zones" key.
        section = None
        for value in cameras.values():
            if isinstance(value, dict) and "zones" in value:
                section = value
                break
    if section is None:
        return {}

    zones: Dict[str, Any] = {}
    for key, polygon in (section.get("zones") or {}).items():
        name = _ZONE_ALIASES.get(str(key).strip().lower())
        if name is None:
            continue
        cleaned = clean_polygon(polygon)
        if cleaned:
            zones[name] = cleaned
    return zones
