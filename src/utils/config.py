from __future__ import annotations

from pathlib import Path
import re
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


def _resolve_path(value: str, config_dir: Path) -> str:
    """Normalize portable config paths, including legacy Windows project paths."""
    stripped = re.sub(r"^[A-Za-z]:[/\\\\]", "", value)
    stripped = re.sub(r"^Face-recognition-ubunto[/\\\\]", "", stripped, flags=re.IGNORECASE)
    stripped = stripped.replace("\\", "/")
    # Settings live in <project>/config; paths such as models/... are project
    # relative, while a truly absolute POSIX path remains untouched.
    candidate = Path(stripped)
    if candidate.is_absolute():
        return str(candidate)
    return str(config_dir.parent / candidate)


def load_settings(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else ROOT / "config" / "settings.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    config_dir = cfg_path.resolve().parent
    # Only declared path-valued fields are resolved: blindly treating every
    # slash-containing value as a path would corrupt RTSP and Redis URLs.
    for key in ("yolo_weights", "yolo_onnx", "tracker_config"):
        if key in data.get("models", {}) and data["models"][key]:
            data["models"][key] = _resolve_path(data["models"][key], config_dir)
    for section, keys in (
        ("models", ("face_root", "person_reid_weights")),
        ("events", ("db_path", "faces_db_path", "snapshots_dir")),
        ("attendance", ("db_path",)),
        ("gallery", ("images_dir",)),
        ("identity_fusion", ("person_reid_weights",)),
    ):
        for key in keys:
            if key in data.get(section, {}) and data[section][key]:
                data[section][key] = _resolve_path(data[section][key], config_dir)
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
