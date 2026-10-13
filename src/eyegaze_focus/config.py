"""Configuration management for eyegaze-focus."""

import os
from pathlib import Path
import tomllib


def get_default_config_path() -> Path:
    """Return standard XDG config file path: ~/.config/eyegaze-focus/config.toml."""
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    base_dir = Path(xdg_config) if xdg_config else Path.home() / ".config"
    return base_dir / "eyegaze-focus" / "config.toml"


def load_config(config_path: Path | None = None) -> dict:
    """Load and parse TOML configuration file if it exists.

    Returns flattened dictionary of configuration key-value pairs.
    """
    path = config_path or get_default_config_path()
    if not path.is_file():
        return {}

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        return {}

    flat = {}
    for section in ("tracking", "geometry", "backend"):
        if section in data and isinstance(data[section], dict):
            for k, v in data[section].items():
                if section == "backend" and k == "name":
                    flat["backend"] = v
                else:
                    flat[k] = v

    if "head_only" in flat and "disable_eye_tracking" not in flat:
        flat["disable_eye_tracking"] = flat["head_only"]

    return flat


def save_config(values: dict, config_path: Path | None = None) -> Path:
    """Generate and write formatted TOML configuration file."""
    path = config_path or get_default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    camera = values.get("camera", 0)
    fps = values.get("fps", 30.0)
    width = values.get("width", 640)
    height = values.get("height", 480)
    dwell = values.get("dwell", 0.15)
    ear_threshold = values.get("ear_threshold", 0.20)
    pitch_down_ignore = values.get("pitch_down_ignore", 25.0)
    disable_eye_tracking = values.get("disable_eye_tracking", False)
    typing_cooldown = values.get("typing_cooldown", 1.0)

    center_span = values.get("center_span", 25.0)
    hysteresis = values.get("hysteresis", 5.0)
    iris_weight = values.get("iris_weight", 20.0)
    center_yaw_offset = values.get("center_yaw_offset", 0.0)

    backend = values.get("backend", "auto")

    content = f"""# eyegaze-focus configuration file
# Location: {path}

[tracking]
# Video capture device index (0 = default webcam)
camera = {camera}
# Frame rate for video capture and inference
fps = {fps:.1f}
# Capture resolution
width = {width}
height = {height}
# Dwell duration required before switching focus (seconds)
dwell = {dwell:.2f}
# Eye Aspect Ratio threshold (frames below this are skipped as blinks)
ear_threshold = {ear_threshold:.2f}
# Ignore gaze focus switches when head tilts downward past this angle (looking at keyboard)
pitch_down_ignore = {pitch_down_ignore:.1f}
# Disable eye/iris tracking and determine focus strictly from head orientation
disable_eye_tracking = {str(disable_eye_tracking).lower()}
# Freeze focus changes for this duration after keyboard typing (seconds, 0.0 to disable)
typing_cooldown = {typing_cooldown:.2f}

[geometry]
# Half-width of center screen deadband in degrees (center spans [-{center_span:.1f}°, +{center_span:.1f}°])
center_span = {center_span:.1f}
# Hysteresis margin in degrees before releasing focus back to center
hysteresis = {hysteresis:.1f}
# Maximum ocular deflection contribution in degrees
iris_weight = {iris_weight:.1f}
# Baseline head yaw offset in degrees to compensate for camera positioning
center_yaw_offset = {center_yaw_offset:.1f}

[backend]
# Window manager display backend: \"auto\", \"sway\", \"i3\", \"yabai\", or \"osascript\"
name = \"{backend}\"
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return path
