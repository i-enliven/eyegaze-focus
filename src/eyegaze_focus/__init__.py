"""eyegaze-focus: Gaze-directed screen focus daemon for Sway, i3, and macOS."""

__version__ = "0.1.0"

from eyegaze_focus.classifier import DwellFilter, ScreenClassifier
from eyegaze_focus.config import get_default_config_path, load_config, save_config
from eyegaze_focus.geometry import (
    calculate_ear,
    compute_gaze_angle_deg,
    compute_gaze_score,
    estimate_head_yaw,
    get_bilateral_iris_ratio,
)
from eyegaze_focus.ipc import (
    DisplayBackend,
    IpcClient,
    OsascriptBackend,
    SwayI3Backend,
    YabaiBackend,
)
from eyegaze_focus.keyboard import KeyboardMonitor

__all__ = [
    "DisplayBackend",
    "DwellFilter",
    "IpcClient",
    "KeyboardMonitor",
    "OsascriptBackend",
    "ScreenClassifier",
    "SwayI3Backend",
    "YabaiBackend",
    "calculate_ear",
    "compute_gaze_angle_deg",
    "compute_gaze_score",
    "estimate_head_yaw",
    "get_bilateral_iris_ratio",
    "get_default_config_path",
    "load_config",
    "save_config",
]
