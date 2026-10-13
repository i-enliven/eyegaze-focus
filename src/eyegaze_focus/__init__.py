"""eyegaze-focus: Gaze-directed screen focus daemon for Sway/i3."""

__version__ = "0.1.0"

from eyegaze_focus.classifier import DwellFilter, ScreenClassifier
from eyegaze_focus.geometry import (
    calculate_ear,
    compute_gaze_score,
    estimate_head_yaw,
    get_bilateral_iris_ratio,
)
from eyegaze_focus.ipc import IpcClient

__all__ = [
    "DwellFilter",
    "IpcClient",
    "ScreenClassifier",
    "calculate_ear",
    "compute_gaze_score",
    "estimate_head_yaw",
    "get_bilateral_iris_ratio",
]
