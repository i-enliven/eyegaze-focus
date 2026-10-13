"""Unit tests for geometric and head pose calculations."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from eyegaze_focus.geometry import (
    FACE_3D_MODEL,
    calculate_ear,
    compute_gaze_angle_deg,
    estimate_head_yaw,
    get_bilateral_iris_ratio,
)


def _make_mock_landmarks(coords_dict: dict[int, tuple[float, float]]) -> dict[int, SimpleNamespace]:
    """Create a dictionary of mock landmarks from (x, y) coordinates."""
    res = {}
    for i in range(500):
        res[i] = SimpleNamespace(x=0.5, y=0.5, z=0.0)
    for idx, (x, y) in coords_dict.items():
        res[idx] = SimpleNamespace(x=x, y=y, z=0.0)
    return res


def test_calculate_ear_open_eyes():
    w, h = 640.0, 480.0
    coords = {
        # Right eye: outer=33, inner=133
        33: (0.3, 0.5),
        133: (0.4, 0.5),
        160: (0.33, 0.48),
        158: (0.37, 0.48),
        144: (0.33, 0.52),
        153: (0.37, 0.52),
        # Left eye: inner=362, outer=263
        362: (0.6, 0.5),
        263: (0.7, 0.5),
        385: (0.63, 0.48),
        387: (0.67, 0.48),
        380: (0.63, 0.52),
        373: (0.67, 0.52),
    }
    landmarks = _make_mock_landmarks(coords)
    r_ear, l_ear, avg_ear = calculate_ear(landmarks, w, h)

    assert r_ear > 0.25
    assert l_ear > 0.25
    assert avg_ear > 0.25


def test_calculate_ear_closed_eyes():
    w, h = 640.0, 480.0
    coords = {
        33: (0.3, 0.5),
        133: (0.4, 0.5),
        160: (0.33, 0.5),
        158: (0.37, 0.5),
        144: (0.33, 0.5),
        153: (0.37, 0.5),
        362: (0.6, 0.5),
        263: (0.7, 0.5),
        385: (0.63, 0.5),
        387: (0.67, 0.5),
        380: (0.63, 0.5),
        373: (0.67, 0.5),
    }
    landmarks = _make_mock_landmarks(coords)
    r_ear, l_ear, avg_ear = calculate_ear(landmarks, w, h)

    assert r_ear == pytest.approx(0.0, abs=1e-5)
    assert l_ear == pytest.approx(0.0, abs=1e-5)
    assert avg_ear == pytest.approx(0.0, abs=1e-5)


def test_bilateral_iris_ratio_centered():
    w, h = 640.0, 480.0
    coords = {
        33: (0.3, 0.5),
        133: (0.4, 0.5),
        469: (0.35, 0.5),
        470: (0.35, 0.5),
        471: (0.35, 0.5),
        472: (0.35, 0.5),
        362: (0.6, 0.5),
        263: (0.7, 0.5),
        474: (0.65, 0.5),
        475: (0.65, 0.5),
        476: (0.65, 0.5),
        477: (0.65, 0.5),
    }
    landmarks = _make_mock_landmarks(coords)
    ratio = get_bilateral_iris_ratio(landmarks, w, h)
    assert ratio == pytest.approx(0.5, abs=0.01)


def test_bilateral_iris_ratio_looking_left():
    w, h = 640.0, 480.0
    coords = {
        33: (0.3, 0.5),
        133: (0.4, 0.5),
        469: (0.38, 0.5),
        470: (0.38, 0.5),
        471: (0.38, 0.5),
        472: (0.38, 0.5),
        362: (0.6, 0.5),
        263: (0.7, 0.5),
        474: (0.68, 0.5),
        475: (0.68, 0.5),
        476: (0.68, 0.5),
        477: (0.68, 0.5),
    }
    landmarks = _make_mock_landmarks(coords)
    ratio = get_bilateral_iris_ratio(landmarks, w, h)
    assert ratio > 0.65


def test_estimate_head_yaw_synthetic():
    w, h = 640.0, 480.0
    cam_matrix = np.array([[w, 0.0, w / 2], [0.0, w, h / 2], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # 1. Straight ahead
    rvec_straight = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
    tvec = np.array([[0.0], [0.0], [1000.0]], dtype=np.float64)
    pts_straight, _ = cv2.projectPoints(FACE_3D_MODEL, rvec_straight, tvec, cam_matrix, dist_coeffs)

    landmark_map = {
        1: (pts_straight[0, 0, 0] / w, pts_straight[0, 0, 1] / h),
        199: (pts_straight[1, 0, 0] / w, pts_straight[1, 0, 1] / h),
        33: (pts_straight[2, 0, 0] / w, pts_straight[2, 0, 1] / h),
        263: (pts_straight[3, 0, 0] / w, pts_straight[3, 0, 1] / h),
        61: (pts_straight[4, 0, 0] / w, pts_straight[4, 0, 1] / h),
        291: (pts_straight[5, 0, 0] / w, pts_straight[5, 0, 1] / h),
    }
    landmarks = _make_mock_landmarks(landmark_map)
    yaw_center = estimate_head_yaw(landmarks, w, h)
    assert yaw_center == pytest.approx(0.0, abs=0.5)

    # 2. Turned to user's Left (+15 deg)
    rvec_left = np.array([[0.0], [np.radians(-15.0)], [0.0]], dtype=np.float64)
    pts_left, _ = cv2.projectPoints(FACE_3D_MODEL, rvec_left, tvec, cam_matrix, dist_coeffs)
    landmark_map_left = {
        1: (pts_left[0, 0, 0] / w, pts_left[0, 0, 1] / h),
        199: (pts_left[1, 0, 0] / w, pts_left[1, 0, 1] / h),
        33: (pts_left[2, 0, 0] / w, pts_left[2, 0, 1] / h),
        263: (pts_left[3, 0, 0] / w, pts_left[3, 0, 1] / h),
        61: (pts_left[4, 0, 0] / w, pts_left[4, 0, 1] / h),
        291: (pts_left[5, 0, 0] / w, pts_left[5, 0, 1] / h),
    }
    landmarks_left = _make_mock_landmarks(landmark_map_left)
    yaw_left = estimate_head_yaw(landmarks_left, w, h)
    assert yaw_left == pytest.approx(15.0, abs=1.0)


def test_compute_gaze_angle_deg():
    # Center: head 0°, iris 0.50 -> 0°
    assert compute_gaze_angle_deg(0.50, 0.0) == pytest.approx(0.0, abs=0.1)

    # Looking Left: head +20°, iris 0.60 -> 20 + (0.1 * 20) = +22°
    left_angle = compute_gaze_angle_deg(0.60, 20.0, iris_weight_deg=20.0)
    assert left_angle == pytest.approx(22.0, abs=0.1)

    # Looking Right: head -20°, iris 0.40 -> -20 + (-0.1 * 20) = -22°
    right_angle = compute_gaze_angle_deg(0.40, -20.0, iris_weight_deg=20.0)
    assert right_angle == pytest.approx(-22.0, abs=0.1)

    # Baseline offset compensation
    offset_angle = compute_gaze_angle_deg(0.50, -5.0, yaw_offset_deg=-5.0)
    assert offset_angle == pytest.approx(0.0, abs=0.1)
