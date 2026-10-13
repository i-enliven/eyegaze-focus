"""Geometry and computer vision calculations for gaze and head pose estimation."""

import cv2
import numpy as np

# MediaPipe FaceMesh indices
# Right eye (person's right, viewer's left)
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
RIGHT_EYE_TOP_1 = 160
RIGHT_EYE_TOP_2 = 158
RIGHT_EYE_BOT_1 = 144
RIGHT_EYE_BOT_2 = 153
RIGHT_IRIS = [469, 470, 471, 472]

# Left eye (person's left, viewer's right)
LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263
LEFT_EYE_TOP_1 = 385
LEFT_EYE_TOP_2 = 387
LEFT_EYE_BOT_1 = 380
LEFT_EYE_BOT_2 = 373
LEFT_IRIS = [474, 475, 476, 477]

# 3D facial model for Head Pose Estimation (solvePnP)
FACE_3D_MODEL = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose tip (landmark 1)
        (0.0, -330.0, -65.0),     # Chin (landmark 199)
        (-225.0, 170.0, -135.0),  # Right eye corner (landmark 33)
        (225.0, 170.0, -135.0),   # Left eye corner (landmark 263)
        (-150.0, -150.0, -125.0), # Right mouth corner (landmark 61)
        (150.0, -150.0, -125.0),  # Left mouth corner (landmark 291)
    ],
    dtype=np.float64,
)
HEAD_POSE_LANDMARKS = [1, 199, 33, 263, 61, 291]


def _landmark_to_xy(landmark, w: float, h: float) -> np.ndarray:
    """Convert a normalized MediaPipe landmark to pixel coordinates."""
    return np.array([landmark.x * w, landmark.y * h], dtype=np.float64)


def calculate_ear(landmarks, w: float, h: float) -> tuple[float, float, float]:
    """Calculate Eye Aspect Ratio (EAR) for right eye, left eye, and average.

    Returns:
        (right_ear, left_ear, avg_ear)
    """
    # Right eye
    r_outer = _landmark_to_xy(landmarks[RIGHT_EYE_OUTER], w, h)
    r_inner = _landmark_to_xy(landmarks[RIGHT_EYE_INNER], w, h)
    r_top1 = _landmark_to_xy(landmarks[RIGHT_EYE_TOP_1], w, h)
    r_top2 = _landmark_to_xy(landmarks[RIGHT_EYE_TOP_2], w, h)
    r_bot1 = _landmark_to_xy(landmarks[RIGHT_EYE_BOT_1], w, h)
    r_bot2 = _landmark_to_xy(landmarks[RIGHT_EYE_BOT_2], w, h)

    r_width = np.linalg.norm(r_inner - r_outer)
    if r_width > 0:
        r_ear = (np.linalg.norm(r_top1 - r_bot1) + np.linalg.norm(r_top2 - r_bot2)) / (2.0 * r_width)
    else:
        r_ear = 0.0

    # Left eye
    l_inner = _landmark_to_xy(landmarks[LEFT_EYE_INNER], w, h)
    l_outer = _landmark_to_xy(landmarks[LEFT_EYE_OUTER], w, h)
    l_top1 = _landmark_to_xy(landmarks[LEFT_EYE_TOP_1], w, h)
    l_top2 = _landmark_to_xy(landmarks[LEFT_EYE_TOP_2], w, h)
    l_bot1 = _landmark_to_xy(landmarks[LEFT_EYE_BOT_1], w, h)
    l_bot2 = _landmark_to_xy(landmarks[LEFT_EYE_BOT_2], w, h)

    l_width = np.linalg.norm(l_outer - l_inner)
    if l_width > 0:
        l_ear = (np.linalg.norm(l_top1 - l_bot1) + np.linalg.norm(l_top2 - l_bot2)) / (2.0 * l_width)
    else:
        l_ear = 0.0

    avg_ear = (r_ear + l_ear) / 2.0
    return float(r_ear), float(l_ear), float(avg_ear)


def get_bilateral_iris_ratio(landmarks, w: float, h: float) -> float:
    """Calculate normalized horizontal iris ratio averaged across both eyes.

    0.0 = Far Right gaze (user looking towards their right / Screen 2)
    0.5 = Center gaze
    1.0 = Far Left gaze (user looking towards their left / Screen 0)
    """
    # Right eye: outer (33) to inner (133) along horizontal X axis
    r_outer = _landmark_to_xy(landmarks[RIGHT_EYE_OUTER], w, h)
    r_inner = _landmark_to_xy(landmarks[RIGHT_EYE_INNER], w, h)
    r_iris_pts = np.array([_landmark_to_xy(landmarks[idx], w, h) for idx in RIGHT_IRIS])
    r_iris_center = np.mean(r_iris_pts, axis=0)

    r_span = r_inner[0] - r_outer[0]
    if abs(r_span) > 1e-3:
        r_ratio = (r_iris_center[0] - r_outer[0]) / r_span
    else:
        r_ratio = 0.5

    # Left eye: inner (362) to outer (263) along horizontal X axis
    l_inner = _landmark_to_xy(landmarks[LEFT_EYE_INNER], w, h)
    l_outer = _landmark_to_xy(landmarks[LEFT_EYE_OUTER], w, h)
    l_iris_pts = np.array([_landmark_to_xy(landmarks[idx], w, h) for idx in LEFT_IRIS])
    l_iris_center = np.mean(l_iris_pts, axis=0)

    l_span = l_outer[0] - l_inner[0]
    if abs(l_span) > 1e-3:
        l_ratio = (l_iris_center[0] - l_inner[0]) / l_span
    else:
        l_ratio = 0.5

    # Clamp individual ratios to reasonable ocular range [0.0, 1.0]
    r_clamped = float(np.clip(r_ratio, 0.0, 1.0))
    l_clamped = float(np.clip(l_ratio, 0.0, 1.0))

    return (r_clamped + l_clamped) / 2.0


def estimate_head_yaw(landmarks, w: float, h: float) -> float:
    """Estimate head yaw rotation angle in degrees using Perspective-n-Point.

    Positive yaw = Head turned to user's Left (+deg)
    Negative yaw = Head turned to user's Right (-deg)
    0.0 = Head centered
    """
    image_points = np.array(
        [[landmarks[idx].x * w, landmarks[idx].y * h] for idx in HEAD_POSE_LANDMARKS],
        dtype=np.float64,
    )

    focal_length = float(w)
    center = (w / 2.0, h / 2.0)
    cam_matrix = np.array(
        [[focal_length, 0.0, center[0]], [0.0, focal_length, center[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, _ = cv2.solvePnP(
        FACE_3D_MODEL,
        image_points,
        cam_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0

    rot_matrix, _ = cv2.Rodrigues(rvec)
    # Yaw angle around vertical Y axis
    yaw_rad = np.arctan2(rot_matrix[0, 2], rot_matrix[2, 2])
    return float(np.degrees(yaw_rad))


def compute_gaze_score(
    iris_ratio: float,
    head_yaw_deg: float,
    yaw_weight: float = 0.015,
) -> float:
    """Fuse bilateral iris ratio and head yaw into a unified horizontal score.

    Score interpretation:
      < 0.42: Gaze directed to Right screen (Screen 2)
      0.42 - 0.58: Gaze directed to Center screen (Screen 1)
      > 0.58: Gaze directed to Left screen (Screen 0)
    """
    composite = iris_ratio + (yaw_weight * head_yaw_deg)
    return float(np.clip(composite, 0.0, 1.0))
