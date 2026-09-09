"""
Extracts the per-arm quantities the rest of the package needs from a single
MediaPipe pose result, in one place, so the detector and the game can't
silently drift out of sync with each other.
"""

from dataclasses import dataclass

import mediapipe as mp

from .pose_utils import calculate_angle_2d
from .tracker import ArmMotionTracker
from . import config

mp_pose = mp.solutions.pose


@dataclass
class FrameLandmarks:
    left_shoulder: object
    left_elbow: object
    left_wrist: object
    right_shoulder: object
    right_elbow: object
    right_wrist: object
    nose: object
    hip_y: float
    left_angle: float
    right_angle: float
    midline_x: float
    shoulder_width: float
    left_elbow_dx_norm: float
    left_dy_norm: float
    right_elbow_dx_norm: float
    right_dy_norm: float
    left_frame_data: dict
    right_frame_data: dict
    left_visible: bool
    right_visible: bool


def extract(pose_landmarks):
    """Build a FrameLandmarks from one MediaPipe PoseLandmarkerResult-style
    `pose_landmarks.landmark` list. Assumes the frame was cv2.flip'd
    horizontally before being passed to the pose model (mirror/selfie view),
    which is why MediaPipe's own LEFT/RIGHT labels are swapped back here to
    match the person's actual side."""
    landmarks = pose_landmarks.landmark

    right_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    hip_y = (left_hip.y + right_hip.y) / 2

    left_angle = calculate_angle_2d(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle_2d(right_shoulder, right_elbow, right_wrist)

    midline_x = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_width = abs(right_shoulder.x - left_shoulder.x)

    left_frame_data = ArmMotionTracker.build_frame_data(
        left_angle, left_wrist, left_shoulder, left_elbow, midline_x, shoulder_width
    )
    right_frame_data = ArmMotionTracker.build_frame_data(
        right_angle, right_wrist, right_shoulder, right_elbow, midline_x, shoulder_width
    )

    safe_w = shoulder_width if shoulder_width > 1e-6 else 1.0
    left_elbow_dx_norm = abs(left_elbow.x - left_shoulder.x) / safe_w
    left_dy_norm = (left_wrist.y - left_shoulder.y) / safe_w
    right_elbow_dx_norm = abs(right_elbow.x - right_shoulder.x) / safe_w
    right_dy_norm = (right_wrist.y - right_shoulder.y) / safe_w

    # MediaPipe still reports a position for a joint even when it's barely
    # visible/out of frame -- just a noisy guess. Gate on its own confidence
    # (visibility) rather than trusting the position outright, or a noisy
    # off-screen estimate can spike past the activation threshold and look
    # like a punch that never happened.
    left_visible = min(left_shoulder.visibility, left_elbow.visibility, left_wrist.visibility) >= config.VISIBILITY_THRESHOLD
    right_visible = min(right_shoulder.visibility, right_elbow.visibility, right_wrist.visibility) >= config.VISIBILITY_THRESHOLD

    return FrameLandmarks(
        left_shoulder=left_shoulder, left_elbow=left_elbow, left_wrist=left_wrist,
        right_shoulder=right_shoulder, right_elbow=right_elbow, right_wrist=right_wrist,
        nose=nose, hip_y=hip_y,
        left_angle=left_angle, right_angle=right_angle,
        midline_x=midline_x, shoulder_width=shoulder_width,
        left_elbow_dx_norm=left_elbow_dx_norm, left_dy_norm=left_dy_norm,
        right_elbow_dx_norm=right_elbow_dx_norm, right_dy_norm=right_dy_norm,
        left_frame_data=left_frame_data, right_frame_data=right_frame_data,
        left_visible=left_visible, right_visible=right_visible,
    )
