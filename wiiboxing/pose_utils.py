"""Small, dependency-free geometry helpers shared across the package."""

import numpy as np


def calculate_angle_2d(a, b, c):
    """Angle at point b (elbow), between points a (shoulder) and c (wrist).

    Uses only x/y -- MediaPipe's z is a much noisier relative-depth estimate
    and isn't reliable enough to include in this calculation.
    """
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cosine_angle = np.dot(ba, bc) / denom
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def classify_height(wrist, nose, hip_y, shoulder_w, head_zone_margin):
    """Classify which vertical zone a wrist landed in: HEAD, BODY, or LOW.

    Relative to nose (head-level reference) and hip (body-level reference)
    rather than fixed image coordinates, so it holds up regardless of
    distance from the camera.
    """
    if shoulder_w < 1e-6:
        return "BODY"
    margin = shoulder_w * head_zone_margin
    if wrist.y < nose.y + margin:
        return "HEAD"
    if wrist.y < hip_y:
        return "BODY"
    return "LOW"
