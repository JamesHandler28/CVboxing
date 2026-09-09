"""
Windowed feature extraction and punch classification.

Punches are classified using WINDOWED features (peak/min values over the
whole motion), not single-frame checks -- single-frame heuristics couldn't
reliably separate a jab from a hook when this was tried (see
docs/classifier_notes.md for the history). The classify_from_summary_*
functions are decision-tree rules fit from labeled training data; see
config.py for where each threshold came from and how to refit them.
"""

from . import config


def summarize_window(frames):
    """
    frames: list of dicts with keys angle, wrist_x, wrist_y, shoulder_x,
    shoulder_y, elbow_x, elbow_y, midline_x, shoulder_w -- one dict per
    frame of a single punch's motion.

    Returns a dict of normalized summary features, or None if the window is
    too short to trust.
    """
    if len(frames) < 3:
        return None

    shoulder_w_avg = sum(f["shoulder_w"] for f in frames) / len(frames)
    if shoulder_w_avg < 1e-6:
        return None

    angles = [f["angle"] for f in frames]
    peak_angle = max(angles)

    dx_from_shoulder = [abs(f["wrist_x"] - f["shoulder_x"]) for f in frames]
    peak_abs_dx_from_shoulder = max(dx_from_shoulder)

    dy_from_shoulder = [f["wrist_y"] - f["shoulder_y"] for f in frames]
    min_dy_from_shoulder = min(dy_from_shoulder)

    elbow_dx = [abs(f["elbow_x"] - f["shoulder_x"]) for f in frames]
    peak_elbow_dx = max(elbow_dx)

    # Path length: sum of consecutive wrist-to-wrist distances.
    path_length = 0.0
    for i in range(1, len(frames)):
        dx = frames[i]["wrist_x"] - frames[i - 1]["wrist_x"]
        dy = frames[i]["wrist_y"] - frames[i - 1]["wrist_y"]
        path_length += (dx ** 2 + dy ** 2) ** 0.5

    # Net displacement: straight-line distance from the first frame to the
    # frame of peak extension.
    peak_idx = angles.index(peak_angle)
    dx_net = frames[peak_idx]["wrist_x"] - frames[0]["wrist_x"]
    dy_net = frames[peak_idx]["wrist_y"] - frames[0]["wrist_y"]
    net_displacement = (dx_net ** 2 + dy_net ** 2) ** 0.5

    straightness_ratio = (net_displacement / path_length) if path_length > 1e-6 else 0.0

    return {
        "peak_angle": peak_angle,
        "elbow_dx_norm": peak_elbow_dx / shoulder_w_avg,
        "dy_min_norm": min_dy_from_shoulder / shoulder_w_avg,
        "dx_shoulder_norm": peak_abs_dx_from_shoulder / shoulder_w_avg,
        "straightness": straightness_ratio,
        "peak_wrist_x": frames[peak_idx]["wrist_x"],
        "peak_wrist_y": frames[peak_idx]["wrist_y"],
    }


def classify_from_summary_right(s):
    """Decision-tree rule fit from RIGHT-arm rows in punch_training_data.csv."""
    if s["elbow_dx_norm"] > config.HOOK_ELBOW_NORM_THRESHOLD:
        return "HOOK"
    if s["straightness"] > config.JAB_STRAIGHTNESS_THRESHOLD:
        return "JAB"
    if s["dy_min_norm"] <= config.UPPERCUT_DY_NORM_THRESHOLD:
        return "UPPERCUT"
    return "JAB"


def classify_from_summary_left(s):
    """Decision-tree rule fit SEPARATELY from LEFT-arm rows -- do not reuse
    the right-arm thresholds here, they don't transfer (see config.py)."""
    if s["elbow_dx_norm"] <= config.LEFT_JAB_ELBOW_NORM_THRESHOLD:
        return "JAB"
    if s["dy_min_norm"] <= config.LEFT_UPPERCUT_DY_NORM_THRESHOLD:
        return "UPPERCUT"
    if s["peak_angle"] <= config.LEFT_UPPERCUT_ANGLE_THRESHOLD:
        return "UPPERCUT"
    return "HOOK"
