"""
Tests for the punch classifier.

The unit tests below check each decision-tree branch in isolation with
synthetic feature vectors. The regression test at the bottom replays the
actual labeled training data (data/punch_training_data.csv) through the
classifier and asserts it still scores at least as well as it did when the
thresholds were fit -- if this test starts failing, something about the
thresholds in config.py has drifted out of sync with the data they were
supposedly fit from.
"""

import csv
import os

import pytest

from wiiboxing.classifier import classify_from_summary_left, classify_from_summary_right, summarize_window

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAINING_CSV = os.path.join(DATA_DIR, "punch_training_data.csv")


# ---------------------------------------------------------------------------
# Right-arm classifier: one case per branch
# ---------------------------------------------------------------------------

def test_right_hook_high_elbow_flare():
    s = {"elbow_dx_norm": 0.70, "straightness": 0.30, "dy_min_norm": 0.0, "peak_angle": 160}
    assert classify_from_summary_right(s) == "HOOK"


def test_right_jab_high_straightness():
    s = {"elbow_dx_norm": 0.20, "straightness": 0.90, "dy_min_norm": 0.0, "peak_angle": 170}
    assert classify_from_summary_right(s) == "JAB"


def test_right_uppercut_low_dy():
    s = {"elbow_dx_norm": 0.30, "straightness": 0.30, "dy_min_norm": -0.80, "peak_angle": 150}
    assert classify_from_summary_right(s) == "UPPERCUT"


def test_right_fallback_is_jab():
    s = {"elbow_dx_norm": 0.30, "straightness": 0.30, "dy_min_norm": 0.50, "peak_angle": 60}
    assert classify_from_summary_right(s) == "JAB"


# ---------------------------------------------------------------------------
# Left-arm classifier: fit separately, do not reuse right-arm thresholds
# ---------------------------------------------------------------------------

def test_left_jab_low_elbow_flare():
    s = {"elbow_dx_norm": 0.25, "dy_min_norm": 0.0, "peak_angle": 50, "straightness": 0.8}
    assert classify_from_summary_left(s) == "JAB"


def test_left_uppercut_low_dy():
    s = {"elbow_dx_norm": 0.50, "dy_min_norm": -1.2, "peak_angle": 160, "straightness": 0.3}
    assert classify_from_summary_left(s) == "UPPERCUT"


def test_left_uppercut_low_angle():
    s = {"elbow_dx_norm": 0.50, "dy_min_norm": -0.5, "peak_angle": 90, "straightness": 0.3}
    assert classify_from_summary_left(s) == "UPPERCUT"


def test_left_hook_high_elbow_high_angle():
    s = {"elbow_dx_norm": 0.70, "dy_min_norm": -0.3, "peak_angle": 160, "straightness": 0.4}
    assert classify_from_summary_left(s) == "HOOK"


# ---------------------------------------------------------------------------
# summarize_window: geometry sanity checks
# ---------------------------------------------------------------------------

def _make_frame(angle, wrist_x, wrist_y, shoulder_x=0.0, shoulder_y=0.0, elbow_x=0.0, elbow_y=0.0, shoulder_w=0.2):
    return {
        "angle": angle,
        "wrist_x": wrist_x, "wrist_y": wrist_y,
        "shoulder_x": shoulder_x, "shoulder_y": shoulder_y,
        "elbow_x": elbow_x, "elbow_y": elbow_y,
        "midline_x": 0.0,
        "shoulder_w": shoulder_w,
    }


def test_summarize_window_straight_line_has_high_straightness():
    # Wrist moves in a straight line from (0,0) to (0.3, 0), angle peaks at the end.
    frames = [_make_frame(angle=10 + i * 20, wrist_x=i * 0.06, wrist_y=0.0) for i in range(6)]
    summary = summarize_window(frames)
    assert summary is not None
    assert summary["straightness"] > 0.95


def test_summarize_window_l_shaped_path_has_lower_straightness():
    # Wrist moves right, then up -- an L-shaped path is less "straight" than
    # a direct line between start and end.
    frames = (
        [_make_frame(angle=10, wrist_x=i * 0.06, wrist_y=0.0) for i in range(4)]
        + [_make_frame(angle=10 + i * 40, wrist_x=0.18, wrist_y=-i * 0.06) for i in range(1, 4)]
    )
    summary = summarize_window(frames)
    assert summary is not None
    assert summary["straightness"] < 0.95


def test_summarize_window_too_short_returns_none():
    frames = [_make_frame(angle=10, wrist_x=0, wrist_y=0), _make_frame(angle=20, wrist_x=0.1, wrist_y=0)]
    assert summarize_window(frames) is None


# ---------------------------------------------------------------------------
# Regression test against the actual labeled training data
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(TRAINING_CSV), reason="training CSV not present in this checkout")
def test_classifier_accuracy_on_training_data():
    rows = list(csv.DictReader(open(TRAINING_CSV)))
    assert rows, "training CSV is empty"

    correct = 0
    for row in rows:
        sw = float(row["shoulder_width"])
        s = {
            "elbow_dx_norm": float(row["peak_elbow_dx_from_shoulder"]) / sw,
            "straightness": float(row["straightness_ratio"]),
            "dy_min_norm": float(row["min_dy_from_shoulder"]) / sw,
            "peak_angle": float(row["peak_angle"]),
        }
        is_left = row["label"].startswith("LEFT_")
        label = row["label"].replace("LEFT_", "").replace("RIGHT_", "")
        pred = classify_from_summary_left(s) if is_left else classify_from_summary_right(s)
        correct += pred == label

    accuracy = correct / len(rows)
    # 23/24 was the accuracy when these thresholds were fit, for both the
    # left and right arm data individually. Allow a little slack for future
    # additions to the CSV, but this should never crater.
    assert accuracy >= 0.85, f"classifier accuracy on training data dropped to {accuracy:.2%}"
