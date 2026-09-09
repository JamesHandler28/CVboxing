"""Tests for check_position: visibility, distance, and centering checks."""

from wiiboxing import config
from wiiboxing.calibration import check_position

HALF_W = 640.0


def _good_shoulder_width():
    mid = (config.CALIBRATION_SHOULDER_WIDTH_MIN_FRAC + config.CALIBRATION_SHOULDER_WIDTH_MAX_FRAC) / 2
    return HALF_W * mid


def test_not_visible_fails():
    ok, msg = check_position(False, HALF_W / 2, _good_shoulder_width(), HALF_W)
    assert not ok
    assert "frame" in msg.lower()


def test_perfectly_centered_and_good_distance_is_ready():
    ok, msg = check_position(True, HALF_W / 2, _good_shoulder_width(), HALF_W)
    assert ok
    assert msg == "Ready!"


def test_too_far_away_says_step_closer():
    tiny_shoulder = HALF_W * (config.CALIBRATION_SHOULDER_WIDTH_MIN_FRAC - 0.02)
    ok, msg = check_position(True, HALF_W / 2, tiny_shoulder, HALF_W)
    assert not ok
    assert msg == "Step closer"


def test_too_close_says_step_back():
    huge_shoulder = HALF_W * (config.CALIBRATION_SHOULDER_WIDTH_MAX_FRAC + 0.05)
    ok, msg = check_position(True, HALF_W / 2, huge_shoulder, HALF_W)
    assert not ok
    assert msg == "Step back"


def test_off_center_to_the_right_says_move_left():
    off_center_x = HALF_W / 2 + HALF_W * (config.CALIBRATION_CENTER_TOLERANCE + 0.05)
    ok, msg = check_position(True, off_center_x, _good_shoulder_width(), HALF_W)
    assert not ok
    assert msg == "Move left"


def test_off_center_to_the_left_says_move_right():
    off_center_x = HALF_W / 2 - HALF_W * (config.CALIBRATION_CENTER_TOLERANCE + 0.05)
    ok, msg = check_position(True, off_center_x, _good_shoulder_width(), HALF_W)
    assert not ok
    assert msg == "Move right"


def test_zero_width_half_frame_fails_safely():
    ok, msg = check_position(True, 0.0, 10.0, 0.0)
    assert not ok
