"""
Pre-fight calibration: checks a player is visible, at a workable distance
from the camera, and roughly centered in their half of the frame, before
the fight starts. Pure logic (takes plain numbers, not landmark objects) so
it's trivial to test.
"""

from . import config


def check_position(visible, midline_x_px, shoulder_width_px, half_w):
    """
    visible: bool -- is the player's pose confidently detected at all.
    midline_x_px: the player's body midline x-position, in pixels, within
        their half of the frame (0 = left edge of their half).
    shoulder_width_px: the player's shoulder width, in pixels.
    half_w: width of that player's half of the frame, in pixels.

    Returns (ok, message). `message` is a short instruction to display when
    not ok, or a ready message when ok.
    """
    if not visible:
        return False, "Step into frame"

    if half_w <= 0:
        return False, "Step into frame"

    shoulder_frac = shoulder_width_px / half_w
    if shoulder_frac < config.CALIBRATION_SHOULDER_WIDTH_MIN_FRAC:
        return False, "Step closer"
    if shoulder_frac > config.CALIBRATION_SHOULDER_WIDTH_MAX_FRAC:
        return False, "Step back"

    center = half_w / 2
    offset = midline_x_px - center
    tolerance = half_w * config.CALIBRATION_CENTER_TOLERANCE
    if offset > tolerance:
        return False, "Move left"
    if offset < -tolerance:
        return False, "Move right"

    return True, "Ready!"
