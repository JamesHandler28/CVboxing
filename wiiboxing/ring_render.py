"""
Shape-based rendering for the first-person duel view. Still no sprites or
video compositing (per the confirmed design choice) -- but the opponent now
has real limbs driven by their actual tracked arm positions (so you see
their punches swing, not just a static blob), hands look like gloves
instead of plain dots, and the ring has some actual depth to it.
"""

import cv2
import numpy as np

from . import config


def _vgradient_fill(image, x_offset, y0, y1, half_w, color_top, color_bottom):
    """Fill a horizontal band with a smooth vertical color gradient."""
    y0, y1 = int(y0), int(y1)
    if y1 <= y0:
        return
    height = y1 - y0
    t = np.linspace(0, 1, height).reshape(-1, 1, 1)
    top = np.array(color_top, dtype=np.float32).reshape(1, 1, 3)
    bottom = np.array(color_bottom, dtype=np.float32).reshape(1, 1, 3)
    band = (top * (1 - t) + bottom * t).astype(np.uint8)
    band = np.repeat(band, half_w, axis=1)
    image[y0:y1, x_offset:x_offset + half_w] = band


def draw_ring_background(image, x_offset, half_w, half_h):
    """Arena glow above, canvas floor with rope lines and corner posts."""
    horizon = int(half_h * 0.55)

    _vgradient_fill(image, x_offset, 0, horizon, half_w, (20, 30, 60), (40, 70, 110))
    _vgradient_fill(image, x_offset, horizon, half_h, half_w, (40, 90, 150), (25, 50, 90))

    # Perspective-ish floor: a trapezoid narrower toward the horizon.
    inset = int(half_w * 0.18)
    floor_pts = np.array([
        [x_offset + inset, horizon],
        [x_offset + half_w - inset, horizon],
        [x_offset + half_w, half_h],
        [x_offset, half_h],
    ], dtype=np.int32)
    overlay = image.copy()
    cv2.fillPoly(overlay, [floor_pts], (20, 40, 60))
    cv2.addWeighted(overlay, 0.35, image, 0.65, 0, dst=image)

    # Ropes: three lines with a slight sag, alternating white/red.
    rope_colors = [(240, 240, 240), (60, 60, 220), (240, 240, 240)]
    for i, color in enumerate(rope_colors):
        y = int(horizon - 14 + i * 10)
        sag = 6
        pts = np.array([
            [x_offset, y],
            [x_offset + half_w // 2, y + sag],
            [x_offset + half_w, y],
        ], dtype=np.int32)
        cv2.polylines(image, [pts], False, color, 3)

    # Corner posts at each edge of this half.
    post_w = max(6, int(half_w * 0.015))
    post_top = horizon - 40
    for post_x in (x_offset + 4, x_offset + half_w - post_w - 4):
        cv2.rectangle(image, (post_x, post_top), (post_x + post_w, half_h), (30, 30, 200), -1)
        cv2.rectangle(image, (post_x, post_top), (post_x + post_w, post_top + 16), (240, 240, 240), -1)


def _anchor_and_scale(fl, half_w, half_h, shift_px):
    anchor_x = half_w / 2 + shift_px
    anchor_y = half_h * 0.38
    scale = half_w * 0.32
    return anchor_x, anchor_y, scale


def _rel(landmark, ref_x, ref_y, shoulder_w):
    return (landmark.x - ref_x) / shoulder_w, (landmark.y - ref_y) / shoulder_w


def draw_opponent(image, x_offset, half_w, half_h, lean_offset, fl=None, downed=False):
    """Opponent: head, torso, and (when we have their live pose) actual
    arms swinging with their real punches. Mirrored horizontally -- their
    real-world rightward lean/punch should appear shifted toward your left,
    since you're facing each other."""
    shift_px = -lean_offset * half_w * 0.25 * config.LEAN_RENDER_SCALE
    color = (100, 100, 220) if downed else (225, 225, 225)
    glove_color = (60, 60, 230)

    if downed:
        center_x = x_offset + half_w / 2 + shift_px
        head_y, torso_y = half_h * 0.60, half_h * 0.62
        torso_axes = (int(half_w * 0.20), int(half_h * 0.06))
        cv2.ellipse(image, (int(center_x), int(torso_y)), torso_axes, 0, 0, 360, color, -1)
        cv2.circle(image, (int(center_x), int(head_y)), int(half_w * 0.09), color, -1)
        return

    if fl is None:
        # No live pose to work with (e.g. a static dummy) -- simple blob.
        center_x = x_offset + half_w / 2 + shift_px
        head_r = int(half_w * 0.09)
        torso_axes = (int(half_w * 0.13), int(half_h * 0.16))
        cv2.ellipse(image, (int(center_x), int(half_h * 0.50)), torso_axes, 0, 0, 360, color, -1)
        cv2.circle(image, (int(center_x), int(half_h * 0.30)), head_r, color, -1)
        return

    anchor_x, anchor_y, scale = _anchor_and_scale(fl, half_w, half_h, shift_px)
    sw = fl.shoulder_width if fl.shoulder_width > 1e-6 else 1.0

    def to_screen(landmark):
        dx, dy = _rel(landmark, fl.midline_x, (fl.left_shoulder.y + fl.right_shoulder.y) / 2, sw)
        dx = max(-2.2, min(2.2, dx))
        dy = max(-1.2, min(2.0, dy))
        # Mirrored horizontally (facing each other), same vertical sense.
        sx = x_offset + anchor_x - dx * scale
        sy = anchor_y + dy * scale
        return int(sx), int(sy)

    head_pt = to_screen(fl.nose)
    l_shoulder, r_shoulder = to_screen(fl.left_shoulder), to_screen(fl.right_shoulder)
    l_elbow, r_elbow = to_screen(fl.left_elbow), to_screen(fl.right_elbow)
    l_wrist, r_wrist = to_screen(fl.left_wrist), to_screen(fl.right_wrist)
    shoulder_mid = ((l_shoulder[0] + r_shoulder[0]) // 2, (l_shoulder[1] + r_shoulder[1]) // 2)
    hip_pt = (shoulder_mid[0], shoulder_mid[1] + int(scale * 0.7))

    # Torso, then a neck line bridging to the head (keeps the head from
    # reading as a disconnected floating circle when its tracked distance
    # from the shoulders doesn't neatly match the fixed torso proportions
    # below), then arms, then head on top.
    torso_axes = (int(scale * 0.22), int(scale * 0.32))
    torso_center = ((shoulder_mid[0] + hip_pt[0]) // 2, (shoulder_mid[1] + hip_pt[1]) // 2)
    cv2.ellipse(image, torso_center, torso_axes, 0, 0, 360, color, -1)
    cv2.line(image, shoulder_mid, head_pt, color, max(6, int(scale * 0.10)))

    for shoulder, elbow, wrist in ((l_shoulder, l_elbow, l_wrist), (r_shoulder, r_elbow, r_wrist)):
        cv2.line(image, shoulder, elbow, color, max(4, int(scale * 0.05)))
        cv2.line(image, elbow, wrist, color, max(4, int(scale * 0.05)))
        cv2.circle(image, wrist, max(6, int(scale * 0.07)), glove_color, -1)
        cv2.circle(image, wrist, max(6, int(scale * 0.07)), (255, 255, 255), 2)

    cv2.circle(image, head_pt, int(scale * 0.16), color, -1)


def draw_own_hands(image, x_offset, half_w, half_h, left_wrist_norm, right_wrist_norm):
    """Own hands as glove-like shapes, each with a forearm line back to a
    fixed shoulder anchor near the bottom corners, so movement reads as
    an arm swinging rather than a disconnected dot."""
    anchors = {
        "left": (x_offset + int(half_w * 0.20), int(half_h * 0.95)),
        "right": (x_offset + int(half_w * 0.80), int(half_h * 0.95)),
    }
    gloves = {
        "left": ((60, 60, 230), left_wrist_norm),
        "right": ((60, 230, 60), right_wrist_norm),
    }
    for side, (color, wrist_norm) in gloves.items():
        x = x_offset + int(wrist_norm[0] * half_w)
        y = int(wrist_norm[1] * half_h)
        anchor = anchors[side]

        cv2.line(image, anchor, (x, y), (230, 210, 190), max(6, int(half_w * 0.02)))

        glove_r = max(16, int(half_w * 0.05))
        cv2.circle(image, (x, y), glove_r, color, -1)
        cv2.circle(image, (x, y), glove_r, (20, 20, 20), 2)
        # Small highlight for a bit of shading.
        cv2.circle(image, (x - glove_r // 3, y - glove_r // 3), max(3, glove_r // 4), (255, 255, 255), -1)
        # Thumb nub.
        cv2.circle(image, (x + glove_r // 2, y + glove_r // 3), max(4, glove_r // 3), color, -1)
