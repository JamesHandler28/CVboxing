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
from .appearance import DEFAULT_APPEARANCE


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


def _dim(color, factor):
    return tuple(int(c * factor) for c in color)


def draw_opponent(image, x_offset, half_w, half_h, lean_offset, fl=None, downed=False, appearance=None):
    """Opponent: head, torso, and (when we have their live pose) actual
    arms swinging with their real punches. Mirrored horizontally -- their
    real-world rightward lean/punch should appear shifted toward your left,
    since you're facing each other.

    `appearance` is the skin/hair/shirt/pants BGR palette sampled once
    during calibration (see appearance.py) -- if omitted, a neutral
    default palette is used instead. Gloves stay a fixed, clearly-readable
    color regardless of appearance, so hands/blocking are always easy to
    track at a glance."""
    appearance = appearance or DEFAULT_APPEARANCE
    skin, hair, shirt = appearance["skin"], appearance["hair"], appearance["shirt"]
    pants = appearance["pants"]

    shift_px = -lean_offset * half_w * 0.25 * config.LEAN_RENDER_SCALE
    glove_color = (60, 60, 230)

    if downed:
        center_x = x_offset + half_w / 2 + shift_px
        head_y, torso_y = half_h * 0.60, half_h * 0.62
        torso_axes = (int(half_w * 0.20), int(half_h * 0.06))
        cv2.ellipse(image, (int(center_x), int(torso_y)), torso_axes, 0, 0, 360, _dim(shirt, 0.55), -1)
        cv2.circle(image, (int(center_x), int(head_y)), int(half_w * 0.09), _dim(skin, 0.55), -1)
        return

    if fl is None:
        # No live pose to work with -- simple blob.
        center_x = x_offset + half_w / 2 + shift_px
        head_r = int(half_w * 0.09)
        torso_axes = (int(half_w * 0.13), int(half_h * 0.16))
        cv2.ellipse(image, (int(center_x), int(half_h * 0.50)), torso_axes, 0, 0, 360, shirt, -1)
        cv2.circle(image, (int(center_x), int(half_h * 0.30)), head_r, skin, -1)
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

    # Torso, then a waistband hint of pants color at the bottom of the
    # torso, then a neck line bridging to the head (keeps the head from
    # reading as a disconnected floating circle when its tracked distance
    # from the shoulders doesn't neatly match the fixed torso proportions
    # below), then arms, then head + hair on top.
    torso_axes = (int(scale * 0.22), int(scale * 0.32))
    torso_center = ((shoulder_mid[0] + hip_pt[0]) // 2, (shoulder_mid[1] + hip_pt[1]) // 2)
    cv2.ellipse(image, torso_center, torso_axes, 0, 0, 360, shirt, -1)
    waist_y = torso_center[1] + int(torso_axes[1] * 0.55)
    cv2.ellipse(image, (torso_center[0], waist_y), (torso_axes[0], max(3, int(torso_axes[1] * 0.3))),
                0, 0, 360, pants, -1)
    cv2.line(image, shoulder_mid, head_pt, shirt, max(6, int(scale * 0.10)))

    for shoulder, elbow, wrist in ((l_shoulder, l_elbow, l_wrist), (r_shoulder, r_elbow, r_wrist)):
        cv2.line(image, shoulder, elbow, shirt, max(4, int(scale * 0.05)))
        cv2.line(image, elbow, wrist, skin, max(4, int(scale * 0.05)))
        cv2.circle(image, wrist, max(7, int(scale * 0.08)), glove_color, -1)
        cv2.circle(image, wrist, max(7, int(scale * 0.08)), (255, 255, 255), 2)

    cv2.circle(image, head_pt, int(scale * 0.16), skin, -1)
    cv2.ellipse(image, (head_pt[0], head_pt[1] - int(scale * 0.06)), (int(scale * 0.17), int(scale * 0.11)),
                0, 180, 360, hair, -1)


def draw_floating_sphere(image, center_x, center_y, radius, base_color, now):
    """Render a recovery target as a glowing floating orb instead of a flat
    ring: a soft pulsing outer glow, a few darkening concentric circles to
    fake sphere shading without a real lighting model, and a glossy
    highlight."""
    pulse = 0.5 + 0.5 * np.sin(now * 3.0)
    glow_r = int(radius * (1.25 + 0.15 * pulse))
    overlay = image.copy()
    cv2.circle(overlay, (center_x, center_y), glow_r, base_color, -1)
    cv2.addWeighted(overlay, 0.25, image, 0.75, 0, dst=image)

    steps = 4
    for i in range(steps, 0, -1):
        r = int(radius * i / steps)
        factor = 0.5 + 0.5 * (i / steps)
        shaded = tuple(int(c * factor) for c in base_color)
        cv2.circle(image, (center_x, center_y), r, shaded, -1)

    hi_r = max(3, int(radius * 0.28))
    hi_offset = int(radius * 0.35)
    cv2.circle(image, (center_x - hi_offset, center_y - hi_offset), hi_r, (255, 255, 255), -1)

    cv2.circle(image, (center_x, center_y), radius, (255, 255, 255), 2)


def draw_impact_flash(image, x_offset, cx, cy, age, duration):
    """An expanding, fading burst at (cx, cy) -- a bright core circle plus a
    few radiating spike lines, comic-panel style, without any text. `age`
    and `duration` are both in seconds; the effect is fully transparent
    once age >= duration (caller is expected to drop it at that point)."""
    t = max(0.0, min(1.0, age / duration))
    alpha = 1.0 - t
    if alpha <= 0:
        return

    base_radius = 18
    radius = int(base_radius + 40 * t)
    center = (x_offset + int(cx), int(cy))

    overlay = image.copy()
    cv2.circle(overlay, center, radius, (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.6 * alpha, image, 1 - 0.6 * alpha, 0, dst=image)

    for angle_deg in range(0, 360, 45):
        rad = np.deg2rad(angle_deg)
        x2 = center[0] + int(radius * 1.4 * np.cos(rad))
        y2 = center[1] + int(radius * 1.4 * np.sin(rad))
        cv2.line(image, center, (x2, y2), (255, 255, 255), max(2, int(4 * alpha)))


def draw_own_hands(image, x_offset, half_w, half_h, left_wrist_norm, right_wrist_norm, appearance=None):
    """Own hands as glove-like shapes, each with a forearm line back to a
    fixed shoulder anchor near the bottom corners, so movement reads as an
    arm swinging rather than a disconnected dot. The forearm line is drawn
    in two segments (shirt-colored near the shoulder, skin-colored near
    the wrist) using the sampled appearance palette, if provided. Gloves
    stay a fixed, clearly-readable color -- and are a bit bigger than a
    bare fist, matching the enlarged blocking hitbox (see
    GLOVE_BLOCK_COVERAGE in config.py)."""
    appearance = appearance or DEFAULT_APPEARANCE
    skin, shirt = appearance["skin"], appearance["shirt"]

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
        mid = ((anchor[0] + x) // 2, (anchor[1] + y) // 2)

        thickness = max(6, int(half_w * 0.02))
        cv2.line(image, anchor, mid, shirt, thickness)
        cv2.line(image, mid, (x, y), skin, thickness)

        glove_r = max(18, int(half_w * 0.058))
        cv2.circle(image, (x, y), glove_r, color, -1)
        cv2.circle(image, (x, y), glove_r, (20, 20, 20), 2)
        # Small highlight for a bit of shading.
        cv2.circle(image, (x - glove_r // 3, y - glove_r // 3), max(3, glove_r // 4), (255, 255, 255), -1)
        # Thumb nub.
        cv2.circle(image, (x + glove_r // 2, y + glove_r // 3), max(4, glove_r // 3), color, -1)