"""
Samples a small approximate color palette (skin, hair, shirt, pants) from
the live camera image so each player's rendered silhouette loosely matches
what they're actually wearing -- while keeping the existing shape-based
rendering style (no photorealistic compositing, no sprites) and always
keeping boxing gloves on the hands regardless of what's sampled.

Sampling happens ONCE, during calibration (see duel.py) -- not every
frame -- so a player's colors stay stable through the fight instead of
drifting with lighting flicker or a hand briefly passing in front of their
shirt.
"""

import numpy as np

# BGR fallbacks (OpenCV color order), used per-key whenever that region
# can't be sampled -- e.g. pants are frequently out of frame in a
# waist-up boxing camera view.
DEFAULT_APPEARANCE = {
    "skin": (150, 190, 220),
    "hair": (40, 40, 40),
    "shirt": (180, 120, 60),
    "pants": (90, 70, 60),
}


def _patch_average(image, cx, cy, w, h, patch_size=10):
    """Average BGR color in a small square patch centered at (cx, cy),
    clamped to stay inside the image. Returns None if the patch would be
    degenerate (e.g. way outside the frame)."""
    half = patch_size // 2
    x = int(np.clip(cx, half, w - half - 1))
    y = int(np.clip(cy, half, h - half - 1))
    patch = image[y - half:y + half, x - half:x + half]
    if patch.size == 0:
        return None
    mean = patch.reshape(-1, 3).mean(axis=0)
    return tuple(int(c) for c in mean)


def sample_appearance(image_half, fl, half_w, half_h):
    """Returns a dict of BGR tuples for skin/hair/shirt/pants, sampled from
    `image_half` at points derived from `fl` (that frame's FrameLandmarks).
    Falls back to DEFAULT_APPEARANCE per-key if a region can't be sampled."""
    result = dict(DEFAULT_APPEARANCE)

    shoulder_mid_x = (fl.left_shoulder.x + fl.right_shoulder.x) / 2 * half_w
    shoulder_mid_y = (fl.left_shoulder.y + fl.right_shoulder.y) / 2 * half_h
    nose_x, nose_y = fl.nose.x * half_w, fl.nose.y * half_h
    hip_y_px = fl.hip_y * half_h
    sw_px = max(10, fl.shoulder_width * half_w)

    # Skin: a patch on the cheek, just beside and below the nose.
    skin = _patch_average(image_half, nose_x + sw_px * 0.35, nose_y + sw_px * 0.25, half_w, half_h)
    if skin:
        result["skin"] = skin

    # Hair: a patch above the top of the head (extrapolated above the nose;
    # MediaPipe doesn't give us a head-top landmark directly).
    hair_y = nose_y - sw_px * 0.9
    if hair_y > 4:
        hair = _patch_average(image_half, nose_x, hair_y, half_w, half_h)
        if hair:
            result["hair"] = hair

    # Shirt: torso midpoint between the shoulder line and the hip line.
    shirt_y = (shoulder_mid_y + hip_y_px) / 2
    shirt = _patch_average(image_half, shoulder_mid_x, shirt_y, half_w, half_h, patch_size=16)
    if shirt:
        result["shirt"] = shirt

    # Pants: below the hip line -- only if that's actually still inside the
    # frame with some margin. A waist-up boxing framing often won't have
    # this, in which case the fallback color is used instead.
    pants_y = hip_y_px + sw_px * 0.9
    if pants_y < half_h - 12:
        pants = _patch_average(image_half, shoulder_mid_x, pants_y, half_w, half_h, patch_size=16)
        if pants:
            result["pants"] = pants

    return result