"""Target spawning and hit-testing, shared by game.py (freeplay) and
survival.py (knockdown mode)."""

import numpy as np

from . import config


class Target:
    _next_id = 1

    def __init__(self, x, y, radius):
        self.id = Target._next_id
        Target._next_id += 1
        self.x = x
        self.y = y
        self.radius = radius
        self.flash_color = None
        self.flash_until = 0.0


def spawn_target(midline_x_px, nose_y_px, hip_y_px, shoulder_w_px, w, h, avoid_points, avoid_radius):
    """Spawn a new target in either the HEAD or BODY vertical band, at a
    random horizontal offset from the body midline. All in pixel space.
    Retries a few times if the spot lands too close to any point in
    `avoid_points` (e.g. your current hand positions), so it doesn't spawn
    directly on your guard."""
    for _ in range(20):
        zone = np.random.choice(["HEAD", "BODY"])
        if zone == "HEAD":
            y_min = max(20, nose_y_px - shoulder_w_px * 0.6)
            y_max = nose_y_px + shoulder_w_px * 0.3
        else:
            y_min = nose_y_px + shoulder_w_px * 0.3
            y_max = min(h - 20, hip_y_px)
        if y_max <= y_min:
            y_min, y_max = h * 0.25, h * 0.65
        y = np.random.uniform(y_min, y_max)

        x_spread = shoulder_w_px * config.TARGET_X_SPREAD_FACTOR
        x = np.clip(np.random.uniform(midline_x_px - x_spread, midline_x_px + x_spread), 40, w - 40)

        if all(((x - ax) ** 2 + (y - ay) ** 2) ** 0.5 >= avoid_radius for ax, ay in avoid_points):
            radius = max(20, shoulder_w_px * config.TARGET_RADIUS_FACTOR)
            return Target(x, y, radius)

    # Fell through every retry (rare) -- spawn anyway rather than getting stuck.
    radius = max(20, shoulder_w_px * config.TARGET_RADIUS_FACTOR)
    return Target(x, y, radius)


def check_hit(target, wrist_x_px, wrist_y_px):
    dist = ((wrist_x_px - target.x) ** 2 + (wrist_y_px - target.y) ** 2) ** 0.5
    return dist <= target.radius
