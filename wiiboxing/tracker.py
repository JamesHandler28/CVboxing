"""Per-arm motion tracking: detects when a punch starts/ends and hands the
buffered window off to the classifier."""

from . import config
from .classifier import summarize_window


class ArmMotionTracker:
    """
    Tracks whether an arm is currently "in motion" (mid-punch) or idle.

    Maintains a live, self-calibrating baseline of the arm's resting
    angle/elbow/dy (an EMA updated only while at rest), and triggers
    "active" based on deviation from THAT baseline rather than a fixed
    number -- see config.py for why fixed thresholds broke during combos.

    While in motion, buffers frames; once motion ends
    (REARM_NEUTRAL_FRAMES of inactivity in a row, or the buffer hits
    MAX_BUFFER_FRAMES), the whole window is summarized and classified in
    one shot.
    """

    def __init__(self, classify_fn):
        self.classify_fn = classify_fn
        self.active = False
        self.buffer = []
        self.neutral_streak = 0
        self.display_label = None
        self.display_until = 0.0

        self.baseline_angle = None
        self.baseline_elbow = None
        self.baseline_dy = None
        self.last_trigger_reason = None

    def _update_baseline(self, angle, elbow_dx_norm, dy_norm):
        if self.baseline_angle is None:
            self.baseline_angle = angle
            self.baseline_elbow = elbow_dx_norm
            self.baseline_dy = dy_norm
        else:
            a = config.BASELINE_EMA_ALPHA
            self.baseline_angle = (1 - a) * self.baseline_angle + a * angle
            self.baseline_elbow = (1 - a) * self.baseline_elbow + a * elbow_dx_norm
            self.baseline_dy = (1 - a) * self.baseline_dy + a * dy_norm

    def _is_active_on(self, angle, elbow_dx_norm, dy_norm):
        if self.baseline_angle is None:
            return False  # no baseline yet, can't judge deviation
        if angle > self.baseline_angle + config.MARGIN_ON_ANGLE:
            self.last_trigger_reason = (
                f"ANGLE {angle:.1f} > baseline {self.baseline_angle:.1f} + {config.MARGIN_ON_ANGLE}"
            )
            return True
        if elbow_dx_norm > self.baseline_elbow + config.MARGIN_ON_ELBOW:
            self.last_trigger_reason = (
                f"ELBOW {elbow_dx_norm:.3f} > baseline {self.baseline_elbow:.3f} + {config.MARGIN_ON_ELBOW}"
            )
            return True
        if dy_norm < self.baseline_dy - config.MARGIN_ON_DY:
            self.last_trigger_reason = (
                f"DY {dy_norm:+.3f} < baseline {self.baseline_dy:+.3f} - {config.MARGIN_ON_DY}"
            )
            return True
        return False

    def _is_active_off(self, angle, elbow_dx_norm, dy_norm):
        if self.baseline_angle is None:
            return False
        return (
            angle > self.baseline_angle + config.MARGIN_OFF_ANGLE
            or elbow_dx_norm > self.baseline_elbow + config.MARGIN_OFF_ELBOW
            or dy_norm < self.baseline_dy - config.MARGIN_OFF_DY
        )

    def update(self, frame_data, angle, elbow_dx_norm, dy_norm, now):
        """Call once per frame. Returns (label, summary) if a punch just
        finished classifying this frame, else None."""
        fired = None

        if not self.active:
            if self._is_active_on(angle, elbow_dx_norm, dy_norm):
                self.active = True
                self.buffer = [frame_data]
                self.neutral_streak = 0
            else:
                # Only learn the resting baseline while confidently idle.
                self._update_baseline(angle, elbow_dx_norm, dy_norm)
        else:
            self.buffer.append(frame_data)
            if self._is_active_off(angle, elbow_dx_norm, dy_norm):
                self.neutral_streak = 0
            else:
                self.neutral_streak += 1

            if self.neutral_streak >= config.REARM_NEUTRAL_FRAMES or len(self.buffer) >= config.MAX_BUFFER_FRAMES:
                summary = summarize_window(self.buffer)
                if summary is not None:
                    label = self.classify_fn(summary)
                    fired = (label, summary)
                    self.display_label = label
                    self.display_until = now + 0.6
                self.active = False
                self.buffer = []
                self.neutral_streak = 0

        return fired

    def get_display(self, now):
        if self.display_label and now < self.display_until:
            return self.display_label
        return None

    @staticmethod
    def build_frame_data(angle, wrist, shoulder, elbow, midline_x, shoulder_w):
        return {
            "angle": angle,
            "wrist_x": wrist.x, "wrist_y": wrist.y,
            "shoulder_x": shoulder.x, "shoulder_y": shoulder.y,
            "elbow_x": elbow.x, "elbow_y": elbow.y,
            "midline_x": midline_x,
            "shoulder_w": shoulder_w,
        }
