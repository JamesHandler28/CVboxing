"""
Duel mode: menu -> calibration -> fight.

Two-player only: split frame in half (one MediaPipe Pose model per half,
since the classic Pose solution only tracks one person at a time). Each
player sees their own hands and the opponent's silhouette, mirrored so that
leaning your real right shifts your on-screen position left from the
opponent's point of view. Landing a punch (classified the same way as
freeplay mode -- jab/hook/uppercut, HEAD/BODY/LOW zone) damages the
opponent unless they're leaning far enough to dodge a head shot, or their
own glove is covering that zone (blocked). Zero HP knocks a player down;
they get a time-limited window to punch floating recovery targets back up,
with returned HP scaling by performance -- see duel_state.py. A player's
MAX_KNOCKDOWNS'th knockdown is an automatic KO.
"""

import time

import cv2
import mediapipe as mp
import numpy as np

from . import config, sound
from .appearance import DEFAULT_APPEARANCE, sample_appearance
from .calibration import check_position
from .classifier import classify_from_summary_left, classify_from_summary_right
from .duel_state import GAME_OVER, KNOCKDOWN, PLAYING, PlayerState, damage_for_zone, is_dodged
from .landmarks import extract
from .pose_utils import classify_height, glove_covers_zone
from .ring_render import draw_floating_sphere, draw_impact_flash, draw_opponent, draw_own_hands, draw_ring_background
from .targets import check_hit, spawn_target
from .tracker import ArmMotionTracker

mp_pose = mp.solutions.pose

MENU, CALIBRATION, FIGHT = "MENU", "CALIBRATION", "FIGHT"


class PlayerRuntime:
    """Everything about an active (real) player that isn't pure HP/knockdown
    logic: per-arm motion trackers (used while PLAYING), the current
    recovery target (used while KNOCKDOWN), the calibrated neutral lean
    baseline captured before the fight started, and the skin/hair/shirt/
    pants palette sampled from the camera during calibration (see
    appearance.py)."""

    def __init__(self):
        self.trackers = {
            "LEFT": ArmMotionTracker(classify_from_summary_left),
            "RIGHT": ArmMotionTracker(classify_from_summary_right),
        }
        self.recovery_target = None
        self.baseline_midline_x_px = None
        self.calibration_hold_start = None
        self.appearance = dict(DEFAULT_APPEARANCE)


def _lean_offset(fl, half_w, baseline_midline_x_px):
    if baseline_midline_x_px is None or fl.shoulder_width <= 1e-6:
        return 0.0
    current_x_px = fl.midline_x * half_w
    return (current_x_px - baseline_midline_x_px) / (fl.shoulder_width * half_w)


def _is_blocked(defender_fl, zone):
    """A punch aimed at `zone` is blocked if the DEFENDER's own wrist --
    given an enlarged, glove-sized hitbox -- is currently covering it."""
    if defender_fl is None:
        return False
    for wrist in (defender_fl.left_wrist, defender_fl.right_wrist):
        if glove_covers_zone(wrist, defender_fl.nose, defender_fl.hip_y,
                              defender_fl.shoulder_width, zone, config.GLOVE_BLOCK_COVERAGE):
            return True
    return False


def _run_calibration_frame(image_half, half_w, half_h, pose_model, runtime, now):
    """Returns (ok, message, fl_or_None). Updates runtime.calibration_hold_start."""
    rgb = cv2.cvtColor(image_half, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = pose_model.process(rgb)
    rgb.flags.writeable = True

    if not results.pose_landmarks:
        runtime.calibration_hold_start = None
        return False, "Step into frame", None

    fl = extract(results.pose_landmarks)
    visible = fl.left_visible and fl.right_visible
    ok, message = check_position(visible, fl.midline_x * half_w, fl.shoulder_width * half_w, half_w)

    if ok:
        if runtime.calibration_hold_start is None:
            runtime.calibration_hold_start = now
        # Sample colors on every well-positioned frame (not just once) so
        # by the time the hold completes, runtime.appearance reflects a
        # recent, correctly-framed reading rather than a stale early one.
        runtime.appearance = sample_appearance(image_half, fl, half_w, half_h)
    else:
        runtime.calibration_hold_start = None

    return ok, message, fl


def _draw_calibration_hud(image, x_offset, half_w, half_h, ok, message, hold_start, now):
    color = (0, 200, 0) if ok else (0, 0, 255)
    cv2.putText(image, message, (x_offset + 30, int(half_h * 0.5)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    if ok and hold_start is not None:
        remaining = max(0.0, config.CALIBRATION_HOLD_SECONDS - (now - hold_start))
        cv2.putText(image, f"Hold... {remaining:.1f}s", (x_offset + 30, int(half_h * 0.5) + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def _process_fighting_half(image_half, half_w, half_h, pose_model, runtime, x_offset):
    """PLAYING-phase processing: pose + arm trackers. Returns (fl_or_None,
    fired_events) where fired_events is a list of (side_name, zone).
    Drawing (hands/ring) is done by the caller once both halves are known,
    since resolving a punch needs the OPPONENT's data from this same frame."""
    rgb = cv2.cvtColor(image_half, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = pose_model.process(rgb)
    rgb.flags.writeable = True

    if not results.pose_landmarks:
        return None, []

    fl = extract(results.pose_landmarks)
    fired_events = []

    for side_name, visible, angle, elbow_norm, dy_norm, frame_data, wrist in (
        ("LEFT", fl.left_visible, fl.left_angle, fl.left_elbow_dx_norm, fl.left_dy_norm, fl.left_frame_data, fl.left_wrist),
        ("RIGHT", fl.right_visible, fl.right_angle, fl.right_elbow_dx_norm, fl.right_dy_norm, fl.right_frame_data, fl.right_wrist),
    ):
        if not visible:
            continue
        tracker = runtime.trackers[side_name]
        fired = tracker.update(frame_data, angle, elbow_norm, dy_norm, time.time())
        if fired:
            zone = classify_height(wrist, fl.nose, fl.hip_y, fl.shoulder_width, config.HEAD_ZONE_MARGIN)
            fired_events.append((side_name, zone))

    return fl, fired_events


def _process_knockdown_half(image_half, half_w, half_h, pose_model, state, runtime, now):
    """KNOCKDOWN-phase processing: spawn/hit-test a recovery target using
    live wrist position (instant response -- no need to wait for full punch
    classification when all that matters is contact)."""
    rgb = cv2.cvtColor(image_half, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = pose_model.process(rgb)
    rgb.flags.writeable = True

    if not results.pose_landmarks:
        return None

    fl = extract(results.pose_landmarks)
    left_wrist_px = (fl.left_wrist.x * half_w, fl.left_wrist.y * half_h)
    right_wrist_px = (fl.right_wrist.x * half_w, fl.right_wrist.y * half_h)

    target = runtime.recovery_target
    if target is not None and target.flash_color is not None and now >= target.flash_until:
        target = None
    if target is None:
        avoid_radius = max(20, fl.shoulder_width * half_w * config.TARGET_RADIUS_FACTOR) * 1.3
        target = spawn_target(
            fl.midline_x * half_w, fl.nose.y * half_h, fl.hip_y * half_h, fl.shoulder_width * half_w,
            half_w, half_h, avoid_points=[left_wrist_px, right_wrist_px], avoid_radius=avoid_radius,
        )
    runtime.recovery_target = target

    if target.flash_color is None:
        left_hit = fl.left_visible and check_hit(target, *left_wrist_px)
        right_hit = fl.right_visible and check_hit(target, *right_wrist_px)
        if left_hit or right_hit:
            state.register_recovery_hit(now)
            sound.play("recovery_hit")
            target.flash_color = (0, 255, 0)
            target.flash_until = now + config.TARGET_HIT_FLASH_SECONDS

    return fl


def _get_screen_size():
    """Best-effort primary-monitor resolution, for letterboxing the fullscreen
    display so the camera frame's real aspect ratio is preserved instead of
    being stretched to fill the screen unevenly. Returns None if we can't
    determine it (falls back to showing the frame as-is)."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:
        return None


def _letterbox(image, target_w, target_h):
    """Resize `image` to fit within (target_w, target_h) preserving aspect
    ratio, padding the rest with black bars, so it fills the target size
    exactly without distortion."""
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def run():
    cap = cv2.VideoCapture(0)

    window_name = "Wii Boxing Duel"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    fullscreen = True
    screen_size = _get_screen_size()

    def show(display):
        """Show a frame, letterboxed to the real screen size when fullscreen
        and we know that size, so the camera's aspect ratio isn't distorted."""
        if fullscreen and screen_size is not None:
            display = _letterbox(display, *screen_size)
        cv2.imshow(window_name, display)

    def toggle_fullscreen():
        nonlocal fullscreen
        fullscreen = not fullscreen
        cv2.setWindowProperty(
            window_name, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
        )

    app_phase = MENU

    p1_state = PlayerState("P1")
    p2_state = PlayerState("P2")
    p1_runtime = PlayerRuntime()
    p2_runtime = PlayerRuntime()

    effects = []  # active impact-flash dicts: {"x_offset", "cx", "cy", "start"}
    shake_until = 0.0
    shake_magnitude = 0

    pose_p1 = None
    pose_p2 = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        now = time.time()

        if app_phase == MENU:
            display = frame.copy()
            cv2.putText(display, "SPACE: Start 2-Player duel",
                        (30, h // 2 - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(display, "F: toggle fullscreen   Q: quit",
                        (30, h // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            show(display)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
            if key == ord("f"):
                toggle_fullscreen()
            if key == ord(" "):
                pose_p1 = mp_pose.Pose(
                    min_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
                    min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
                )
                pose_p2 = mp_pose.Pose(
                    min_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
                    min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
                )
                app_phase = CALIBRATION
            continue

        mid = w // 2

        if app_phase == CALIBRATION:
            left_half = frame[:, :mid]
            right_half = frame[:, mid:]
            p1_ok, p1_msg, p1_fl = _run_calibration_frame(left_half, mid, h, pose_p1, p1_runtime, now)
            p2_ok, p2_msg, p2_fl = _run_calibration_frame(right_half, w - mid, h, pose_p2, p2_runtime, now)

            display = frame.copy()
            _draw_calibration_hud(display, 0, mid, h, p1_ok, p1_msg, p1_runtime.calibration_hold_start, now)
            _draw_calibration_hud(display, mid, w - mid, h, p2_ok, p2_msg, p2_runtime.calibration_hold_start, now)
            cv2.line(display, (mid, 0), (mid, h), (255, 255, 255), 2)

            p1_ready = (p1_runtime.calibration_hold_start is not None
                        and now - p1_runtime.calibration_hold_start >= config.CALIBRATION_HOLD_SECONDS)
            p2_ready = (p2_runtime.calibration_hold_start is not None
                        and now - p2_runtime.calibration_hold_start >= config.CALIBRATION_HOLD_SECONDS)

            if p1_ready and p2_ready:
                if p1_fl is not None:
                    p1_runtime.baseline_midline_x_px = p1_fl.midline_x * mid
                if p2_fl is not None:
                    p2_runtime.baseline_midline_x_px = p2_fl.midline_x * (w - mid)
                app_phase = FIGHT

            show(display)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
            if key == ord("f"):
                toggle_fullscreen()
            continue

        # ---- FIGHT ----
        p1_before, p2_before = p1_state.phase, p2_state.phase
        p1_state.tick(now)
        p2_state.tick(now)

        match_over = p1_state.phase == GAME_OVER or p2_state.phase == GAME_OVER

        left_half = frame[:, :mid]
        right_half = frame[:, mid:]

        display = frame.copy()

        if not match_over:
            draw_ring_background(display, 0, mid, h)
            draw_ring_background(display, mid, w - mid, h)

            # --- P1 ---
            p1_fl, p1_fired = (None, [])
            if p1_state.phase == PLAYING:
                p1_fl, p1_fired = _process_fighting_half(left_half, mid, h, pose_p1, p1_runtime, 0)
            elif p1_state.phase == KNOCKDOWN:
                p1_fl = _process_knockdown_half(left_half, mid, h, pose_p1, p1_state, p1_runtime, now)

            # --- P2 ---
            p2_fl, p2_fired = (None, [])
            if p2_state.phase == PLAYING:
                p2_fl, p2_fired = _process_fighting_half(right_half, w - mid, h, pose_p2, p2_runtime, mid)
            elif p2_state.phase == KNOCKDOWN:
                p2_fl = _process_knockdown_half(right_half, w - mid, h, pose_p2, p2_state, p2_runtime, now)

            # --- Resolve punches thrown this frame ---
            p1_lean = _lean_offset(p1_fl, mid, p1_runtime.baseline_midline_x_px) if p1_fl else 0.0
            p2_lean = _lean_offset(p2_fl, w - mid, p2_runtime.baseline_midline_x_px) if p2_fl else 0.0

            for side_name, zone in p1_fired:
                if is_dodged(zone, p2_lean):
                    continue
                if _is_blocked(p2_fl, zone):
                    sound.play("block")
                    continue
                p2_state.take_damage(now, damage_for_zone(zone))
                sound.play("punch")
                effects.append({"x_offset": mid, "cx": (w - mid) / 2,
                                 "cy": h * config.IMPACT_POINT_Y_FRAC[zone], "start": now})
                shake_until = max(shake_until, now + config.HIT_SHAKE_DURATION_SECONDS)
                shake_magnitude = max(shake_magnitude, config.HIT_SHAKE_MAGNITUDE_PX[zone])

            for side_name, zone in p2_fired:
                if is_dodged(zone, p1_lean):
                    continue
                if _is_blocked(p1_fl, zone):
                    sound.play("block")
                    continue
                p1_state.take_damage(now, damage_for_zone(zone))
                sound.play("punch")
                effects.append({"x_offset": 0, "cx": mid / 2,
                                 "cy": h * config.IMPACT_POINT_Y_FRAC[zone], "start": now})
                shake_until = max(shake_until, now + config.HIT_SHAKE_DURATION_SECONDS)
                shake_magnitude = max(shake_magnitude, config.HIT_SHAKE_MAGNITUDE_PX[zone])

            # --- Knockdown / KO transition sounds (covers both damage-
            # triggered and count-timeout-triggered transitions) ---
            for before, state in ((p1_before, p1_state), (p2_before, p2_state)):
                if before == PLAYING and state.phase == KNOCKDOWN:
                    sound.play("knockdown")
                elif before != GAME_OVER and state.phase == GAME_OVER:
                    sound.play("ko")
                elif before == KNOCKDOWN and state.phase == PLAYING:
                    sound.play("recovered")

            # --- Draw hands + opponent silhouettes ---
            if p1_fl is not None and p1_state.phase == PLAYING:
                draw_own_hands(display, 0, mid, h, (p1_fl.left_wrist.x, p1_fl.left_wrist.y),
                                (p1_fl.right_wrist.x, p1_fl.right_wrist.y), appearance=p1_runtime.appearance)
            opponent_fl_for_p1 = p2_fl if p2_state.phase == PLAYING else None
            draw_opponent(display, 0, mid, h, p2_lean, fl=opponent_fl_for_p1,
                          downed=p2_state.phase != PLAYING, appearance=p2_runtime.appearance)

            if p2_fl is not None and p2_state.phase == PLAYING:
                draw_own_hands(display, mid, w - mid, h, (p2_fl.left_wrist.x, p2_fl.left_wrist.y),
                                (p2_fl.right_wrist.x, p2_fl.right_wrist.y), appearance=p2_runtime.appearance)
            p1_fl_for_p2 = p1_fl if p1_state.phase == PLAYING else None
            draw_opponent(display, mid, w - mid, h, p1_lean, fl=p1_fl_for_p2, downed=p1_state.phase != PLAYING,
                          appearance=p1_runtime.appearance)
            cv2.line(display, (mid, 0), (mid, h), (255, 255, 255), 2)

            # --- Recovery target overlay + big rising knockdown count ---
            for state, runtime, x_offset, half_width in (
                (p1_state, p1_runtime, 0, mid),
                (p2_state, p2_runtime, mid, w - mid),
            ):
                if state.phase != KNOCKDOWN:
                    continue

                if runtime.recovery_target is not None:
                    t = runtime.recovery_target
                    cx, cy = x_offset + int(t.x), int(t.y)
                    if t.flash_color is not None:
                        cv2.circle(display, (cx, cy), int(t.radius), t.flash_color, -1)
                        cv2.circle(display, (cx, cy), int(t.radius), (255, 255, 255), 3)
                    else:
                        draw_floating_sphere(display, cx, cy, int(t.radius), (60, 140, 230), now)

                elapsed = now - state.knockdown_start_time
                count = min(10, int(elapsed) + 1)
                text = str(count)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 4.0, 8)
                cv2.putText(display, text, (x_offset + half_width // 2 - tw // 2, int(h * 0.30) + th // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 0, 255), 8)

            # --- HUD ---
            def hud_text(state):
                if state.phase == PLAYING:
                    return f"{state.name}: HP {state.hp}/{state.max_hp}", (255, 255, 255)
                if state.phase == KNOCKDOWN:
                    return f"{state.name} DOWN! {state.knockdown_hits}/{state.required_hits} hits", (0, 0, 255)
                return f"{state.name} IS OUT", (0, 0, 255)

            text, color = hud_text(p1_state)
            cv2.putText(display, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            text, color = hud_text(p2_state)
            cv2.putText(display, text, (mid + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # --- Impact flashes ---
            still_active = []
            for fx in effects:
                age = now - fx["start"]
                if age < config.HIT_FLASH_DURATION_SECONDS:
                    draw_impact_flash(display, fx["x_offset"], fx["cx"], fx["cy"], age, config.HIT_FLASH_DURATION_SECONDS)
                    still_active.append(fx)
            effects = still_active

        if match_over:
            winner = "P1" if p2_state.phase == GAME_OVER else "P2"
            cv2.putText(display, f"{winner} WINS! Press R to restart, Q to quit",
                        (30, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        if now < shake_until:
            dx = np.random.randint(-shake_magnitude, shake_magnitude + 1)
            dy = np.random.randint(-shake_magnitude, shake_magnitude + 1)
            shake_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            display = cv2.warpAffine(display, shake_matrix, (display.shape[1], display.shape[0]),
                                      borderMode=cv2.BORDER_REPLICATE)

        show(display)

        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        if key == ord("f"):
            toggle_fullscreen()
        if key == ord("r") and match_over:
            p1_state.reset()
            p2_state.reset()
            p1_runtime = PlayerRuntime()
            p2_runtime = PlayerRuntime()
            effects = []
            shake_until = 0.0
            shake_magnitude = 0
            app_phase = MENU

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()