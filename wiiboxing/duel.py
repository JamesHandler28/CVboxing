"""
Duel mode: menu -> calibration -> fight.

1-player: solo practice against a static dummy (full-frame view, no HP/
knockdown for you -- just a hit counter).

2-player: split frame in half (one MediaPipe Pose model per half, since the
classic Pose solution only tracks one person at a time). Each player sees
their own hands and the opponent's silhouette, mirrored so that leaning
your real right shifts your on-screen position left from the opponent's
point of view. Landing a punch (classified the same way as freeplay mode --
jab/hook/uppercut, HEAD/BODY/LOW zone) damages the opponent unless they're
leaning far enough to dodge a head shot. Zero HP knocks a player down; they
get a time-limited window to punch a floating recovery target back up
(reusing the same target logic as freeplay mode), at the cost of lower max
HP each time. Failing a recovery window loses the match.
"""

import time

import cv2
import mediapipe as mp
import numpy as np

from . import config
from .calibration import check_position
from .classifier import classify_from_summary_left, classify_from_summary_right
from .duel_state import GAME_OVER, KNOCKDOWN, PLAYING, PlayerState, damage_for_zone, is_dodged
from .landmarks import extract
from .pose_utils import classify_height
from .ring_render import draw_opponent, draw_own_hands, draw_ring_background
from .targets import check_hit, spawn_target
from .tracker import ArmMotionTracker

mp_pose = mp.solutions.pose

MENU, CALIBRATION, FIGHT = "MENU", "CALIBRATION", "FIGHT"


class PlayerRuntime:
    """Everything about an active (real) player that isn't pure HP/knockdown
    logic: per-arm motion trackers (used while PLAYING), the current
    recovery target (used while KNOCKDOWN), and the calibrated neutral lean
    baseline captured before the fight started."""

    def __init__(self):
        self.trackers = {
            "LEFT": ArmMotionTracker(classify_from_summary_left),
            "RIGHT": ArmMotionTracker(classify_from_summary_right),
        }
        self.recovery_target = None
        self.baseline_midline_x_px = None
        self.calibration_hold_start = None


class DummyOpponent:
    """The stationary practice-dummy opponent for 1-player mode. Never
    dodges, never attacks back, no HP -- just tallies what connected."""

    def __init__(self):
        self.hits_by_zone = {"HEAD": 0, "BODY": 0, "LOW": 0}
        self.lean_offset = 0.0

    def take_hit(self, zone):
        self.hits_by_zone[zone] += 1

    @property
    def total_hits(self):
        return sum(self.hits_by_zone.values())


def _lean_offset(fl, half_w, baseline_midline_x_px):
    if baseline_midline_x_px is None or fl.shoulder_width <= 1e-6:
        return 0.0
    current_x_px = fl.midline_x * half_w
    return (current_x_px - baseline_midline_x_px) / (fl.shoulder_width * half_w)


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
    num_players = None  # 1 or 2
    preview_mode = False

    p1_state = PlayerState("P1")
    p2_state = PlayerState("P2")
    p1_runtime = PlayerRuntime()
    p2_runtime = PlayerRuntime()
    dummy = DummyOpponent()

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
            cv2.putText(display, "1: Solo practice   2: 2-Player duel   3: Mirror preview (solo)",
                        (30, h // 2 - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(display, "F: toggle fullscreen   Q: quit",
                        (30, h // 2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            show(display)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
            if key == ord("f"):
                toggle_fullscreen()
            if key in (ord("1"), ord("2"), ord("3")):
                preview_mode = key == ord("3")
                num_players = 2 if key == ord("2") else 1
                pose_p1 = mp_pose.Pose(
                    min_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
                    min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
                )
                pose_p2 = mp_pose.Pose(
                    min_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
                    min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
                ) if num_players == 2 else None
                app_phase = CALIBRATION
            continue

        mid = w // 2 if (num_players == 2 or preview_mode) else w

        if app_phase == CALIBRATION:
            left_half = frame[:, :mid]
            p1_ok, p1_msg, p1_fl = _run_calibration_frame(left_half, mid, h, pose_p1, p1_runtime, now)

            if num_players == 2:
                right_half = frame[:, mid:]
                p2_ok, p2_msg, p2_fl = _run_calibration_frame(right_half, w - mid, h, pose_p2, p2_runtime, now)
            else:
                p2_ok, p2_msg, p2_fl = True, "Ready!", None
                p2_runtime.calibration_hold_start = now  # dummy: always "held"

            display = frame.copy()
            _draw_calibration_hud(display, 0, mid, h, p1_ok, p1_msg, p1_runtime.calibration_hold_start, now)
            if num_players == 2:
                _draw_calibration_hud(display, mid, w - mid, h, p2_ok, p2_msg, p2_runtime.calibration_hold_start, now)
                cv2.line(display, (mid, 0), (mid, h), (255, 255, 255), 2)

            p1_ready = (p1_runtime.calibration_hold_start is not None
                        and now - p1_runtime.calibration_hold_start >= config.CALIBRATION_HOLD_SECONDS)
            p2_ready = num_players == 1 or (
                p2_runtime.calibration_hold_start is not None
                and now - p2_runtime.calibration_hold_start >= config.CALIBRATION_HOLD_SECONDS
            )

            if p1_ready and p2_ready:
                if p1_fl is not None:
                    p1_runtime.baseline_midline_x_px = p1_fl.midline_x * mid
                if num_players == 2 and p2_fl is not None:
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
        if preview_mode:
            left_half = frame[:, :mid]
            display = frame.copy()
            draw_ring_background(display, 0, mid, h)
            draw_ring_background(display, mid, w - mid, h)

            p1_fl, _fired = _process_fighting_half(left_half, mid, h, pose_p1, p1_runtime, 0)

            if p1_fl is not None:
                draw_own_hands(display, 0, mid, h, (p1_fl.left_wrist.x, p1_fl.left_wrist.y),
                                (p1_fl.right_wrist.x, p1_fl.right_wrist.y))
                p1_lean = _lean_offset(p1_fl, mid, p1_runtime.baseline_midline_x_px)
                # This is the key thing being previewed: the SAME live pose
                # data, rendered through draw_opponent's mirroring, is
                # exactly what a real P2 would see of you.
                draw_opponent(display, mid, w - mid, h, p1_lean, fl=p1_fl, downed=False)

            cv2.line(display, (mid, 0), (mid, h), (255, 255, 255), 2)
            cv2.putText(display, "YOU", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(display, "WHAT YOUR OPPONENT WOULD SEE", (mid + 20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(display, "Lean/punch to test mirroring -- M: menu   Q: quit",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            show(display)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                break
            if key == ord("f"):
                toggle_fullscreen()
            if key == ord("m"):
                p1_runtime = PlayerRuntime()
                app_phase = MENU
                num_players = None
                preview_mode = False
            continue

        p1_state.tick(now)
        if num_players == 2:
            p2_state.tick(now)

        match_over = p1_state.phase == GAME_OVER or (num_players == 2 and p2_state.phase == GAME_OVER)

        left_half = frame[:, :mid]
        right_half = frame[:, mid:] if num_players == 2 else None

        display = frame.copy()

        if not match_over:
            draw_ring_background(display, 0, mid, h)
            if num_players == 2:
                draw_ring_background(display, mid, w - mid, h)

            # --- P1 ---
            p1_fl, p1_fired = (None, [])
            if p1_state.phase == PLAYING:
                p1_fl, p1_fired = _process_fighting_half(left_half, mid, h, pose_p1, p1_runtime, 0)
            elif p1_state.phase == KNOCKDOWN:
                p1_fl = _process_knockdown_half(left_half, mid, h, pose_p1, p1_state, p1_runtime, now)

            # --- P2 / dummy ---
            p2_fl, p2_fired = (None, [])
            if num_players == 2:
                if p2_state.phase == PLAYING:
                    p2_fl, p2_fired = _process_fighting_half(right_half, w - mid, h, pose_p2, p2_runtime, mid)
                elif p2_state.phase == KNOCKDOWN:
                    p2_fl = _process_knockdown_half(right_half, w - mid, h, pose_p2, p2_state, p2_runtime, now)

            # --- Resolve punches thrown this frame ---
            p1_lean = _lean_offset(p1_fl, mid, p1_runtime.baseline_midline_x_px) if p1_fl else 0.0
            p2_lean = (_lean_offset(p2_fl, w - mid, p2_runtime.baseline_midline_x_px)
                       if (num_players == 2 and p2_fl) else 0.0)

            for side_name, zone in p1_fired:
                if num_players == 2:
                    if not is_dodged(zone, p2_lean):
                        p2_state.take_damage(now, damage_for_zone(zone))
                else:
                    dummy.take_hit(zone)

            if num_players == 2:
                for side_name, zone in p2_fired:
                    if not is_dodged(zone, p1_lean):
                        p1_state.take_damage(now, damage_for_zone(zone))

            # --- Draw hands + opponent silhouettes ---
            if p1_fl is not None and p1_state.phase == PLAYING:
                draw_own_hands(display, 0, mid, h, (p1_fl.left_wrist.x, p1_fl.left_wrist.y),
                                (p1_fl.right_wrist.x, p1_fl.right_wrist.y))
            opponent_lean_for_p1 = p2_lean if num_players == 2 else dummy.lean_offset
            opponent_downed_for_p1 = num_players == 2 and p2_state.phase != PLAYING
            opponent_fl_for_p1 = p2_fl if (num_players == 2 and p2_state.phase == PLAYING) else None
            draw_opponent(display, 0, mid, h, opponent_lean_for_p1, fl=opponent_fl_for_p1,
                          downed=opponent_downed_for_p1)

            if num_players == 2:
                if p2_fl is not None and p2_state.phase == PLAYING:
                    draw_own_hands(display, mid, w - mid, h, (p2_fl.left_wrist.x, p2_fl.left_wrist.y),
                                    (p2_fl.right_wrist.x, p2_fl.right_wrist.y))
                p1_fl_for_p2 = p1_fl if p1_state.phase == PLAYING else None
                draw_opponent(display, mid, w - mid, h, p1_lean, fl=p1_fl_for_p2, downed=p1_state.phase != PLAYING)
                cv2.line(display, (mid, 0), (mid, h), (255, 255, 255), 2)

            # --- Recovery target overlay ---
            for state, runtime, x_offset, half_width in (
                (p1_state, p1_runtime, 0, mid),
                *([(p2_state, p2_runtime, mid, w - mid)] if num_players == 2 else []),
            ):
                if state.phase == KNOCKDOWN and runtime.recovery_target is not None:
                    t = runtime.recovery_target
                    color = t.flash_color if t.flash_color is not None else (255, 255, 255)
                    cv2.circle(display, (x_offset + int(t.x), int(t.y)), int(t.radius), color, 4)
                    cv2.circle(display, (x_offset + int(t.x), int(t.y)), 6, color, -1)

            # --- HUD ---
            def hud_text(state):
                if state.phase == PLAYING:
                    return f"{state.name}: HP {state.hp}/{state.max_hp}", (255, 255, 255)
                if state.phase == KNOCKDOWN:
                    remaining = max(0.0, config.KNOCKDOWN_TIME_LIMIT_SECONDS - (now - state.knockdown_start_time))
                    return f"{state.name} DOWN! {state.knockdown_hits}/{state.required_hits} -- {remaining:.1f}s", (0, 0, 255)
                return f"{state.name} IS OUT", (0, 0, 255)

            if num_players == 2:
                text, color = hud_text(p1_state)
                cv2.putText(display, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                text, color = hud_text(p2_state)
                cv2.putText(display, text, (mid + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                cv2.putText(display, f"Hits -- HEAD:{dummy.hits_by_zone['HEAD']} "
                                      f"BODY:{dummy.hits_by_zone['BODY']} LOW:{dummy.hits_by_zone['LOW']}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if match_over:
            winner = "P1" if (num_players == 1 or p2_state.phase == GAME_OVER) else "P2"
            cv2.putText(display, f"{winner} WINS! Press R to restart, Q to quit",
                        (30, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

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
            dummy = DummyOpponent()
            app_phase = MENU
            num_players = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()