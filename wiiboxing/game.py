"""Target-practice mode: a ball spawns in your HEAD or BODY zone; land a
punch's peak-extension point inside it to score a hit."""

import time

import cv2
import mediapipe as mp

from . import config
from .classifier import classify_from_summary_left, classify_from_summary_right
from .landmarks import extract
from .pose_utils import classify_height
from .targets import Target, check_hit, spawn_target
from .tracker import ArmMotionTracker

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def run():
    cap = cv2.VideoCapture(0)

    left_tracker = ArmMotionTracker(classify_from_summary_left)
    right_tracker = ArmMotionTracker(classify_from_summary_right)

    left_punch_count = 0
    right_punch_count = 0
    hits = 0
    misses = 0
    target = None
    # Remembers which target was active when each arm's CURRENT punch
    # started, so a late-arriving classification result gets attributed to
    # the target the person was actually aiming at -- not whatever target
    # happens to be on screen by the time classification finishes (which may
    # already be a new one if a different punch landed a live hit meanwhile).
    punch_target_id = {"LEFT": None, "RIGHT": None}

    with mp_pose.Pose(
        min_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            now = time.time()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                fl = extract(results.pose_landmarks)

                left_wrist_px = (fl.left_wrist.x * w, fl.left_wrist.y * h)
                right_wrist_px = (fl.right_wrist.x * w, fl.right_wrist.y * h)

                if target is not None and target.flash_color is not None and now >= target.flash_until:
                    target = None
                if target is None:
                    avoid_radius = max(20, fl.shoulder_width * w * config.TARGET_RADIUS_FACTOR) * 1.3
                    target = spawn_target(
                        fl.midline_x * w, fl.nose.y * h, fl.hip_y * h, fl.shoulder_width * w, w, h,
                        avoid_points=[left_wrist_px, right_wrist_px], avoid_radius=avoid_radius,
                    )

                # Live hit test every frame, independent of punch classification --
                # scoring a hit only needs to know a fist touched the ball, not
                # what type of punch it was, so this doesn't wait for the
                # classifier's motion-window latency. Only test arms MediaPipe
                # is actually confident about this frame (see visibility note
                # on ArmMotionTracker.update below).
                if target is not None and target.flash_color is None:
                    left_hit = fl.left_visible and check_hit(target, *left_wrist_px)
                    right_hit = fl.right_visible and check_hit(target, *right_wrist_px)
                    if left_hit or right_hit:
                        hits += 1
                        target.flash_color = (0, 255, 0)
                        target.flash_until = now + config.TARGET_HIT_FLASH_SECONDS
                        print(f"HIT! Score: {hits}/{hits + misses}")

                # Only feed a real (visible) reading into each arm's tracker.
                # A noisy off-screen/occluded position estimate otherwise
                # gets fed into motion detection as if it were trustworthy,
                # which can spike past the activation margin and look like a
                # punch that never actually happened.
                left_fired = None
                if fl.left_visible:
                    was_active = left_tracker.active
                    left_fired = left_tracker.update(
                        fl.left_frame_data, fl.left_angle, fl.left_elbow_dx_norm, fl.left_dy_norm, now
                    )
                    if not was_active and left_tracker.active:
                        punch_target_id["LEFT"] = target.id if target is not None else None

                right_fired = None
                if fl.right_visible:
                    was_active = right_tracker.active
                    right_fired = right_tracker.update(
                        fl.right_frame_data, fl.right_angle, fl.right_elbow_dx_norm, fl.right_dy_norm, now
                    )
                    if not was_active and right_tracker.active:
                        punch_target_id["RIGHT"] = target.id if target is not None else None

                for fired, wrist, side_name in (
                    (left_fired, fl.left_wrist, "LEFT"),
                    (right_fired, fl.right_wrist, "RIGHT"),
                ):
                    if not fired:
                        continue
                    label, summary = fired
                    if side_name == "LEFT":
                        left_punch_count += 1
                        count = left_punch_count
                    else:
                        right_punch_count += 1
                        count = right_punch_count
                    height = classify_height(wrist, fl.nose, fl.hip_y, fl.shoulder_width, config.HEAD_ZONE_MARGIN)
                    print(f"{side_name} {label} {height} #{count}  "
                          f"(elbow_dx_norm={summary['elbow_dx_norm']:.3f} "
                          f"straightness={summary['straightness']:.3f} "
                          f"dy_min_norm={summary['dy_min_norm']:+.3f})")
                    # Only count as a miss against the target that was
                    # actually active when THIS punch started -- if the
                    # target has since changed (already hit/resolved by
                    # something else), this punch's outcome is stale and
                    # shouldn't be blamed on whatever's on screen now.
                    target_at_punch_start = punch_target_id[side_name]
                    if (target is not None and target.flash_color is None
                            and target.id == target_at_punch_start):
                        misses += 1
                        target.flash_color = (0, 0, 255)
                        target.flash_until = now + config.TARGET_MISS_FLASH_SECONDS
                        print(f"  -> MISS! Score: {hits}/{hits + misses}")

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            left_display = left_tracker.get_display(now)
            right_display = right_tracker.get_display(now)

            if left_display:
                punch_status, status_color = f"LEFT {left_display}!", (0, 255, 0)
            elif right_display:
                punch_status, status_color = f"RIGHT {right_display}!", (0, 0, 255)
            else:
                punch_status, status_color = "GUARD UP", (255, 255, 255)

            cv2.putText(image, f"Status: {punch_status}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
            cv2.putText(image, f"L:{left_punch_count}  R:{right_punch_count}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            cv2.putText(image, f"Score: {hits}/{hits + misses}", (30, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            if target is not None:
                color = target.flash_color if target.flash_color is not None else (255, 255, 255)
                cv2.circle(image, (int(target.x), int(target.y)), int(target.radius), color, 4)
                cv2.circle(image, (int(target.x), int(target.y)), 6, color, -1)

            cv2.imshow("Wii Boxing Target Practice", image)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
