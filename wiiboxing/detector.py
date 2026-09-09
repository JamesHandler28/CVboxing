"""Console-only punch detector: prints each detected punch (side, type,
height zone) with its feature values. No game overlay -- see game.py for
the target-practice mode."""

import time

import cv2
import mediapipe as mp

from . import config
from .classifier import classify_from_summary_left, classify_from_summary_right
from .landmarks import extract
from .pose_utils import classify_height
from .tracker import ArmMotionTracker

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def run():
    cap = cv2.VideoCapture(0)

    left_tracker = ArmMotionTracker(classify_from_summary_left)
    right_tracker = ArmMotionTracker(classify_from_summary_right)

    left_punch_count = 0
    right_punch_count = 0

    with mp_pose.Pose(
        min_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            now = time.time()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                fl = extract(results.pose_landmarks)

                left_fired = None
                if fl.left_visible:
                    left_fired = left_tracker.update(
                        fl.left_frame_data, fl.left_angle, fl.left_elbow_dx_norm, fl.left_dy_norm, now
                    )
                right_fired = None
                if fl.right_visible:
                    right_fired = right_tracker.update(
                        fl.right_frame_data, fl.right_angle, fl.right_elbow_dx_norm, fl.right_dy_norm, now
                    )

                if left_fired:
                    label, summary = left_fired
                    left_punch_count += 1
                    height = classify_height(fl.left_wrist, fl.nose, fl.hip_y, fl.shoulder_width, config.HEAD_ZONE_MARGIN)
                    print(f"LEFT {label} {height} #{left_punch_count}  "
                          f"(elbow_dx_norm={summary['elbow_dx_norm']:.3f} "
                          f"straightness={summary['straightness']:.3f} "
                          f"dy_min_norm={summary['dy_min_norm']:+.3f})")
                if right_fired:
                    label, summary = right_fired
                    right_punch_count += 1
                    height = classify_height(fl.right_wrist, fl.nose, fl.hip_y, fl.shoulder_width, config.HEAD_ZONE_MARGIN)
                    print(f"RIGHT {label} {height} #{right_punch_count}  "
                          f"(elbow_dx_norm={summary['elbow_dx_norm']:.3f} "
                          f"straightness={summary['straightness']:.3f} "
                          f"dy_min_norm={summary['dy_min_norm']:+.3f})")

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

            cv2.imshow("Wii Boxing Vision", image)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
