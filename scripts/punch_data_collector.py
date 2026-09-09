import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# ---------------------------------------------------------------------------
# How this works:
#   Press 1/2/3 to start a batch of RIGHT JAB/HOOK/UPPERCUT.
#   Press 4/5/6 to start a batch of LEFT  JAB/HOOK/UPPERCUT.
#   Press 7 to start a batch of RIGHT-then-LEFT COMBO throws (for debugging
#   why combos aren't detecting well) -- this records a full raw per-frame
#   trace for BOTH arms simultaneously, instead of a single summary, so we
#   can see exactly what each arm is doing throughout the combo.
#   Once started, the script runs entirely on its own: a countdown, then
#   "THROW NOW", then a rest gap, repeated REPS_PER_BATCH times. You never
#   need to touch the keyboard mid-punch.
# ---------------------------------------------------------------------------

REPS_PER_BATCH = 8
COUNTDOWN_SECONDS = 1.5
CAPTURE_WINDOW_SECONDS = 0.7
COMBO_CAPTURE_WINDOW_SECONDS = 1.6   # longer, since two punches need to fit
REST_SECONDS = 1.2
CSV_PATH = "punch_training_data.csv"
COMBO_CSV_PATH = "combo_raw_data.csv"

COMBO_CSV_COLUMNS = [
    "rep", "frame", "t",
    "right_angle", "right_elbow_dx_norm", "right_dy_norm",
    "left_angle", "left_elbow_dx_norm", "left_dy_norm",
    "shoulder_width",
]

CSV_COLUMNS = [
    "label",
    "peak_angle",              # most extended the elbow got
    "min_angle",                # most bent the elbow got (during the window)
    "peak_dx_from_midline",     # signed; how far past midline the wrist reached
    "peak_abs_dx_from_shoulder",# how far sideways from own shoulder, absolute
    "min_dy_from_shoulder",     # most negative = highest above shoulder (uppercut signal)
    "max_dy_from_shoulder",     # most positive = lowest below shoulder
    "peak_elbow_dx_from_shoulder",  # how far the elbow flares out sideways
    "peak_elbow_dy_from_shoulder",  # how much the elbow rises/drops
    "path_length",              # total wrist travel distance during the window
    "net_displacement",         # straight-line distance from start to peak-extension point
    "straightness_ratio",       # net_displacement / path_length; 1.0 = perfectly straight line, lower = curved/arcing
    "shoulder_width",            # for reference/normalization sanity checks
]


def calculate_angle_2d(a, b, c):
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return 0.0
    cosine_angle = np.dot(ba, bc) / denom
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def ensure_csv_header():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
    if not os.path.exists(COMBO_CSV_PATH):
        with open(COMBO_CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(COMBO_CSV_COLUMNS)


def summarize_capture(frames, shoulder_w_avg):
    """
    frames: list of dicts, each with keys:
        angle, wrist_x, wrist_y, shoulder_x, shoulder_y,
        elbow_x, elbow_y, midline_x
    Returns a dict of summary features, or None if not enough data.
    """
    if len(frames) < 3:
        return None

    angles = [f["angle"] for f in frames]
    peak_angle = max(angles)
    min_angle = min(angles)

    dx_from_mid = [f["wrist_x"] - f["midline_x"] for f in frames]
    # signed peak (largest magnitude, keeping sign)
    peak_dx_from_midline = max(dx_from_mid, key=abs)

    dx_from_shoulder = [abs(f["wrist_x"] - f["shoulder_x"]) for f in frames]
    peak_abs_dx_from_shoulder = max(dx_from_shoulder)

    dy_from_shoulder = [f["wrist_y"] - f["shoulder_y"] for f in frames]
    min_dy_from_shoulder = min(dy_from_shoulder)  # most negative = highest
    max_dy_from_shoulder = max(dy_from_shoulder)  # most positive = lowest

    elbow_dx = [abs(f["elbow_x"] - f["shoulder_x"]) for f in frames]
    peak_elbow_dx = max(elbow_dx)
    elbow_dy = [f["elbow_y"] - f["shoulder_y"] for f in frames]
    # peak flare: whichever direction moved most from shoulder height
    peak_elbow_dy = max(elbow_dy, key=abs)

    # Path length: sum of consecutive wrist-to-wrist distances
    path_length = 0.0
    for i in range(1, len(frames)):
        dx = frames[i]["wrist_x"] - frames[i - 1]["wrist_x"]
        dy = frames[i]["wrist_y"] - frames[i - 1]["wrist_y"]
        path_length += (dx ** 2 + dy ** 2) ** 0.5

    # Net displacement: from first frame to the frame of peak extension
    peak_idx = angles.index(peak_angle)
    dx_net = frames[peak_idx]["wrist_x"] - frames[0]["wrist_x"]
    dy_net = frames[peak_idx]["wrist_y"] - frames[0]["wrist_y"]
    net_displacement = (dx_net ** 2 + dy_net ** 2) ** 0.5

    straightness_ratio = (net_displacement / path_length) if path_length > 1e-6 else 0.0

    return {
        "peak_angle": round(peak_angle, 1),
        "min_angle": round(min_angle, 1),
        "peak_dx_from_midline": round(peak_dx_from_midline, 4),
        "peak_abs_dx_from_shoulder": round(peak_abs_dx_from_shoulder, 4),
        "min_dy_from_shoulder": round(min_dy_from_shoulder, 4),
        "max_dy_from_shoulder": round(max_dy_from_shoulder, 4),
        "peak_elbow_dx_from_shoulder": round(peak_elbow_dx, 4),
        "peak_elbow_dy_from_shoulder": round(peak_elbow_dy, 4),
        "path_length": round(path_length, 4),
        "net_displacement": round(net_displacement, 4),
        "straightness_ratio": round(straightness_ratio, 4),
        "shoulder_width": round(shoulder_w_avg, 4),
    }


def main():
    ensure_csv_header()
    cap = cv2.VideoCapture(0)

    # Batch state machine phases: "idle", "countdown", "capture", "rest"
    phase = "idle"
    batch_label = None
    batch_reps_done = 0
    phase_start_time = 0.0

    record_frames = []
    record_shoulder_widths = []
    combo_rep_index = 0
    combo_frame_rows = []

    saved_count = 0

    print("Ready. Press 1/2/3=RIGHT JAB/HOOK/UPPERCUT batch, 4/5/6=LEFT JAB/HOOK/UPPERCUT batch, 7=RIGHT-LEFT COMBO batch.")
    print(f"Each batch runs {REPS_PER_BATCH} reps automatically -- just throw when it says THROW NOW.")
    print(f"Single-punch data saves to {CSV_PATH}. Combo raw traces save to {COMBO_CSV_PATH}. Press q to quit.")

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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

            status_text = "Press 1-3=RIGHT, 4-6=LEFT, 7=COMBO to start a batch"
            status_color = (255, 255, 255)
            sub_text = ""

            right_landmarks_ok = False
            left_landmarks_ok = False

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Same left/right swap as before: frame is mirrored, so we
                # correct MediaPipe's own labels to match the person's actual side.
                right_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                right_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                left_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                left_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

                right_angle = calculate_angle_2d(right_shoulder, right_elbow, right_wrist)
                left_angle = calculate_angle_2d(left_shoulder, left_elbow, left_wrist)
                midline_x = (left_shoulder.x + right_shoulder.x) / 2
                shoulder_width = abs(right_shoulder.x - left_shoulder.x)
                right_landmarks_ok = True
                left_landmarks_ok = True

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # ---- Batch state machine ----
            if phase == "countdown":
                elapsed = now - phase_start_time
                remaining = COUNTDOWN_SECONDS - elapsed
                status_text = f"{batch_label} #{batch_reps_done + 1}/{REPS_PER_BATCH}: GET READY"
                sub_text = f"{max(remaining, 0):.1f}s"
                status_color = (0, 255, 255)
                if remaining <= 0:
                    phase = "capture"
                    phase_start_time = now
                    record_frames = []
                    record_shoulder_widths = []
                    combo_frame_rows = []

            elif phase == "capture":
                is_combo = batch_label == "COMBO_RIGHT_LEFT"
                window = COMBO_CAPTURE_WINDOW_SECONDS if is_combo else CAPTURE_WINDOW_SECONDS
                elapsed = now - phase_start_time
                remaining = window - elapsed
                status_text = f"{batch_label} #{batch_reps_done + 1}/{REPS_PER_BATCH}: THROW NOW"
                status_color = (0, 0, 255)

                if is_combo:
                    if right_landmarks_ok and left_landmarks_ok:
                        r_elbow_dx_norm = abs(right_elbow.x - right_shoulder.x) / shoulder_width if shoulder_width > 1e-6 else 0
                        r_dy_norm = (right_wrist.y - right_shoulder.y) / shoulder_width if shoulder_width > 1e-6 else 0
                        l_elbow_dx_norm = abs(left_elbow.x - left_shoulder.x) / shoulder_width if shoulder_width > 1e-6 else 0
                        l_dy_norm = (left_wrist.y - left_shoulder.y) / shoulder_width if shoulder_width > 1e-6 else 0
                        combo_frame_rows.append([
                            combo_rep_index, len(combo_frame_rows), round(elapsed, 4),
                            round(right_angle, 1), round(r_elbow_dx_norm, 4), round(r_dy_norm, 4),
                            round(left_angle, 1), round(l_elbow_dx_norm, 4), round(l_dy_norm, 4),
                            round(shoulder_width, 4),
                        ])
                else:
                    is_left = batch_label.startswith("LEFT")
                    landmarks_ok = left_landmarks_ok if is_left else right_landmarks_ok
                    if landmarks_ok:
                        if is_left:
                            angle, wrist, shoulder, elbow = left_angle, left_wrist, left_shoulder, left_elbow
                        else:
                            angle, wrist, shoulder, elbow = right_angle, right_wrist, right_shoulder, right_elbow
                        record_frames.append({
                            "angle": angle,
                            "wrist_x": wrist.x,
                            "wrist_y": wrist.y,
                            "shoulder_x": shoulder.x,
                            "shoulder_y": shoulder.y,
                            "elbow_x": elbow.x,
                            "elbow_y": elbow.y,
                            "midline_x": midline_x,
                        })
                        record_shoulder_widths.append(shoulder_width)

                if remaining <= 0:
                    if is_combo:
                        if combo_frame_rows:
                            with open(COMBO_CSV_PATH, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerows(combo_frame_rows)
                            saved_count += 1
                            print(f"Saved combo rep #{combo_rep_index} ({len(combo_frame_rows)} frames) to {COMBO_CSV_PATH}")
                        else:
                            print("No pose detected during combo capture, discarded.")
                        combo_rep_index += 1
                        combo_frame_rows = []
                    elif record_frames:
                        shoulder_w_avg = sum(record_shoulder_widths) / len(record_shoulder_widths)
                        summary = summarize_capture(record_frames, shoulder_w_avg)
                        if summary:
                            with open(CSV_PATH, "a", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow([batch_label] + [summary[c] for c in CSV_COLUMNS[1:]])
                            saved_count += 1
                            print(f"Saved #{saved_count}: {batch_label} -> {summary}")
                        else:
                            print("Capture too short, discarded.")
                    else:
                        print("No pose detected during capture, discarded.")

                    batch_reps_done += 1
                    if batch_reps_done >= REPS_PER_BATCH:
                        print(f"Batch complete: {batch_label} x{REPS_PER_BATCH}")
                        phase = "idle"
                        batch_label = None
                        batch_reps_done = 0
                    else:
                        phase = "rest"
                        phase_start_time = now

            elif phase == "rest":
                elapsed = now - phase_start_time
                remaining = REST_SECONDS - elapsed
                status_text = f"{batch_label} #{batch_reps_done + 1}/{REPS_PER_BATCH}: rest"
                sub_text = f"{max(remaining, 0):.1f}s"
                status_color = (255, 255, 0)
                if remaining <= 0:
                    phase = "countdown"
                    phase_start_time = now

            cv2.putText(image, status_text, (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
            if sub_text:
                cv2.putText(image, sub_text, (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
            cv2.putText(image, f"Saved: {saved_count}", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if phase == "idle":
                cv2.putText(image, "1-3=RIGHT jhu  4-6=LEFT jhu  7=RIGHT-LEFT COMBO  q=quit", (30, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow('Punch Data Collector', image)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif phase == "idle" and key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7')):
                batch_label = {
                    ord('1'): "RIGHT_JAB", ord('2'): "RIGHT_HOOK", ord('3'): "RIGHT_UPPERCUT",
                    ord('4'): "LEFT_JAB", ord('5'): "LEFT_HOOK", ord('6'): "LEFT_UPPERCUT",
                    ord('7'): "COMBO_RIGHT_LEFT",
                }[key]
                batch_reps_done = 0
                combo_rep_index = 0
                phase = "countdown"
                phase_start_time = now
                print(f"Starting batch: {batch_label} x{REPS_PER_BATCH}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {saved_count} samples saved to {CSV_PATH}")


if __name__ == "__main__":
    main()
