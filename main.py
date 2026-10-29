
import cv2
import mediapipe as mp
import numpy as np
import math
import csv
import time
from datetime import datetime
from pathlib import Path

# =====================================
# 1. SETTINGS
# =====================================
MODEL_PATH = "pose_landmarker_lite.task"

TARGET_ANGLE = 95
STANDING_ANGLE = 160
VISIBILITY_LIMIT = 0.60

# =====================================
# 2. CALCULATE KNEE ANGLE
# =====================================
def calculate_angle(hip, knee, ankle):
    a = np.array(hip, dtype=float)
    b = np.array(knee, dtype=float)
    c = np.array(ankle, dtype=float)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator == 0:
        return None

    cosine = np.dot(ba, bc) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return math.degrees(math.acos(cosine))


def get_knee_angle(landmarks, width, height):
    candidates = []

    # Left leg and right leg
    for hip_id, knee_id, ankle_id in [
        (23, 25, 27),
        (24, 26, 28)
    ]:
        hip = landmarks[hip_id]
        knee = landmarks[knee_id]
        ankle = landmarks[ankle_id]

        visibility = min(
            hip.visibility,
            knee.visibility,
            ankle.visibility
        )

        if visibility < VISIBILITY_LIMIT:
            continue

        hip_xy = (hip.x * width, hip.y * height)
        knee_xy = (knee.x * width, knee.y * height)
        ankle_xy = (ankle.x * width, ankle.y * height)

        angle = calculate_angle(
            hip_xy, knee_xy, ankle_xy
        )

        if angle is not None:
            candidates.append(
                (visibility, angle, knee_xy)
            )

    if not candidates:
        return None, None

    # Use the leg with the clearest landmarks
    visibility, angle, knee_xy = max(
        candidates, key=lambda item: item[0]
    )

    return angle, knee_xy


# =====================================
# 3. CHECK MODEL FILE
# =====================================
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(
        "Model file not found: " + MODEL_PATH
    )


# =====================================
# 4. MEDIAPIPE SETUP
# =====================================
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# =====================================
# 5. SKELETON CONNECTIONS
# =====================================
connections = [
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 12),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28)
]


# =====================================
# 6. START CAMERA AND COUNTER
# =====================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Camera could not open. Check macOS camera permissions."
    )

reps = 0
target_reached = False

start_time = time.monotonic()
previous_timestamp_ms = -1

csv_file = open(
    "squat_results.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
)

writer = csv.writer(csv_file)

writer.writerow([
    "timestamp",
    "elapsed_seconds",
    "knee_angle",
    "target_reached",
    "repetitions",
    "status"
])


# =====================================
# 7. MAIN CAMERA LOOP
# =====================================
try:
    with PoseLandmarker.create_from_options(
        options
    ) as landmarker:

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Could not read camera frame.")
                break

            # Mirror view
            frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            elapsed = time.monotonic() - start_time
            elapsed_ms = int(elapsed * 1000)

            timestamp_ms = max(
                elapsed_ms,
                previous_timestamp_ms + 1
            )
            previous_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            angle = None
            knee_xy = None
            status = "Show your full body to camera"

            # ---------------------------------
            # DETECT BODY AND CALCULATE ANGLE
            # ---------------------------------
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                # Draw skeleton
                for start_id, end_id in connections:
                    p1 = landmarks[start_id]
                    p2 = landmarks[end_id]

                    if min(
                        p1.visibility,
                        p2.visibility
                    ) >= 0.5:

                        x1 = int(p1.x * width)
                        y1 = int(p1.y * height)
                        x2 = int(p2.x * width)
                        y2 = int(p2.y * height)

                        cv2.line(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            3
                        )

                angle, knee_xy = get_knee_angle(
                    landmarks, width, height
                )

                # Draw knee marker
                if knee_xy is not None:
                    cv2.circle(
                        frame,
                        (int(knee_xy[0]), int(knee_xy[1])),
                        8,
                        (0, 255, 255),
                        -1
                    )

            # ---------------------------------
            # SQUAT REPETITION LOGIC
            # ---------------------------------
            if angle is None:
                status = "Knee not visible"

            elif angle <= TARGET_ANGLE:
                target_reached = True
                status = "TARGET REACHED!"

            elif angle >= STANDING_ANGLE:
                if target_reached:
                    reps += 1
                    target_reached = False
                    status = "REP COMPLETED!"
                else:
                    status = "Ready - squat down"

            elif target_reached:
                status = "Target reached - stand up"

            else:
                status = "Squatting..."

            # ---------------------------------
            # SAVE RESULTS
            # ---------------------------------
            if angle is not None:
                writer.writerow([
                    datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    round(elapsed, 2),
                    round(angle, 1),
                    target_reached,
                    reps,
                    status
                ])
                csv_file.flush()

            # ---------------------------------
            # DISPLAY INFORMATION PANEL
            # ---------------------------------
            # Dark panel at the top
            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (width, 145),
                (30, 30, 30),
                -1
            )

            # Make the panel slightly transparent
            frame = cv2.addWeighted(
                overlay, 0.78, frame, 0.22, 0
            )

            cv2.putText(
                frame,
                "SQUAT TRACKER",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )

            angle_text = (
                f"Knee angle: {angle:.1f} deg"
                if angle is not None
                else "Knee angle: --"
            )

            cv2.putText(
                frame,
                angle_text,
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"REPS: {reps}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )

            status_color = (
                (0, 255, 0)
                if target_reached
                else (255, 255, 255)
            )

            cv2.putText(
                frame,
                status,
                (max(20, width - 390), 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2
            )

            cv2.putText(
                frame,
                "Press Q to quit",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Show live camera
            cv2.imshow("Squat Tracker", frame)

            # Press Q to stop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()

    print("Squat tracker stopped.")
    print(f"Total repetitions: {reps}")
    print("Results saved to squat_results.csv")
