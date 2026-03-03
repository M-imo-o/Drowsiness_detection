"""
Drowsiness Detection System
============================
Uses MediaPipe Face Landmarker + Eye Aspect Ratio (EAR) to detect drowsiness.
When eyes are closed for >= 1 second, an alarm sounds.
The alarm stops immediately when eyes reopen.

Press 'q' to quit.
"""

import cv2
import numpy as np
import time
import threading
import os
import winsound
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MODEL_PATH = "face_landmarker_v2_with_blendshapes.task"
ALARM_WAV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fahhhhh.wav")
EAR_THRESHOLD = 0.2          # Below this → eyes considered closed
CLOSED_DURATION_LIMIT = 1.0  # Seconds of closure before alarm
FLASH_DELAY = 2.0           # Seconds after alarm before flash starts
FLASH_INTERVAL = 0.1        # Seconds between black/white toggles

# MediaPipe face‑mesh indices for each eye
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ──────────────────────────────────────────────
# Globals updated by the async callback
# ──────────────────────────────────────────────
latest_ear = None
latest_landmarks = None
lock = threading.Lock()


# ──────────────────────────────────────────────
# EAR calculation
# ──────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices):
    """Compute EAR for a single eye given 6 landmark indices."""
    pts = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
    # Vertical distances
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    # Horizontal distance
    h = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.0


# ──────────────────────────────────────────────
# MediaPipe async callback
# ──────────────────────────────────────────────
def on_result(result, output_image, timestamp_ms):
    global latest_ear, latest_landmarks
    if result.face_landmarks:
        lm = result.face_landmarks[0]
        left_ear = eye_aspect_ratio(lm, LEFT_EYE)
        right_ear = eye_aspect_ratio(lm, RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        with lock:
            latest_ear = ear
            latest_landmarks = lm
    else:
        with lock:
            latest_ear = None
            latest_landmarks = None


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    # --- Verify alarm sound file exists ---
    if not os.path.isfile(ALARM_WAV):
        print(f"ERROR: Alarm sound file not found: {ALARM_WAV}")
        return
    print(f"Using alarm sound: {ALARM_WAV}")

    # --- MediaPipe Face Landmarker (LIVE_STREAM mode) ---
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        result_callback=on_result,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    # --- Webcam ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    eyes_closed_start: float = 0.0  # timestamp when eyes first closed (0 = not closed)
    drowsy = False
    alarm_playing = False
    alarm_start_time: float = 0.0   # when the alarm started
    flash_active = False
    flash_state = False             # False = black, True = white
    last_flash_toggle: float = 0.0
    frame_ts = 0

    print("Drowsiness Detector started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_ts += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_image, frame_ts)

        # --- Read latest results ---
        with lock:
            ear = latest_ear

        h, w = frame.shape[:2]

        if ear is not None:
            eyes_closed = ear < EAR_THRESHOLD

            if eyes_closed:
                if eyes_closed_start == 0.0:
                    eyes_closed_start = time.time()
                elapsed = time.time() - eyes_closed_start

                if elapsed >= CLOSED_DURATION_LIMIT and not drowsy:
                    drowsy = True
                    # Play beep WAV in a loop through speakers (async)
                    winsound.PlaySound(  # type: ignore[attr-defined]
                        ALARM_WAV,
                        winsound.SND_FILENAME | winsound.SND_LOOP | winsound.SND_ASYNC  # type: ignore[attr-defined]
                    )
                    alarm_playing = True
                    alarm_start_time = time.time()
                    print("ALARM ON - eyes closed for 1+ seconds!")

                # Show countdown / alert
                if drowsy:
                    cv2.putText(frame, "!! DROWSINESS ALERT !!",
                                (30, h - 60), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 0, 255), 3)

                    # --- Flashing lights (2s after alarm) ---
                    if time.time() - alarm_start_time >= FLASH_DELAY:
                        if not flash_active:
                            flash_active = True
                            cv2.namedWindow("WAKE UP", cv2.WINDOW_NORMAL)
                            cv2.setWindowProperty("WAKE UP", cv2.WND_PROP_FULLSCREEN,
                                                  cv2.WINDOW_FULLSCREEN)
                            last_flash_toggle = time.time()
                            print("FLASH ON - flashing lights activated!")

                        # Toggle black ↔ white every FLASH_INTERVAL
                        now = time.time()
                        if now - last_flash_toggle >= FLASH_INTERVAL:
                            flash_state = not flash_state
                            last_flash_toggle = now
                        color_val = 255 if flash_state else 0
                        flash_frame = np.full((800, 1280), color_val, dtype=np.uint8)
                        cv2.imshow("WAKE UP", flash_frame)
                else:
                    cv2.putText(frame, f"Eyes closed: {elapsed:.1f}s / {CLOSED_DURATION_LIMIT:.0f}s",
                                (30, h - 60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 165, 255), 2)
            else:
                # Eyes open → reset everything
                eyes_closed_start = 0.0
                if drowsy:
                    drowsy = False
                    # Stop the alarm immediately
                    winsound.PlaySound(None, winsound.SND_PURGE)  # type: ignore[attr-defined]
                    alarm_playing = False
                    alarm_start_time = 0.0
                    # Stop flashing
                    if flash_active:
                        flash_active = False
                        flash_state = False
                        cv2.destroyWindow("WAKE UP")
                        print("FLASH OFF - flashing lights stopped.")
                    print("ALARM OFF - eyes opened.")

            # EAR readout
            color = (0, 255, 0) if not eyes_closed else (0, 0, 255)
            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            state = "OPEN" if not eyes_closed else "CLOSED"
            cv2.putText(frame, f"Eyes: {state}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            cv2.putText(frame, "No face detected", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Drowsiness Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    winsound.PlaySound(None, winsound.SND_PURGE)  # type: ignore[attr-defined]
    if flash_active:
        cv2.destroyWindow("WAKE UP")
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Exited cleanly.")


if __name__ == "__main__":
    main()
