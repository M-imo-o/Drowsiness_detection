"""
Drowsiness Detection System
============================
Uses MediaPipe Face Landmarker + Eye Aspect Ratio (EAR) to detect drowsiness.
When eyes are closed for >= 5 seconds, a beep alarm sounds.
The alarm stops immediately when eyes reopen.

Press 'q' to quit.
"""

import cv2
import numpy as np
import time
import threading
import struct
import wave
import os
import tempfile
import winsound
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MODEL_PATH = "face_landmarker_v2_with_blendshapes.task"
EAR_THRESHOLD = 0.2          # Below this → eyes considered closed
CLOSED_DURATION_LIMIT = 5.0  # Seconds of closure before alarm

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
# Generate a WAV beep tone file (plays through speakers)
# ──────────────────────────────────────────────
def generate_beep_wav(filepath, freq=2500, duration_ms=600, volume=0.8):
    """Create a WAV file with a sine-wave beep tone."""
    sample_rate = 44100
    n_samples = int(sample_rate * duration_ms / 1000)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        value = volume * 32767 * np.sin(2 * np.pi * freq * t)
        samples.append(int(value))

    with wave.open(filepath, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f'<{len(samples)}h', *samples))


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
    # --- Generate alarm WAV file ---
    beep_wav = os.path.join(tempfile.gettempdir(), "drowsiness_beep.wav")
    generate_beep_wav(beep_wav, freq=2500, duration_ms=600)
    print(f"Alarm sound generated: {beep_wav}")

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

    eyes_closed_start = None  # timestamp when eyes first closed
    drowsy = False
    alarm_playing = False
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
                if eyes_closed_start is None:
                    eyes_closed_start = time.time()
                elapsed = time.time() - eyes_closed_start

                if elapsed >= CLOSED_DURATION_LIMIT and not drowsy:
                    drowsy = True
                    # Play beep WAV in a loop through speakers (async)
                    winsound.PlaySound(  # type: ignore[attr-defined]
                        beep_wav,
                        winsound.SND_FILENAME | winsound.SND_LOOP | winsound.SND_ASYNC  # type: ignore[attr-defined]
                    )
                    alarm_playing = True
                    print("ALARM ON - eyes closed for 5+ seconds!")

                # Show countdown / alert
                if drowsy:
                    cv2.putText(frame, "!! DROWSINESS ALERT !!",
                                (30, h - 60), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (0, 0, 255), 3)
                else:
                    cv2.putText(frame, f"Eyes closed: {elapsed:.1f}s / {CLOSED_DURATION_LIMIT:.0f}s",
                                (30, h - 60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 165, 255), 2)
            else:
                # Eyes open → reset everything
                eyes_closed_start = None
                if drowsy:
                    drowsy = False
                    # Stop the alarm immediately
                    winsound.PlaySound(None, winsound.SND_PURGE)  # type: ignore[attr-defined]
                    alarm_playing = False
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
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("Exited cleanly.")


if __name__ == "__main__":
    main()
