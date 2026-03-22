import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import sys
import os

# ── Text to speech ──────────────────────────────────────────────────────────────
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ── MediaPipe setup ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

# ── ASL finger helpers ──────────────────────────────────────────────────────────
TIPS   = [4, 8, 12, 16, 20]
JOINTS = [3, 6, 10, 14, 18]

def fingers_up(lm):
    up = []
    # Thumb: compare x axis
    up.append(1 if lm[4].x < lm[3].x else 0)
    # Other fingers: compare y axis
    for tip, joint in zip(TIPS[1:], JOINTS[1:]):
        up.append(1 if lm[tip].y < lm[joint].y else 0)
    return up  # [thumb, index, middle, ring, pinky]

def finger_angles(lm):
    """Get normalized distances between key landmarks for better detection."""
    def dist(a, b):
        return np.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)
    return {
        "thumb_index":  dist(4, 8),
        "thumb_middle": dist(4, 12),
        "thumb_pinky":  dist(4, 20),
        "index_middle": dist(8, 12),
        "index_pinky":  dist(8, 20),
        "wrist_middle": dist(0, 12),
    }

# ── ASL gesture map ─────────────────────────────────────────────────────────────
def detect_gesture(lm):
    up = fingers_up(lm)
    d  = finger_angles(lm)
    thumb, index, middle, ring, pinky = up

    # Common words first
    # HELLO — all 5 fingers open, hand raised
    if up == [1, 1, 1, 1, 1]:
        return "HELLO"

    # YES — fist with thumb up nod (fist shape)
    if up == [0, 0, 0, 0, 0] and d["wrist_middle"] < 0.15:
        return "YES"

    # NO — index and middle extended, others closed
    if up == [0, 1, 1, 0, 0] and d["index_middle"] < 0.05:
        return "NO"

    # THANKS — all fingers together pointing out (open hand angled)
    if up == [1, 1, 1, 1, 1] and d["thumb_index"] < 0.08:
        return "THANKS"

    # PLEASE — flat hand on chest (all fingers extended, thumb out)
    if up == [1, 1, 1, 1, 1] and d["thumb_index"] > 0.2:
        return "PLEASE"

    # I LOVE YOU — thumb, index, pinky extended
    if thumb == 1 and index == 1 and middle == 0 and ring == 0 and pinky == 1:
        return "I LOVE YOU"

    # STOP — flat hand, all up
    if up == [0, 1, 1, 1, 1]:
        return "STOP"

    # ── ASL Alphabet ────────────────────────────────────────────────────────────
    # A — fist, thumb to side
    if up == [1, 0, 0, 0, 0] and d["thumb_index"] > 0.1:
        return "A"

    # B — four fingers up, thumb folded
    if up == [0, 1, 1, 1, 1]:
        return "B"

    # C — curved hand (all fingers slightly bent)
    if up == [1, 1, 0, 0, 1] and d["thumb_index"] < 0.15:
        return "C"

    # D — index up, others curled, thumb touches middle
    if up == [0, 1, 0, 0, 0] and d["thumb_middle"] < 0.08:
        return "D"

    # E — all fingers curled down
    if up == [0, 0, 0, 0, 0] and d["wrist_middle"] > 0.15:
        return "E"

    # F — index and thumb touching, others up
    if up == [0, 0, 1, 1, 1] and d["thumb_index"] < 0.05:
        return "F"

    # G — index pointing sideways, thumb out
    if up == [1, 1, 0, 0, 0] and d["index_middle"] > 0.1:
        return "G"

    # H — index and middle pointing sideways
    if up == [0, 1, 1, 0, 0] and d["index_middle"] > 0.08:
        return "H"

    # I — pinky only up
    if up == [0, 0, 0, 0, 1]:
        return "I"

    # K — index and middle up, thumb between them
    if up == [1, 1, 1, 0, 0] and d["thumb_index"] < 0.1:
        return "K"

    # L — thumb and index up (L shape)
    if up == [1, 1, 0, 0, 0] and d["thumb_index"] > 0.15:
        return "L"

    # M — three fingers over thumb
    if up == [0, 0, 0, 0, 0] and d["thumb_index"] < 0.06:
        return "M"

    # N — two fingers over thumb
    if up == [0, 0, 0, 0, 0] and d["thumb_index"] < 0.08:
        return "N"

    # O — all fingers curved to touch thumb
    if d["thumb_index"] < 0.05 and up == [0, 0, 0, 0, 0]:
        return "O"

    # P — index pointing down, thumb out
    if up == [1, 1, 0, 0, 0] and d["thumb_middle"] < 0.1:
        return "P"

    # R — index and middle crossed
    if up == [0, 1, 1, 0, 0] and d["index_middle"] < 0.04:
        return "R"

    # S — fist with thumb over fingers
    if up == [0, 0, 0, 0, 0] and d["thumb_index"] > 0.06:
        return "S"

    # T — thumb between index and middle
    if up == [0, 0, 0, 0, 0] and d["thumb_middle"] < 0.06:
        return "T"

    # U — index and middle up together
    if up == [0, 1, 1, 0, 0] and d["index_middle"] < 0.06:
        return "U"

    # V — index and middle up spread
    if up == [0, 1, 1, 0, 0] and d["index_middle"] > 0.06:
        return "V"

    # W — index, middle, ring up
    if up == [0, 1, 1, 1, 0]:
        return "W"

    # X — index hooked
    if up == [0, 1, 0, 0, 0] and d["thumb_index"] > 0.08:
        return "X"

    # Y — thumb and pinky out
    if up == [1, 0, 0, 0, 1]:
        return "Y"

    return None

# ── Camera open with timeout ─────────────────────────────────────────────────────
def open_camera(index=0, timeout=10):
    """Try to open the camera, attempting index 0 then 1 if needed."""
    for cam_index in [index, 1, 2]:
        print(f"[INFO] Trying camera index {cam_index}...")
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            cap.release()
            continue

        # Try to read a frame with a timeout
        start = time.time()
        while time.time() - start < timeout:
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[INFO] Camera {cam_index} opened successfully.")
                return cap
            time.sleep(0.1)

        print(f"[WARN] Camera {cam_index} opened but no frames received. Trying next...")
        cap.release()

    return None

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("[INFO] Starting ASL translator...")
    print("[INFO] Press Q to quit | SPACE to clear sentence")

    cap = open_camera()
    if cap is None:
        print("[ERROR] Could not get frames from any camera.")
        print("[TIP]  Try running: sudo killall VDCAssistant")
        print("[TIP]  Or restart your Mac to reset the camera session.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Warm up — read 20 frames with timeout
    print("[INFO] Warming up...")
    warmed = 0
    warmup_start = time.time()
    while warmed < 20:
        ret, _ = cap.read()
        if ret:
            warmed += 1
        if time.time() - warmup_start > 10:
            print("[ERROR] Camera timed out during warmup.")
            print("[TIP]  Try running: sudo killall VDCAssistant")
            print("[TIP]  Or restart your Mac to reset the camera session.")
            cap.release()
            sys.exit(1)
    print("[INFO] Warmup done.")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )

    sentence        = ""
    last_gesture    = None
    gesture_start   = 0
    HOLD_TIME       = 1.2    # seconds to hold gesture before it registers
    last_speak_time = 0
    SPEAK_COOLDOWN  = 2.0

    with HandLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = detector.detect(mp_image)

            gesture = None

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]

                # Draw landmarks
                for i in range(len(lm)):
                    cv2.circle(frame, (int(lm[i].x * w), int(lm[i].y * h)), 4, (0, 255, 0), -1)

                gesture = detect_gesture(lm)

                # Hold gesture for HOLD_TIME before registering
                now = time.time()
                if gesture == last_gesture and gesture is not None:
                    held = now - gesture_start
                    # Progress bar
                    bar_w = int((held / HOLD_TIME) * 200)
                    bar_w = min(bar_w, 200)
                    cv2.rectangle(frame, (w//2 - 100, h - 40), (w//2 + 100, h - 20), (50, 50, 50), -1)
                    cv2.rectangle(frame, (w//2 - 100, h - 40), (w//2 - 100 + bar_w, h - 20), (0, 255, 0), -1)
                    cv2.putText(frame, "Hold...", (w//2 - 30, h - 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    if held >= HOLD_TIME:
                        # Register the gesture
                        if gesture in ["HELLO", "YES", "NO", "THANKS", "PLEASE", "I LOVE YOU", "STOP"]:
                            sentence = gesture
                        else:
                            sentence += gesture
                        gesture_start = now  # reset so it doesn't spam

                        # Speak it
                        if now - last_speak_time > SPEAK_COOLDOWN:
                            speak(gesture)
                            last_speak_time = now

                else:
                    last_gesture  = gesture
                    gesture_start = time.time()

            # ── Draw UI ──────────────────────────────────────────────────────────
            # Current gesture
            if gesture:
                cv2.putText(frame, f"Gesture: {gesture}", (15, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 2)
            else:
                cv2.putText(frame, "No gesture", (15, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)

            # Sentence box at bottom
            cv2.rectangle(frame, (0, h - 90), (w, h - 55), (30, 30, 30), -1)
            cv2.putText(frame, f">> {sentence}", (10, h - 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Instructions
            cv2.putText(frame, "SPACE = clear | Q = quit", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            cv2.imshow("ASL Translator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                sentence = ""
                print("[INFO] Sentence cleared.")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")

if __name__ == "__main__":
    main()