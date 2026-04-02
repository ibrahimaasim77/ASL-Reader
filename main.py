import cv2
import mediapipe as mp
import sys
import time
from collections import deque

from config import (
    MODEL_PATH, MIN_DETECTION_CONFIDENCE, MIN_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE, NUM_HANDS, HOLD_TIME, SPEAK_COOLDOWN,
    FRAME_SKIP, EMA_ALPHA
)
from camera  import open_camera, configure_camera, warmup_camera
from speech  import SpeechEngine
from ui      import (draw_landmarks, draw_gesture_label,
                     draw_hold_progress, draw_sentence_box, draw_instructions)
from gestures import detect_gesture, WORD_SET
from gestures.helpers import fingers_up, finger_angles, smooth_landmarks
from gestures.dynamic import TrajectoryTracker

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

def main():
    print("[INFO] Starting ASL translator...")
    print("[INFO] SPACE=clear  DEL=backspace  Q=quit")

    cap = open_camera()
    if cap is None:
        print("[ERROR] No camera found. Try: sudo killall VDCAssistant")
        sys.exit(1)

    configure_camera(cap)
    if not warmup_camera(cap):
        cap.release()
        sys.exit(1)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )

    speech     = SpeechEngine()
    tracker    = TrajectoryTracker()
    lm_history = deque(maxlen=2)

    sentence      = ""
    last_gesture  = None
    gesture_start = 0.0
    frame_count   = 0
    gesture       = None

    with HandLandmarker.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            frame_count += 1

            if frame_count % FRAME_SKIP == 0:
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

                if result.hand_landmarks:
                    raw_lm = result.hand_landmarks[0]
                    handedness = (result.handedness[0][0].category_name
                                  if result.handedness else "Right")

                    lm = smooth_landmarks(lm_history, raw_lm, EMA_ALPHA)
                    draw_landmarks(frame, lm, w, h)

                    up = fingers_up(lm, handedness)
                    d  = finger_angles(lm)
                    gesture, _ = detect_gesture(raw_lm, up, d, tracker)

                    now = time.time()
                    if gesture == last_gesture and gesture is not None:
                        held = now - gesture_start
                        draw_hold_progress(frame, gesture, held, w, h)
                        if held >= HOLD_TIME:
                            if gesture in WORD_SET:
                                sentence = gesture
                            else:
                                sentence += gesture
                            gesture_start = now
                            speech.speak(gesture, SPEAK_COOLDOWN)
                    else:
                        last_gesture  = gesture
                        gesture_start = now
                else:
                    gesture = None
                    tracker.reset()
                    lm_history.clear()

            draw_gesture_label(frame, gesture, w)
            draw_sentence_box(frame, sentence, w, h)
            draw_instructions(frame, h)
            cv2.imshow("ASL Translator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord(' '):
                sentence = ""
            elif key in (127, 8):
                sentence = sentence[:-1]

    cap.release()
    cv2.destroyAllWindows()
    speech.shutdown()
    print("[INFO] Done.")

if __name__ == "__main__":
    main()
