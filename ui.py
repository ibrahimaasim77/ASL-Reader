import cv2
from config import (
    COLOR_GREEN, COLOR_DARK_BG, COLOR_PROGRESS, COLOR_PROGRESS_BG,
    COLOR_LANDMARK, COLOR_BONE, COLOR_TEXT, COLOR_GREY, COLOR_DIM, HOLD_TIME
)
from gestures.helpers import HAND_CONNECTIONS

def draw_landmarks(frame, lm, w, h):
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame,
                 (int(lm[a].x * w), int(lm[a].y * h)),
                 (int(lm[b].x * w), int(lm[b].y * h)),
                 COLOR_BONE, 1)
    for pt in lm:
        cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 4, COLOR_LANDMARK, -1)

def draw_gesture_label(frame, gesture, w):
    if gesture:
        cv2.putText(frame, f"Gesture: {gesture}", (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_GREEN, 2)
    else:
        cv2.putText(frame, "No gesture", (15, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_DIM, 2)

def draw_hold_progress(frame, gesture, held, w, h):
    progress = min(held / HOLD_TIME, 1.0)
    bar_w    = int(progress * 200)
    label    = gesture
    tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0][0]
    cv2.putText(frame, label, (w//2 - tw//2, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_GREEN, 2)
    cv2.rectangle(frame, (w//2 - 100, h - 38), (w//2 + 100, h - 22), COLOR_PROGRESS_BG, -1)
    cv2.rectangle(frame, (w//2 - 100, h - 38), (w//2 - 100 + bar_w, h - 22), COLOR_PROGRESS, -1)

def draw_sentence_box(frame, sentence, w, h):
    cv2.rectangle(frame, (0, h - 90), (w, h - 55), COLOR_DARK_BG, -1)
    display = sentence[-40:] if len(sentence) > 40 else sentence
    cv2.putText(frame, f">> {display}", (10, h - 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)

def draw_instructions(frame, h):
    cv2.putText(frame, "SPACE=clear  DEL=backspace  Q=quit",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_GREY, 1)
