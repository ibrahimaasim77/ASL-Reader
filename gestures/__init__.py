from gestures.helpers import fingers_up, finger_angles, smooth_landmarks, HAND_CONNECTIONS
from gestures.letters import detect_letter
from gestures.words   import detect_word
from gestures.dynamic import TrajectoryTracker, detect_dynamic

WORD_SET = {"HELLO", "YES", "NO", "THANKS", "PLEASE", "I_LOVE_YOU", "STOP"}

def detect_gesture(lm, up, d, tracker):
    """Returns (gesture_name, confidence) or (None, 0.0). Priority: dynamic > word > letter."""
    word,   w_conf = detect_word(lm, up, d)
    letter, l_conf = detect_letter(lm, up, d)
    base = word or letter

    result = detect_dynamic(tracker, lm, base)
    if result:
        return (result, 1.0)
    if word:
        return (word, w_conf)
    if letter:
        return (letter, l_conf)
    return (None, 0.0)
