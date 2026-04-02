import sys
from unittest.mock import MagicMock

# mediapipe imports cv2 only for drawing utils we never use.
# Mocking it prevents the libGL.so.1 error on headless servers.
if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
import threading
import time
import urllib.request
import os
from collections import deque

from config import (MODEL_PATH, MIN_DETECTION_CONFIDENCE, MIN_PRESENCE_CONFIDENCE,
                    MIN_TRACKING_CONFIDENCE, NUM_HANDS, HOLD_TIME, EMA_ALPHA, FRAME_SKIP)
from gestures import detect_gesture, WORD_SET
from gestures.helpers import fingers_up, finger_angles, smooth_landmarks, HAND_CONNECTIONS
from gestures.dynamic import TrajectoryTracker

# ── Styling ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ASL Reader", page_icon="🤟", layout="centered")
st.markdown("""
<style>
  .stApp { background-color: #0d0d0d; color: #ffffff; }
  #MainMenu, footer, header { visibility: hidden; }
  .stButton > button {
    background-color: #cc0000; color: white; border: none;
    border-radius: 8px; font-size: 1rem; padding: 0.6rem 1.2rem;
    transition: background 0.2s;
  }
  .stButton > button:hover { background-color: #ff1a1a; color: white; }
  hr { border-color: #cc0000; }
  table { color: #ffffff !important; }
  th { color: #cc0000 !important; border-bottom: 1px solid #cc0000 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "started" not in st.session_state:
    st.session_state.started = False

# ── Model download ────────────────────────────────────────────────────────────
def download_model():
    if not os.path.exists(MODEL_PATH):
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        with st.spinner("Downloading hand landmark model (~25MB)..."):
            urllib.request.urlretrieve(url, MODEL_PATH)

# ── PIL drawing helpers ───────────────────────────────────────────────────────
def _font(size=20):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()

def pil_landmarks(draw, lm, w, h):
    for a, b in HAND_CONNECTIONS:
        ax, ay = int(lm[a].x * w), int(lm[a].y * h)
        bx, by = int(lm[b].x * w), int(lm[b].y * h)
        draw.line([(ax, ay), (bx, by)], fill=(200, 200, 200), width=1)
    for pt in lm:
        cx, cy = int(pt.x * w), int(pt.y * h)
        draw.ellipse([(cx-4, cy-4), (cx+4, cy+4)], fill=(0, 255, 0))

def pil_gesture_label(draw, gesture, h):
    font = _font(28)
    if gesture:
        draw.text((15, 15), f"Gesture: {gesture}", fill=(0, 220, 0), font=font)
    else:
        draw.text((15, 15), "No gesture", fill=(100, 100, 100), font=font)

def pil_hold_progress(draw, gesture, held, w, h):
    progress = min(held / HOLD_TIME, 1.0)
    bar_w    = int(progress * 200)
    font     = _font(22)
    draw.text((w//2 - 40, h - 72), gesture, fill=(0, 255, 0), font=font)
    draw.rectangle([(w//2-100, h-50), (w//2+100, h-34)], fill=(50, 50, 50))
    if bar_w > 0:
        draw.rectangle([(w//2-100, h-50), (w//2-100+bar_w, h-34)], fill=(0, 255, 0))

def pil_sentence_box(draw, sentence, w, h):
    font = _font(20)
    draw.rectangle([(0, h-80), (w, h-50)], fill=(30, 30, 30))
    display = sentence[-40:] if len(sentence) > 40 else sentence
    draw.text((10, h-74), f">> {display}", fill=(255, 255, 255), font=font)

# ── Video processor (no cv2) ──────────────────────────────────────────────────
class ASLProcessor(VideoProcessorBase):
    def __init__(self):
        BaseOptions           = mp.tasks.BaseOptions
        HandLandmarker        = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode     = mp.tasks.vision.RunningMode
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=MIN_PRESENCE_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self.detector        = HandLandmarker.create_from_options(options)
        self.tracker         = TrajectoryTracker()
        self.lm_history      = deque(maxlen=2)
        self.last_gesture    = None
        self.gesture_start   = 0.0
        self.frame_count     = 0
        self.current_gesture = None
        self.current_lm      = None
        self.sentence        = ""
        self.lock            = threading.Lock()
        self._clear          = threading.Event()
        self._backspace      = threading.Event()

    def recv(self, frame):
        # Get RGB numpy array — no cv2 needed
        img = np.fliplr(frame.to_ndarray(format="rgb24")).copy()
        h, w = img.shape[:2]

        if self._clear.is_set():
            with self.lock: self.sentence = ""
            self._clear.clear()
        if self._backspace.is_set():
            with self.lock: self.sentence = self.sentence[:-1]
            self._backspace.clear()

        self.frame_count += 1

        if self.frame_count % FRAME_SKIP == 0:
            result = self.detector.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=img))

            if result.hand_landmarks:
                raw_lm     = result.hand_landmarks[0]
                handedness = (result.handedness[0][0].category_name
                              if result.handedness else "Right")
                lm = smooth_landmarks(self.lm_history, raw_lm, EMA_ALPHA)
                self.current_lm = lm

                up = fingers_up(lm, handedness)
                d  = finger_angles(lm)
                gesture, _ = detect_gesture(raw_lm, up, d, self.tracker)
                self.current_gesture = gesture

                now = time.time()
                if gesture == self.last_gesture and gesture is not None:
                    held = now - self.gesture_start
                    if held >= HOLD_TIME:
                        with self.lock:
                            if gesture in WORD_SET:
                                self.sentence = gesture
                            else:
                                self.sentence += gesture
                        self.gesture_start = now
                else:
                    self.last_gesture  = gesture
                    self.gesture_start = now
            else:
                self.current_gesture = None
                self.current_lm      = None
                self.tracker.reset()
                self.lm_history.clear()

        # Draw with PIL
        pil_img = Image.fromarray(img)
        draw    = ImageDraw.Draw(pil_img)

        if self.current_lm:
            pil_landmarks(draw, self.current_lm, w, h)
            if (self.current_gesture and
                    self.current_gesture == self.last_gesture):
                held = time.time() - self.gesture_start
                if held > 0:
                    pil_hold_progress(draw, self.current_gesture, held, w, h)

        pil_gesture_label(draw, self.current_gesture, h)
        pil_sentence_box(draw, self.sentence, w, h)

        return av.VideoFrame.from_ndarray(np.array(pil_img), format="rgb24")

    def clear_sentence(self): self._clear.set()
    def do_backspace(self):   self._backspace.set()

# ── Intro screen ──────────────────────────────────────────────────────────────
if not st.session_state.started:
    st.markdown("""
    <div style='text-align:center; padding: 3rem 0 1rem 0;'>
        <div style='font-size: 5rem;'>🤟</div>
        <h1 style='font-size: 3rem; color: #ffffff; margin: 0;'>ASL Reader</h1>
        <p style='color: #cc0000; font-size: 1.2rem; margin: 0.5rem 0 0 0;'>
            Real-time American Sign Language Translator
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown("""<div style='background:#1a1a1a; border-left:3px solid #cc0000;
            padding:1rem; border-radius:6px; text-align:center;'>
            <div style='font-size:2rem;'>📷</div><b>Live Camera</b><br>
            <span style='color:#aaa;font-size:0.85rem;'>Detects your hand in real time</span>
            </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div style='background:#1a1a1a; border-left:3px solid #cc0000;
            padding:1rem; border-radius:6px; text-align:center;'>
            <div style='font-size:2rem;'>🔤</div><b>Full Alphabet A–Z</b><br>
            <span style='color:#aaa;font-size:0.85rem;'>All 26 letters + 7 common words</span>
            </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div style='background:#1a1a1a; border-left:3px solid #cc0000;
            padding:1rem; border-radius:6px; text-align:center;'>
            <div style='font-size:2rem;'>⏱️</div><b>Hold to Confirm</b><br>
            <span style='color:#aaa;font-size:0.85rem;'>1.2s hold registers a sign</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    col_l, col_m, col_r = st.columns([1.5, 1, 1.5])
    with col_m:
        if st.button("▶  Start Translating", use_container_width=True):
            st.session_state.started = True
            st.rerun()

    st.markdown("""<p style='text-align:center; color:#555; font-size:0.8rem; margin-top:2rem;'>
        Built with MediaPipe · Streamlit &nbsp;|&nbsp;
        By <a href='https://github.com/ibrahimaasim77' style='color:#cc0000;'>Ibrahim Asim</a>
        </p>""", unsafe_allow_html=True)

# ── Translator screen ─────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem;'>
        <span style='font-size:2rem;'>🤟</span>
        <h2 style='margin:0;color:#ffffff;'>ASL Reader</h2>
        <span style='color:#cc0000;font-size:0.9rem;margin-left:auto;'>LIVE</span>
    </div>""", unsafe_allow_html=True)

    download_model()

    ctx = webrtc_streamer(
        key="asl-reader",
        video_processor_factory=ASLProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if ctx.video_processor:
        col1, col2 = st.columns(2)
        if col1.button("⌫  Backspace", use_container_width=True):
            ctx.video_processor.do_backspace()
        if col2.button("🗑️  Clear", use_container_width=True):
            ctx.video_processor.clear_sentence()

    st.divider()
    st.markdown("""
**Supported signs**

| Type | Signs |
|------|-------|
| Static letters | A B C D E F G H I K L M N O P R S T U V W X Y |
| Motion letters | **J** · **Q** · **Z** |
| Words | HELLO · YES · NO · PLEASE · THANKS · I_LOVE_YOU · STOP |
""")

    if st.button("← Back to home"):
        st.session_state.started = False
        st.rerun()
