# ASL Reader

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand%20Landmarker-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A real-time American Sign Language (ASL) translator built with Python, MediaPipe, and OpenCV. Detects hand gestures through your webcam and translates them into text and speech — runs both locally and in the browser.

---

## 🔴 Live Demo

**[Try it here → asl-reader-mtptfb3inb6wszxd9vq4d2.streamlit.app](https://asl-reader-mtptfb3inb6wszxd9vq4d2.streamlit.app)**

> Opens in your browser — no installation needed. Allow camera access and hold a sign for 1.2 seconds to register it.

---

## Features

- **Full A–Z alphabet** — all 26 ASL letters including motion-based J, Q, Z
- **7 common words/phrases** — HELLO, YES, NO, PLEASE, THANKS, I LOVE YOU, STOP
- Real-time hand landmark detection via MediaPipe with skeleton overlay
- Hold-to-confirm system (1.2s) prevents accidental input
- Landmark smoothing (EMA) for stable, jitter-free detection
- Text-to-speech output (local app)
- On-screen sentence builder with clear and backspace
- Modular codebase — easy to extend

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| [MediaPipe](https://mediapipe.dev) | Hand landmark detection (21 points) |
| [OpenCV](https://opencv.org) | Camera capture & frame processing |
| [Streamlit](https://streamlit.io) | Web interface |
| [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) | Browser camera access |
| [Pillow](https://pillow.readthedocs.io) | Frame drawing (web version) |
| pyttsx3 | Text-to-speech (local version) |

---

## Quick Start (Local)

**1. Clone the repository**
```bash
git clone https://github.com/ibrahimaasim77/ASL-Reader.git
cd ASL-Reader
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

**3. Install dependencies**
```bash
pip install opencv-python mediapipe numpy pyttsx3
```

**4. Download the MediaPipe hand landmark model**
```bash
curl -L -o hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

**5. Run**
```bash
python3 main.py
```

| Key     | Action                |
|---------|-----------------------|
| `Q`     | Quit                  |
| `SPACE` | Clear sentence        |
| `DEL`   | Delete last character |

Hold a gesture steady for **1.2 seconds** to register it.

---

## Supported Gestures

### Alphabet — All 26 Letters

| Type | Letters |
|------|---------|
| Static pose | A B C D E F G H I K L M N O P R S T U V W X Y |
| Motion-based | **J** **Q** **Z** |

**Motion letters** — hold the base shape and trace the stroke:
- **J** — Hold I (pinky up), draw a J curve downward then hook left
- **Q** — Hold G shape pointing down, move hand downward
- **Z** — Hold index pointing, trace Z (right → down-left → right)

### Words & Phrases

| Sign | Gesture |
|------|---------|
| HELLO | All 5 fingers open, spread wide |
| YES | Tight closed fist |
| NO | Index and middle extended, close together |
| PLEASE | Open hand, fingers close to thumb |
| THANKS | Open hand, fingers moderately spread |
| I LOVE YOU | Thumb, index, and pinky extended wide |
| STOP | Four fingers up, thumb folded |

---

## Project Structure

```
ASL-Reader/
├── main.py          # Entry point & main loop (local)
├── app.py           # Streamlit web app
├── config.py        # All constants and thresholds
├── camera.py        # Camera initialization & warmup
├── speech.py        # Text-to-speech wrapper
├── ui.py            # Drawing functions
└── gestures/
    ├── helpers.py   # fingers_up(), finger_angles(), landmark smoother
    ├── letters.py   # Static letter detection (A–Z minus J/Q/Z)
    ├── words.py     # Word/phrase detection
    └── dynamic.py   # Motion-based detection (J, Q, Z)
```

---

## Troubleshooting

**Camera hangs on startup (macOS)**
```bash
sudo killall VDCAssistant
```

**`hand_landmarker.task` not found**

Run the `curl` command in step 4 above.

---

## Credits

Developed by **Ibrahim Asim**  
Fullerton College — Computer Science  
GitHub: [@ibrahimaasim77](https://github.com/ibrahimaasim77)
