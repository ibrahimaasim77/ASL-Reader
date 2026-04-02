# ASL Reader

A real-time American Sign Language (ASL) translator built with Python, OpenCV, and MediaPipe. Detects hand gestures through your webcam and translates them into text and speech.

## 🔴 Live Demo

**[Try it here → asl-reader-mtptfb3inb6wszxd9vq4d2.streamlit.app](https://asl-reader-mtptfb3inb6wszxd9vq4d2.streamlit.app)**

> Opens in your browser — no installation needed. Allow camera access and hold a sign for 1.2 seconds to register it.

---

## Features

- Real-time hand landmark detection via MediaPipe (with skeleton overlay)
- Recognizes **all 26 ASL letters** (A–Z) including motion-based J, Q, Z
- Recognizes 7 common words/phrases
- Text-to-speech output using pyttsx3
- Hold-to-confirm gesture system (1.2s) to prevent accidental input
- On-screen sentence builder with clear and backspace
- Landmark smoothing (EMA) for stable detection
- Modular codebase — easy to extend

---

## Requirements

- Python 3.10–3.12
- A working webcam
- macOS, Windows, or Linux

---

## Installation

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

---

## Running the App

```bash
python3 main.py
```

| Key     | Action               |
|---------|----------------------|
| `Q`     | Quit                 |
| `SPACE` | Clear sentence       |
| `DEL`   | Delete last character|

Hold a gesture steady for **1.2 seconds** to register it.

---

## Supported Gestures

### Words
| Gesture | Sign |
|---------|------|
| HELLO | All 5 fingers open, spread wide |
| YES | Tight closed fist |
| NO | Index and middle extended, close together |
| PLEASE | Open hand, fingers close to thumb |
| THANKS | Open hand, fingers moderately spread |
| I_LOVE_YOU | Thumb, index, and pinky extended wide |
| STOP | Four fingers up, thumb folded |

### Alphabet — All 26 Letters

| Type | Letters |
|------|---------|
| Static pose | A B C D E F G H I K L M N O P R S T U V W X Y |
| Motion-based | J Q Z |

**Motion letters** — hold the base shape and trace the stroke:
- **J** — Hold I (pinky up), draw a J curve downward then hook left
- **Q** — Hold a G shape pointing down, move hand downward
- **Z** — Hold index finger pointing, trace Z (right → down-left → right)

---

## Project Structure

```
ASL-Reader/
├── main.py          # Entry point & main loop
├── config.py        # All constants and thresholds
├── camera.py        # Camera initialization & warmup
├── speech.py        # Text-to-speech wrapper
├── ui.py            # OpenCV drawing functions
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
Then rerun. If it still hangs, restart your Mac.

**`hand_landmarker.task` not found**

Run the curl command in step 4 of Installation to download the model file.

---

## Credits

Developed by **Ibrahim Asim**  
Fullerton College — Computer Science  
GitHub: [@ibrahimaasim77](https://github.com/ibrahimaasim77)
