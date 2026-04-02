# ASL Reader

A real-time American Sign Language (ASL) translator built with Python, OpenCV, and MediaPipe. Detects hand gestures through your webcam and translates them into text and speech.

---

## Features

- Real-time hand landmark detection via MediaPipe (with skeleton overlay)
- Recognizes all 26 ASL letters (A–Z) including motion-based J, Q, Z
- Recognizes 7 common words/phrases
- Text-to-speech output using pyttsx3
- Hold-to-confirm gesture system (1.2s) to prevent accidental input
- On-screen sentence builder with clear and backspace functionality
- Landmark smoothing for stable detection

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

Download `hand_landmarker.task` from the [MediaPipe releases page](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) and place it in the project root.

---

## Running the App

```bash
python3 main.py
```

| Key     | Action          |
|---------|-----------------|
| `Q`     | Quit            |
| `SPACE` | Clear sentence  |
| `DEL`   | Delete last character |

Hold a gesture steady for **1.2 seconds** to register it.

---

## Supported Gestures

### Words
| Gesture | Sign |
|---------|------|
| HELLO | All 5 fingers open, spread wide |
| YES | Tight closed fist |
| NO | Index and middle fingers extended, close together |
| PLEASE | Open hand, fingers close to thumb |
| THANKS | Open hand, fingers moderately spread |
| I_LOVE_YOU | Thumb, index, and pinky extended wide |
| STOP | Four fingers up, thumb folded |

### Alphabet
**A–Z** — all 26 letters supported.

| Static letters | Motion-based |
|---------------|-------------|
| A B C D E F G H I K L M N O P R S T U V W X Y | J Q Z |

J, Q, and Z are detected by tracking finger movement trajectories:
- **J** — Hold the I shape (pinky up), then trace a J curve downward and hook left
- **Q** — Hold a G shape pointing down, then move the hand downward
- **Z** — Hold index finger pointing, then trace a Z stroke (right → down-left → right)

---

## Project Structure

```
ASL-Reader/
├── main.py          # Entry point
├── config.py        # All constants and thresholds
├── camera.py        # Camera initialization
├── speech.py        # Text-to-speech wrapper
├── ui.py            # OpenCV drawing functions
└── gestures/
    ├── helpers.py   # fingers_up(), finger_angles(), landmark smoother
    ├── letters.py   # Static letter detection (A–Z minus J/Q/Z)
    ├── words.py     # Word/phrase detection
    └── dynamic.py   # Motion-based letter detection (J, Q, Z)
```

---

## Troubleshooting

**Camera hangs on startup (macOS)**
```bash
sudo killall VDCAssistant
```
Then rerun. If it still hangs, restart your Mac.

---

## Credits

Developed by **Ibrahim Asim**  
Fullerton College — Computer Science  
GitHub: [@ibrahimaasim77](https://github.com/ibrahimaasim77)
