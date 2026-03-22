# ASL Reader

A real-time American Sign Language (ASL) translator built with Python, OpenCV, and MediaPipe. It detects hand gestures through your webcam and translates them into text and speech.

---

## Features

- Real-time hand landmark detection via MediaPipe
- Recognizes ASL letters (A–Y) and common words
- Text-to-speech output using pyttsx3
- Hold-to-confirm gesture system to prevent accidental input
- On-screen sentence builder with clear functionality

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
pip install opencv-python mediapipe numpy pyttsx3 matplotlib
```

**4. Download the MediaPipe hand landmark model**

Download `hand_landmarker.task` from the [MediaPipe releases page](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) and place it in the project folder.

---

## Running the App

```bash
python3 sign_language.py
```

| Key     | Action          |
|---------|-----------------|
| `Q`     | Quit            |
| `SPACE` | Clear sentence  |

Hold a gesture steady for **1.2 seconds** to register it.

---

## Supported Gestures

### Words
| Gesture | Sign |
|---------|------|
| HELLO | All 5 fingers open |
| YES | Closed fist |
| NO | Index and middle fingers extended |
| PLEASE | Open hand, thumb out wide |
| THANKS | Open hand, fingers close together |
| I LOVE YOU | Thumb, index, and pinky extended |
| STOP | Four fingers up, thumb folded |

### Alphabet
Supports ASL letters: **A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, R, S, T, U, V, W, X, Y**

---

## Troubleshooting

**Camera hangs on startup (macOS)**
```bash
sudo killall VDCAssistant
```
Then rerun the script. If it still hangs, restart your Mac.

**Matplotlib freezes on import**

Add these two lines to the very top of `sign_language.py`:
```python
import matplotlib
matplotlib.use('Agg')
```

---

## Credits

Developed by **Ibrahim Asim**  
Fullerton College — Computer Science  
GitHub: [@ibrahimaasim77](https://github.com/ibrahimaasim77)
