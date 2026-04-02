import pyttsx3
import time
from config import TTS_RATE

class SpeechEngine:
    def __init__(self, rate=TTS_RATE):
        self._engine = pyttsx3.init()
        self._engine.setProperty('rate', rate)
        self._last_speak = 0.0

    def speak(self, text, cooldown=2.0):
        now = time.time()
        if now - self._last_speak < cooldown:
            return False
        self._engine.say(text)
        self._engine.runAndWait()
        self._last_speak = now
        return True

    def shutdown(self):
        self._engine.stop()
