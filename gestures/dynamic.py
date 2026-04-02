from collections import deque
from config import THRESH

class TrajectoryTracker:
    def __init__(self):
        maxlen = THRESH["TRAJECTORY_MAX_FRAMES"]
        self.positions = deque(maxlen=maxlen)
        self.active_base = None

    def update(self, lm, base_gesture):
        if base_gesture not in ("I", "Q_base", "Z_base"):
            self.reset()
            return None
        if base_gesture != self.active_base:
            self.reset()
            self.active_base = base_gesture
        self.positions.append((lm[8].x, lm[8].y))
        if len(self.positions) < THRESH["TRAJECTORY_MIN_FRAMES"]:
            return None
        return self._classify()

    def reset(self):
        self.positions.clear()
        self.active_base = None

    def _classify(self):
        pts = list(self.positions)
        if self.active_base == "I":    return self._j(pts)
        if self.active_base == "Q_base": return self._q(pts)
        if self.active_base == "Z_base": return self._z(pts)

    def _j(self, pts):
        mid = len(pts) // 2
        disp = THRESH["TRAJECTORY_MIN_DISP"]
        if (pts[mid][1] - pts[0][1] > disp and
                pts[-1][0] - pts[mid][0] < -disp * 0.5):
            return "J"

    def _q(self, pts):
        disp = THRESH["TRAJECTORY_MIN_DISP"]
        dy = pts[-1][1] - pts[0][1]
        dx = abs(pts[-1][0] - pts[0][0])
        if dy > disp and dx < disp:
            return "Q"

    def _z(self, pts):
        disp = THRESH["TRAJECTORY_MIN_DISP"]
        signs = []
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i-1][0]
            if abs(dx) > 0.005:
                signs.append(1 if dx > 0 else -1)
        if len(signs) < 4:
            return None
        reversals = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i-1])
        span = abs(pts[-1][0] - pts[0][0])
        if reversals >= 2 and span > disp:
            return "Z"


def detect_dynamic(tracker, lm, base_gesture):
    return tracker.update(lm, base_gesture)
