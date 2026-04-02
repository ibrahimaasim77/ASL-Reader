import numpy as np
from collections import deque

TIPS   = [4, 8, 12, 16, 20]
JOINTS = [3, 6, 10, 14, 18]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def fingers_up(lm, handedness="Right"):
    up = []
    if handedness == "Right":
        up.append(1 if lm[4].x < lm[3].x else 0)
    else:
        up.append(1 if lm[4].x > lm[3].x else 0)
    for tip, joint in zip(TIPS[1:], JOINTS[1:]):
        up.append(1 if lm[tip].y < lm[joint].y else 0)
    return up

def finger_angles(lm):
    def dist(a, b):
        return np.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)
    scale = dist(0, 12) or 1e-6
    return {
        "thumb_index":   dist(4, 8)  / scale,
        "thumb_middle":  dist(4, 12) / scale,
        "thumb_ring":    dist(4, 16) / scale,
        "thumb_pinky":   dist(4, 20) / scale,
        "index_middle":  dist(8, 12) / scale,
        "index_pinky":   dist(8, 20) / scale,
        "middle_ring":   dist(12,16) / scale,
        "ring_pinky":    dist(16,20) / scale,
        "wrist_middle":  scale,
        # Raw y positions (image coords: y increases downward)
        "thumb_tip_y":   lm[4].y,
        "index_mcp_y":   lm[5].y,
        "middle_mcp_y":  lm[9].y,
        "ring_mcp_y":    lm[13].y,
    }

class _LM:
    __slots__ = ('x', 'y', 'z')
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

def smooth_landmarks(lm_history, new_lm, alpha=0.7):
    if not lm_history:
        lm_history.append(new_lm)
        return new_lm
    prev = lm_history[-1]
    smoothed = [
        _LM(alpha*n.x + (1-alpha)*p.x,
            alpha*n.y + (1-alpha)*p.y,
            alpha*n.z + (1-alpha)*p.z)
        for p, n in zip(prev, new_lm)
    ]
    lm_history.append(smoothed)
    return smoothed
