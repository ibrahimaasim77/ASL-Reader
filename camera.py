import cv2
import time
import sys
from config import CAMERA_INDICES, CAMERA_WIDTH, CAMERA_HEIGHT, WARMUP_FRAMES, WARMUP_TIMEOUT

def open_camera(indices=None, timeout=10):
    if indices is None:
        indices = CAMERA_INDICES
    for idx in indices:
        print(f"[INFO] Trying camera index {idx}...")
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        start = time.time()
        while time.time() - start < timeout:
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[INFO] Camera {idx} opened.")
                return cap
            time.sleep(0.05)
        print(f"[WARN] Camera {idx}: no frames. Trying next...")
        cap.release()
    return None

def configure_camera(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

def warmup_camera(cap):
    print("[INFO] Warming up camera...")
    count, start = 0, time.time()
    while count < WARMUP_FRAMES:
        ret, _ = cap.read()
        if ret:
            count += 1
        if time.time() - start > WARMUP_TIMEOUT:
            print("[ERROR] Camera warmup timed out.")
            return False
    print("[INFO] Warmup done.")
    return True
