import argparse
import csv
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Local import 
try:
    from model_utils import load_model, predict, draw_prediction_overlay
except ImportError:
    print("[ERROR] model_utils.py not found.")
    sys.exit(1)


# CONFIG 
WEIGHTS_PATH = "model_weights/best_model.pth"
CONFIG_PATH  = "model_weights/model_config.json"
WINDOW_TITLE = "Weld Defect — Real-Time Detection"

OUTPUT_DIR = Path("./outputs")
LOG_FILE   = OUTPUT_DIR / "detection_log.csv"

DISPLAY_W = 960
DISPLAY_H = 540


# FPS TRACKER 
class FPSTracker:
    def __init__(self, window=30):
        self.times = deque(maxlen=window)
        self.t0 = time.perf_counter()

    def tick(self):
        now = time.perf_counter()
        self.times.append(now - self.t0)
        self.t0 = now
        if len(self.times) < 2:
            return 0.0
        return 1.0 / (sum(self.times) / len(self.times))


# LOGGER 
class DetectionLogger:
    def __init__(self, path):
        path.parent.mkdir(exist_ok=True)
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["time", "label", "conf", "latency", "fps"])

    def log(self, r, fps):
        self.w.writerow([
            datetime.now().isoformat(),
            r.get("label", ""),
            r.get("confidence", 0),
            r.get("latency_ms", 0),
            fps
        ])

    def close(self):
        self.f.close()


# CAMERA 
def open_camera(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Camera not found")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_H)
    return cap


# MAIN 
def run(camera_index=0):

    print("[INFO] Loading model...")
    model, class_names, img_size, device = load_model(
        WEIGHTS_PATH, CONFIG_PATH
    )

    print("[INFO] Opening camera...")
    cap = open_camera(camera_index)

    logger = DetectionLogger(LOG_FILE)
    fps_trk = FPSTracker()

    paused = False
    skip_n = 2
    frame_cnt = 0

   
    last_result = {
        "label": "loading",
        "confidence": 0.0,
        "is_defect": False,
        "latency_ms": 0.0
    }

    cv2.namedWindow(WINDOW_TITLE)

    print("Press q to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            fps = fps_trk.tick()
            frame_cnt += 1

            if not paused and frame_cnt % skip_n == 0:
                last_result = predict(
                    model, frame, class_names, img_size, device
                )
                logger.log(last_result, fps)

            display = draw_prediction_overlay(frame, last_result, fps)

            if paused:
                cv2.putText(display, "PAUSED", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)

            cv2.imshow(WINDOW_TITLE, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = not paused
            elif key == ord("f"):
                skip_n = 1 if skip_n > 1 else 3

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.close()
        print("[INFO] Done.")


# CLI 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    run(args.camera)