import csv
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import (
    BooleanVar, DoubleVar, IntVar, StringVar,
    messagebox, ttk,
)
import tkinter as tk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk
from model_utils import load_model, predict, draw_prediction_overlay


WEIGHTS_PATH  = "model_weights/best_model.pth"
CONFIG_PATH   = "model_weights/model_config.json"
OUTPUT_DIR    = Path("./outputs")
LOG_FILE      = OUTPUT_DIR / "tkinter_log.csv"

PREVIEW_W     = 800
PREVIEW_H     = 450
PANEL_W       = 280

BG_DARK       = "#0f1117"
BG_MID        = "#1a1d2e"
BG_CARD       = "#242840"
FG_WHITE      = "#e8eaf6"
FG_MUTED      = "#7986cb"
ACCENT_BLUE   = "#5c6bc0"
RED_DEFECT    = "#ef5350"
GREEN_OK      = "#66bb6a"

FONT_MONO     = ("Consolas", 10)
FONT_LABEL    = ("Arial", 10)
FONT_BIG      = ("Arial", 28, "bold")



# CAMERA SCANNER (FIXED)

def list_cameras(max_check=5):
    cams = []
    for i in range(max_check):
        if os.name == "nt":
            cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        else:
            cap = cv2.VideoCapture(i)

        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cams.append((i, f"Camera {i}"))
        cap.release()

    return cams if cams else [(0, "Camera 0")]


# LOGGER

class Logger:
    def __init__(self, path):
        path.parent.mkdir(exist_ok=True)
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["time", "label", "confidence", "fps"])
        self.count = 0

    def log(self, result, fps):
        self.writer.writerow([
            datetime.now().isoformat(),
            result.get("label"),
            result.get("confidence"),
            fps
        ])
        self.count += 1

    def close(self):
        self.file.close()


# MAIN APP

class WeldDefectApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Weld Defect AI")
        self.configure(bg=BG_DARK)

        self._model = None
        self._cap = None
        self._running = False
        self._thread = None
        self._last_result = {}
        self._fps_queue = deque(maxlen=20)

        self._build_ui()
        self._load_model()

    # UI
    def _build_ui(self):
        self.canvas = tk.Canvas(self, width=800, height=450, bg="black")
        self.canvas.pack()

        bar = tk.Frame(self, bg=BG_MID)
        bar.pack(fill="x")

        self.cam_var = StringVar()
        cams = list_cameras()
        self.cam_map = {label: idx for idx, label in cams}
        self.cam_var.set(cams[0][1])

        ttk.Combobox(bar, textvariable=self.cam_var,
                     values=list(self.cam_map.keys())).pack(side="left")

        tk.Button(bar, text="Start", command=self.start).pack(side="left")
        tk.Button(bar, text="Stop", command=self.stop).pack(side="left")

        self.status = tk.Label(self, text="Loading model...", fg="white", bg=BG_DARK)
        self.status.pack()

    # MODEL
    def _load_model(self):
        def load():
            self._model, self.cls, self.size, self.device = load_model(
                WEIGHTS_PATH, CONFIG_PATH
            )
            self.status.config(text="Model Ready")

        threading.Thread(target=load, daemon=True).start()

    # CAMERA
    def start(self):
        cam_idx = self.cam_map[self.cam_var.get()]

        if os.name == "nt":
            self._cap = cv2.VideoCapture(cam_idx, cv2.CAP_MSMF)
        else:
            self._cap = cv2.VideoCapture(cam_idx)

        if not self._cap.isOpened():
            messagebox.showerror("Error", "Camera failed")
            return

        self._running = True
        self._thread = threading.Thread(target=self.loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()

    # LOOP
    def loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue

            t0 = time.time()

            result = predict(self._model, frame, self.cls, self.size, self.device)
            frame = draw_prediction_overlay(frame, result, 0)

            fps = 1 / (time.time() - t0 + 1e-6)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)

            self.canvas.create_image(0, 0, anchor="nw", image=img)
            self.canvas.image = img

    def on_close(self):
        self.stop()
        self.destroy()


# RUN

if __name__ == "__main__":
    app = WeldDefectApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()