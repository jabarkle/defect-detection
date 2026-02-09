#!/usr/bin/env python3
"""
3D Print Defect Detection GUI
Group 1 - 24-641 Project 1

Professional GUI for real-time defect detection using trained YOLO models.
Features:
  - Live camera feed with bounding-box overlays
  - Model selector with recommended best model pre-selected
  - LED alert system (green = clear, red = defect detected)
  - Adjustable confidence threshold
  - Live FPS and per-class detection counters
  - Screenshot capture
  - Detection log

Usage:
    python defect_detector_gui.py
"""

import cv2
import time
import threading
import tkinter as tk
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

import customtkinter as ctk

# ── Configuration ──────────────────────────────────────────────────────────
PROJECT_DIR = Path("/home/jesse/Desktop/AI Manufacturing Project 1")
RESULTS_DIR = PROJECT_DIR / "training_results"
SCREENSHOTS_DIR = PROJECT_DIR / "screenshots"

CLASS_NAMES = ["spaghetti", "stringing", "warping"]
CLASS_COLORS_BGR = {
    "spaghetti": (0, 0, 255),
    "stringing": (0, 220, 80),
    "warping":   (255, 140, 0),
}
CLASS_COLORS_HEX = {
    "spaghetti": "#FF4444",
    "stringing": "#44DD66",
    "warping":   "#FF8C00",
}

# ── Color palette ──────────────────────────────────────────────────────────
BG_BLACK     = "#0a0a0a"
BG_PANEL     = "#0d1a0d"
BG_SECTION   = "#142814"
BORDER_GREEN = "#2d4a2d"
GREEN_BRIGHT = "#00ff41"
GREEN_MUTED  = "#7a9f7a"
GREEN_DIM    = "#3a5c3a"
AMBER        = "#ffab00"
RED_ALERT    = "#ff1744"
WHITE        = "#e8e8e8"
TEXT_LABEL   = "#8aaa8a"
TEXT_VALUE   = "#00ff41"
LED_GREEN_C  = "#00ff41"
LED_RED_C    = "#ff1744"
LED_OFF_C    = "#333333"
BTN_GREEN    = "#1b5e20"
BTN_GREEN_H  = "#2e7d32"
BTN_RED      = "#b71c1c"
BTN_RED_H    = "#c62828"
BTN_DARK     = "#1a2e1a"
BTN_DARK_H   = "#2d4a2d"


# ── Model discovery ───────────────────────────────────────────────────────

def _discover_models():
    models = {}
    best = RESULTS_DIR / "yolov11_best_extended" / "weights" / "best.pt"
    if best.exists():
        models["YOLOv11 Best (50 ep, lr=0.002)  ★ Recommended"] = str(best)

    v9d = RESULTS_DIR / "yolov9_default" / "weights" / "best.pt"
    if v9d.exists():
        models["YOLOv9  Default (15 ep)"] = str(v9d)

    v11d = RESULTS_DIR / "yolov11_default" / "weights" / "best.pt"
    if v11d.exists():
        models["YOLOv11 Default (15 ep)"] = str(v11d)

    for arch in ["yolov9", "yolov11"]:
        nice = "YOLOv9" if arch == "yolov9" else "YOLOv11"
        for lr_tag, lr_label in [("lr_0.2x", "lr 0.2x"), ("lr_5x", "lr 5x")]:
            p = RESULTS_DIR / f"{arch}_{lr_tag}" / "weights" / "best.pt"
            if p.exists():
                models[f"{nice}  {lr_label} (15 ep)"] = str(p)

    for arch in ["yolov9", "yolov11"]:
        nice = "YOLOv9" if arch == "yolov9" else "YOLOv11"
        for bs_tag, bs_label in [("batch_8", "batch 8"), ("batch_32", "batch 32")]:
            p = RESULTS_DIR / f"{arch}_{bs_tag}" / "weights" / "best.pt"
            if p.exists():
                models[f"{nice}  {bs_label} (15 ep)"] = str(p)

    for name, fname in [("YOLOv9s  Pretrained (COCO)", "yolov9s.pt"),
                        ("YOLOv11s Pretrained (COCO)", "yolo11s.pt")]:
        p = PROJECT_DIR / fname
        if p.exists():
            models[name] = str(p)

    return models


# ── Theme setup ────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")


# ── Main application ───────────────────────────────────────────────────────

class DefectDetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("3D Print Defect Detector")
        self.geometry("1360x880")
        self.minsize(1150, 750)
        self.configure(fg_color=BG_BLACK)

        # State
        self.running = False
        self.cap = None
        self.model = None
        self.model_name = ""
        self.confidence = 0.50
        self.last_fps = 0.0
        self.class_counts = {c: 0 for c in CLASS_NAMES}
        self.defect_detected = False
        self._photo_ref = None
        self._last_annotated_frame = None

        # Models
        self.available_models = _discover_models()
        self.model_keys = list(self.available_models.keys())

        # Build
        self._build_header()
        self._build_body()

    # ── Header ─────────────────────────────────────────────────────────
    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=BG_PANEL, height=56, corner_radius=0,
                              border_width=0)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header, text="  DEFECT DETECTION SYSTEM",
            font=ctk.CTkFont(family="Helvetica", size=22, weight="bold"),
            text_color=GREEN_BRIGHT,
        ).pack(side="left", padx=20)

        ctk.CTkLabel(
            header,
            text="24-641 Manufacturing Data Analytics   |   Group 1   |   Spring 2026  ",
            font=ctk.CTkFont(size=13), text_color=GREEN_MUTED,
        ).pack(side="right", padx=20)

    # ── Body ───────────────────────────────────────────────────────────
    def _build_body(self):
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=12, pady=(8, 12))

        # Video
        self.video_frame = ctk.CTkFrame(body, fg_color=BG_PANEL, corner_radius=10,
                                        border_width=1, border_color=BORDER_GREEN)
        self.video_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.canvas = tk.Canvas(self.video_frame, bg="#000000", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=6, pady=6)

        # Sidebar
        sidebar = ctk.CTkScrollableFrame(
            body, width=340, fg_color=BG_PANEL, corner_radius=10,
            border_width=1, border_color=BORDER_GREEN,
            scrollbar_button_color=GREEN_DIM,
            scrollbar_button_hover_color=GREEN_MUTED,
        )
        sidebar.pack(side="right", fill="y")

        self._build_led_section(sidebar)
        self._build_model_section(sidebar)
        self._build_confidence_section(sidebar)
        self._build_stats_section(sidebar)
        self._build_controls_section(sidebar)
        self._build_log_section(sidebar)

    # ── LED Alert ──────────────────────────────────────────────────────
    def _build_led_section(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=BG_SECTION, corner_radius=10,
                             border_width=1, border_color=BORDER_GREEN)
        frame.pack(fill="x", padx=8, pady=(8, 5))

        ctk.CTkLabel(
            frame, text="ALERT STATUS",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT_LABEL,
        ).pack(pady=(10, 4))

        led_row = ctk.CTkFrame(frame, fg_color="transparent")
        led_row.pack(pady=(4, 10))

        self.led_canvas = tk.Canvas(
            led_row, width=80, height=80, bg=BG_SECTION, highlightthickness=0,
        )
        self.led_canvas.pack(side="left", padx=(24, 14))

        # LED layers — outer glow, mid ring, core
        self.led_outer = self.led_canvas.create_oval(2, 2, 78, 78,
                                                     fill=LED_OFF_C, outline="", width=0)
        self.led_mid = self.led_canvas.create_oval(10, 10, 70, 70,
                                                   fill=LED_OFF_C, outline="", width=0)
        self.led_core = self.led_canvas.create_oval(20, 20, 60, 60,
                                                    fill=LED_OFF_C, outline="#555555", width=1)
        # Gloss highlight
        self.led_canvas.create_oval(28, 24, 42, 36, fill="#666666", outline="")

        self.led_label = ctk.CTkLabel(
            led_row, text="STANDBY",
            font=ctk.CTkFont(size=20, weight="bold"), text_color=LED_OFF_C,
            wraplength=140,
        )
        self.led_label.pack(side="left", padx=(0, 16))

    def _update_led(self, defect):
        self.defect_detected = defect
        if defect:
            c = LED_RED_C
            self.led_label.configure(text="DEFECT\nDETECTED", text_color=RED_ALERT)
        else:
            c = LED_GREEN_C
            self.led_label.configure(text="ALL\nCLEAR", text_color=GREEN_BRIGHT)
        for item in (self.led_outer, self.led_mid, self.led_core):
            self.led_canvas.itemconfig(item, fill=c)

    # ── Model selector ─────────────────────────────────────────────────
    def _build_model_section(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=BG_SECTION, corner_radius=10,
                             border_width=1, border_color=BORDER_GREEN)
        frame.pack(fill="x", padx=8, pady=5)

        ctk.CTkLabel(
            frame, text="MODEL SELECT",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT_LABEL,
        ).pack(pady=(10, 4))

        self.model_var = ctk.StringVar(
            value=self.model_keys[0] if self.model_keys else "No models found",
        )
        self.model_menu = ctk.CTkOptionMenu(
            frame, variable=self.model_var,
            values=self.model_keys if self.model_keys else ["No models found"],
            width=300, dynamic_resizing=False,
            font=ctk.CTkFont(size=12),
            fg_color=BTN_DARK, button_color=GREEN_DIM,
            button_hover_color=GREEN_MUTED,
            dropdown_fg_color=BG_SECTION,
            dropdown_hover_color=GREEN_DIM,
            dropdown_text_color=WHITE,
        )
        self.model_menu.pack(padx=12, pady=6)

        self.load_btn = ctk.CTkButton(
            frame, text="LOAD MODEL", command=self._load_model,
            font=ctk.CTkFont(size=14, weight="bold"), height=40,
            fg_color=BTN_GREEN, hover_color=BTN_GREEN_H,
            border_width=1, border_color=GREEN_DIM,
        )
        self.load_btn.pack(padx=12, pady=(2, 6))

        self.model_status = ctk.CTkLabel(
            frame, text="No model loaded",
            font=ctk.CTkFont(size=12), text_color=GREEN_DIM,
        )
        self.model_status.pack(pady=(0, 10))

    def _load_model(self):
        key = self.model_var.get()
        if key not in self.available_models:
            return
        path = self.available_models[key]
        self.model_status.configure(text="Loading...", text_color=AMBER)
        self.update_idletasks()
        try:
            self.model = YOLO(path)
            self.model_name = key
            short = key.split("(")[0].strip()
            self.model_status.configure(text=f"Active: {short}", text_color=GREEN_BRIGHT)
            self._log(f"Loaded: {key}")
        except Exception as e:
            self.model_status.configure(text=f"Error: {e}", text_color=RED_ALERT)
            self._log(f"Error: {e}")

    # ── Confidence ─────────────────────────────────────────────────────
    def _build_confidence_section(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=BG_SECTION, corner_radius=10,
                             border_width=1, border_color=BORDER_GREEN)
        frame.pack(fill="x", padx=8, pady=5)

        ctk.CTkLabel(
            frame, text="CONFIDENCE THRESHOLD",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT_LABEL,
        ).pack(pady=(10, 2))

        self.conf_label = ctk.CTkLabel(
            frame, text="0.50",
            font=ctk.CTkFont(family="Courier", size=32, weight="bold"),
            text_color=AMBER,
        )
        self.conf_label.pack(pady=(0, 2))

        self.conf_slider = ctk.CTkSlider(
            frame, from_=0.05, to=0.95, number_of_steps=90,
            command=self._on_conf_change, width=280,
            progress_color=GREEN_DIM, button_color=GREEN_BRIGHT,
            button_hover_color=GREEN_MUTED,
            fg_color=BORDER_GREEN,
        )
        self.conf_slider.set(0.50)
        self.conf_slider.pack(padx=14, pady=(0, 12))

    def _on_conf_change(self, val):
        self.confidence = float(val)
        self.conf_label.configure(text=f"{self.confidence:.2f}")

    # ── Stats ──────────────────────────────────────────────────────────
    def _build_stats_section(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=BG_SECTION, corner_radius=10,
                             border_width=1, border_color=BORDER_GREEN)
        frame.pack(fill="x", padx=8, pady=5)

        ctk.CTkLabel(
            frame, text="LIVE STATS",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT_LABEL,
        ).pack(pady=(10, 6))

        # FPS row
        fps_row = ctk.CTkFrame(frame, fg_color="transparent")
        fps_row.pack(fill="x", padx=16, pady=2)
        ctk.CTkLabel(fps_row, text="FPS",
                     font=ctk.CTkFont(size=14, weight="bold"),
                     text_color=TEXT_LABEL).pack(side="left")
        self.fps_value = ctk.CTkLabel(
            fps_row, text="—",
            font=ctk.CTkFont(family="Courier", size=22, weight="bold"),
            text_color=GREEN_BRIGHT,
        )
        self.fps_value.pack(side="right")

        # Separator
        sep = ctk.CTkFrame(frame, fg_color=BORDER_GREEN, height=1)
        sep.pack(fill="x", padx=16, pady=4)

        # Per-class counts
        self.class_value_labels = {}
        for cls in CLASS_NAMES:
            row = ctk.CTkFrame(frame, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=2)
            ctk.CTkLabel(
                row, text=cls.upper(),
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=CLASS_COLORS_HEX[cls],
            ).pack(side="left")
            val = ctk.CTkLabel(
                row, text="0",
                font=ctk.CTkFont(family="Courier", size=22, weight="bold"),
                text_color=WHITE,
            )
            val.pack(side="right")
            self.class_value_labels[cls] = val

        ctk.CTkLabel(frame, text="", height=4).pack()

    def _update_stats(self, fps, counts):
        self.fps_value.configure(text=f"{fps:.1f}")
        for cls in CLASS_NAMES:
            n = counts.get(cls, 0)
            lbl = self.class_value_labels[cls]
            lbl.configure(text=str(n))
            if n > 0:
                lbl.configure(text_color=CLASS_COLORS_HEX[cls])
            else:
                lbl.configure(text_color=WHITE)

    # ── Controls ───────────────────────────────────────────────────────
    def _build_controls_section(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=BG_SECTION, corner_radius=10,
                             border_width=1, border_color=BORDER_GREEN)
        frame.pack(fill="x", padx=8, pady=5)

        ctk.CTkLabel(
            frame, text="CONTROLS",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT_LABEL,
        ).pack(pady=(10, 6))

        self.start_btn = ctk.CTkButton(
            frame, text="▶   START CAMERA", command=self._start_camera,
            font=ctk.CTkFont(size=14, weight="bold"), height=42,
            fg_color=BTN_GREEN, hover_color=BTN_GREEN_H,
            border_width=1, border_color=GREEN_DIM,
        )
        self.start_btn.pack(fill="x", padx=12, pady=3)

        self.stop_btn = ctk.CTkButton(
            frame, text="■   STOP CAMERA", command=self._stop_camera,
            font=ctk.CTkFont(size=14, weight="bold"), height=42,
            fg_color=BTN_RED, hover_color=BTN_RED_H,
            border_width=1, border_color="#5c1a1a",
            state="disabled",
        )
        self.stop_btn.pack(fill="x", padx=12, pady=3)

        self.screenshot_btn = ctk.CTkButton(
            frame, text="SCREENSHOT", command=self._take_screenshot,
            font=ctk.CTkFont(size=13, weight="bold"), height=36,
            fg_color=BTN_DARK, hover_color=BTN_DARK_H,
            border_width=1, border_color=BORDER_GREEN,
        )
        self.screenshot_btn.pack(fill="x", padx=12, pady=(3, 12))

    # ── Log ────────────────────────────────────────────────────────────
    def _build_log_section(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=BG_SECTION, corner_radius=10,
                             border_width=1, border_color=BORDER_GREEN)
        frame.pack(fill="x", padx=8, pady=(5, 8))

        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x", padx=12, pady=(10, 4))

        ctk.CTkLabel(
            header, text="DETECTION LOG",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT_LABEL,
        ).pack(side="left")

        ctk.CTkButton(
            header, text="CLEAR", width=60, height=26,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=BTN_DARK, hover_color=BTN_DARK_H,
            command=self._clear_log,
        ).pack(side="right")

        self.log_text = ctk.CTkTextbox(
            frame, height=150,
            font=ctk.CTkFont(family="Courier", size=12),
            fg_color="#050a05", text_color=GREEN_BRIGHT,
            corner_radius=6, border_width=1, border_color=BORDER_GREEN,
            scrollbar_button_color=GREEN_DIM,
        )
        self.log_text.pack(fill="x", padx=10, pady=(2, 12))

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.see("end")

    def _clear_log(self):
        self.log_text.delete("1.0", "end")

    # ── Camera logic ───────────────────────────────────────────────────

    def _start_camera(self):
        if self.model is None:
            self._log("Load a model first.")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self._log("ERROR: Could not open camera.")
            return

        self.running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._log("Camera started.")
        threading.Thread(target=self._video_loop, daemon=True).start()

    def _stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self._update_led(False)
        self._log("Camera stopped.")

    def _video_loop(self):
        prev = time.perf_counter()
        while self.running:
            if self.cap is None:
                break
            ret, frame = self.cap.read()
            if not ret:
                break

            detections, annotated = self._detect(frame)
            self._last_annotated_frame = annotated

            now = time.perf_counter()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now

            counts = {c: 0 for c in CLASS_NAMES}
            for d in detections:
                counts[d["class"]] = counts.get(d["class"], 0) + 1

            has_defect = len(detections) > 0

            # Convert for display
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw > 1 and ch > 1:
                rgb = cv2.resize(rgb, (cw, ch))

            pil_img = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(image=pil_img)

            self.after(0, self._update_frame, photo, fps, counts, has_defect, detections)
            time.sleep(0.02)

    def _update_frame(self, photo, fps, counts, has_defect, detections):
        self._photo_ref = photo
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=photo)

        # FPS overlay
        self.canvas.create_text(
            14, 14, anchor="nw", text=f"FPS: {fps:.1f}",
            fill=GREEN_BRIGHT, font=("Courier", 16, "bold"),
        )
        # Model name overlay
        if self.model_name:
            short = self.model_name.split("(")[0].strip()
            self.canvas.create_text(
                14, 40, anchor="nw", text=short,
                fill=GREEN_MUTED, font=("Courier", 12),
            )

        self._update_led(has_defect)
        self._update_stats(fps, counts)

        if counts != self.class_counts:
            for d in detections:
                self._log(f"DEFECT: {d['class']} ({d['confidence']:.2f})")
            self.class_counts = counts.copy()

    def _detect(self, frame):
        if self.model is None:
            return [], frame

        results = self.model(frame, conf=self.confidence, verbose=False)

        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                    detections.append({
                        "class": cls_name, "confidence": conf, "bbox": xyxy,
                    })

        annotated = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            color = CLASS_COLORS_BGR.get(d["class"], (255, 255, 255))
            # Thicker box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label = f"{d['class']} {d['confidence']:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 14), (x1 + tw + 8, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return detections, annotated

    # ── Screenshot ─────────────────────────────────────────────────────

    def _take_screenshot(self):
        if self._last_annotated_frame is None:
            self._log("No frame to capture.")
            return
        SCREENSHOTS_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = SCREENSHOTS_DIR / f"screenshot_{ts}.png"
        cv2.imwrite(str(path), self._last_annotated_frame)
        self._log(f"Saved: {path.name}")

    # ── Cleanup ────────────────────────────────────────────────────────

    def _on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.destroy()


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    app = DefectDetectorApp()
    app.protocol("WM_DELETE_WINDOW", app._on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
