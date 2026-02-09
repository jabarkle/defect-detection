#!/usr/bin/env python3
"""
3D Print Defect Detection GUI  —  v2 (PySide6 + Material Design)
Group 1  ·  24-641 Manufacturing Data Analytics  ·  Spring 2026

A polished, demo-ready GUI for real-time defect detection using trained
YOLO models.  Built with PySide6 and qt-material for a modern look.

Features:
  - Live camera feed with bounding-box overlays
  - Model selector with recommended best model pre-selected
  - LED alert indicator (green = clear, red = defect)
  - Adjustable confidence threshold
  - Live FPS and per-class detection counters
  - Screenshot capture
  - Scrollable detection log

Usage:
    conda activate base
    python GUIv2.py
"""

import sys
import cv2
import time
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
from ultralytics import YOLO

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSlider, QTextEdit, QFrame,
    QSizePolicy, QSpacerItem, QGridLayout, QGroupBox,
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QSize
from PySide6.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPen, QBrush, QRadialGradient, QIcon

from qt_material import apply_stylesheet

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
# Qt uses hex #RRGGBB
CLASS_COLORS_HEX = {
    "spaghetti": "#EF5350",
    "stringing": "#66BB6A",
    "warping":   "#FFA726",
}

# ── Palette (supplements qt-material dark_teal theme) ──────────────────────
CARD_BG       = "#1e1e2e"
CARD_BORDER   = "#313244"
SURFACE       = "#181825"
TEXT_DIM      = "#a6adc8"
TEXT_BRIGHT   = "#cdd6f4"
ACCENT        = "#89b4fa"   # teal-blue accent
ACCENT_DIM    = "#45475a"
GREEN_OK      = "#a6e3a1"
RED_ALERT     = "#f38ba8"
AMBER         = "#f9e2af"
OVERLAY_BG    = "rgba(24, 24, 37, 180)"


# ── Model discovery ───────────────────────────────────────────────────────

def _discover_models():
    """Find all trained model weights on disk."""
    models = {}

    best = RESULTS_DIR / "yolov11_best_extended" / "weights" / "best.pt"
    if best.exists():
        models["★  YOLOv11 Best  (50 ep, lr=0.002)"] = str(best)

    v9d = RESULTS_DIR / "yolov9_default" / "weights" / "best.pt"
    if v9d.exists():
        models["YOLOv9   Default  (15 ep)"] = str(v9d)

    v11d = RESULTS_DIR / "yolov11_default" / "weights" / "best.pt"
    if v11d.exists():
        models["YOLOv11  Default  (15 ep)"] = str(v11d)

    for arch in ["yolov9", "yolov11"]:
        nice = "YOLOv9 " if arch == "yolov9" else "YOLOv11"
        for lr_tag, lr_label in [("lr_0.2x", "lr 0.2×"), ("lr_5x", "lr 5×")]:
            p = RESULTS_DIR / f"{arch}_{lr_tag}" / "weights" / "best.pt"
            if p.exists():
                models[f"{nice}  {lr_label}  (15 ep)"] = str(p)

    for arch in ["yolov9", "yolov11"]:
        nice = "YOLOv9 " if arch == "yolov9" else "YOLOv11"
        for bs_tag, bs_label in [("batch_8", "batch 8"), ("batch_32", "batch 32")]:
            p = RESULTS_DIR / f"{arch}_{bs_tag}" / "weights" / "best.pt"
            if p.exists():
                models[f"{nice}  {bs_label}  (15 ep)"] = str(p)

    for name, fname in [("YOLOv9s   Pretrained (COCO)", "yolov9s.pt"),
                        ("YOLOv11s  Pretrained (COCO)", "yolo11s.pt")]:
        p = PROJECT_DIR / fname
        if p.exists():
            models[name] = str(p)

    return models


# ── Helper: card frame ────────────────────────────────────────────────────

def _card(title: str | None = None, parent_layout=None) -> tuple[QFrame, QVBoxLayout]:
    """Create a styled card widget and return (frame, inner_layout)."""
    frame = QFrame()
    frame.setObjectName("card")
    frame.setStyleSheet(f"""
        QFrame#card {{
            background-color: {CARD_BG};
            border: 1px solid {CARD_BORDER};
            border-radius: 10px;
        }}
    """)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(8)

    if title:
        lbl = QLabel(title)
        lbl.setStyleSheet(f"""
            font-size: 11px;
            font-weight: 700;
            color: {TEXT_DIM};
            letter-spacing: 1.5px;
            padding-bottom: 2px;
        """)
        layout.addWidget(lbl)

    if parent_layout is not None:
        parent_layout.addWidget(frame)

    return frame, layout


# ── LED Widget ────────────────────────────────────────────────────────────

class LedIndicator(QWidget):
    """A round LED indicator painted with radial gradients."""

    def __init__(self, size=48, parent=None):
        super().__init__(parent)
        self._size = size
        self._color = QColor(ACCENT_DIM)
        self._label_text = "STANDBY"
        self.setFixedSize(size, size)

    def set_state(self, defect: bool):
        if defect:
            self._color = QColor(RED_ALERT)
            self._label_text = "DEFECT"
        else:
            self._color = QColor(GREEN_OK)
            self._label_text = "CLEAR"
        self.update()

    def set_standby(self):
        self._color = QColor(ACCENT_DIM)
        self._label_text = "STANDBY"
        self.update()

    @property
    def label_text(self):
        return self._label_text

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        cx, cy = self._size / 2, self._size / 2
        r = self._size / 2 - 2

        # Outer glow
        glow = QRadialGradient(cx, cy, r * 1.1)
        glow.setColorAt(0, QColor(self._color.red(), self._color.green(), self._color.blue(), 80))
        glow.setColorAt(1, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(glow))
        p.setPen(Qt.NoPen)
        p.drawEllipse(0, 0, self._size, self._size)

        # Main circle
        grad = QRadialGradient(cx * 0.85, cy * 0.75, r)
        lighter = self._color.lighter(140)
        grad.setColorAt(0, lighter)
        grad.setColorAt(1, self._color)
        p.setBrush(QBrush(grad))
        p.setPen(QPen(self._color.darker(130), 1.5))
        inset = 4
        p.drawEllipse(inset, inset, self._size - 2 * inset, self._size - 2 * inset)

        # Gloss highlight
        gloss = QRadialGradient(cx * 0.8, cy * 0.6, r * 0.45)
        gloss.setColorAt(0, QColor(255, 255, 255, 110))
        gloss.setColorAt(1, QColor(255, 255, 255, 0))
        p.setBrush(QBrush(gloss))
        p.setPen(Qt.NoPen)
        p.drawEllipse(int(cx - r * 0.35), int(cy - r * 0.55), int(r * 0.7), int(r * 0.55))

        p.end()


# ── Signals bridge (thread → GUI) ────────────────────────────────────────

class FrameBridge(QObject):
    """Emits signals from the capture thread to the Qt main thread."""
    frame_ready = Signal(np.ndarray, float, dict, bool, list)


# ── Main Window ───────────────────────────────────────────────────────────

class DefectDetectorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Print Defect Detector")
        self.resize(1400, 900)
        self.setMinimumSize(1100, 700)

        # ── State ──
        self.running = False
        self.cap = None
        self.model = None
        self.model_name = ""
        self.confidence = 0.50
        self._last_annotated_frame = None

        self.available_models = _discover_models()
        self.model_keys = list(self.available_models.keys())

        # Thread → GUI bridge
        self._bridge = FrameBridge()
        self._bridge.frame_ready.connect(self._on_frame)

        # ── Build UI ──
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._build_header(root)
        self._build_body(root)

    # ── Header ─────────────────────────────────────────────────────────

    def _build_header(self, root_layout):
        header = QFrame()
        header.setFixedHeight(56)
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {SURFACE};
                border-bottom: 1px solid {CARD_BORDER};
            }}
        """)
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(20, 0, 20, 0)

        # Title
        title = QLabel("DEFECT DETECTION SYSTEM")
        title.setStyleSheet(f"""
            font-size: 18px;
            font-weight: 800;
            color: {ACCENT};
            letter-spacing: 2px;
        """)
        h_lay.addWidget(title)

        h_lay.addStretch()

        # LED indicator in header
        self.led = LedIndicator(size=36)
        h_lay.addWidget(self.led)

        self.led_label = QLabel("STANDBY")
        self.led_label.setStyleSheet(f"""
            font-size: 14px;
            font-weight: 700;
            color: {TEXT_DIM};
            padding-left: 8px;
        """)
        h_lay.addWidget(self.led_label)

        h_lay.addSpacing(30)

        # Course info
        info = QLabel("24-641 Manufacturing Data Analytics  ·  Group 1  ·  Spring 2026")
        info.setStyleSheet(f"font-size: 12px; color: {TEXT_DIM};")
        h_lay.addWidget(info)

        root_layout.addWidget(header)

    # ── Body ───────────────────────────────────────────────────────────

    def _build_body(self, root_layout):
        body = QWidget()
        body.setStyleSheet(f"background-color: {SURFACE};")
        b_lay = QHBoxLayout(body)
        b_lay.setContentsMargins(12, 10, 12, 12)
        b_lay.setSpacing(12)

        # ── Left column: video + log ──
        left = QVBoxLayout()
        left.setSpacing(10)

        self._build_video(left)
        self._build_log(left)

        b_lay.addLayout(left, stretch=7)

        # ── Right column: sidebar cards ──
        right = QVBoxLayout()
        right.setSpacing(10)

        self._build_model_card(right)
        self._build_confidence_card(right)
        self._build_stats_card(right)
        self._build_controls_card(right)
        right.addStretch()

        b_lay.addLayout(right, stretch=3)

        root_layout.addWidget(body, stretch=1)

    # ── Video panel ────────────────────────────────────────────────────

    def _build_video(self, parent_layout):
        frame, lay = _card(parent_layout=parent_layout)
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay.setContentsMargins(6, 6, 6, 6)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 400)
        self.video_label.setStyleSheet(f"""
            background-color: #11111b;
            border-radius: 8px;
            color: {TEXT_DIM};
            font-size: 16px;
        """)
        self.video_label.setText("Camera feed will appear here\nLoad a model → Start Camera")
        lay.addWidget(self.video_label)

    # ── Model card ─────────────────────────────────────────────────────

    def _build_model_card(self, parent_layout):
        _, lay = _card("MODEL", parent_layout=parent_layout)

        self.model_combo = QComboBox()
        self.model_combo.addItems(self.model_keys if self.model_keys else ["No models found"])
        self.model_combo.setStyleSheet(f"""
            QComboBox {{
                padding: 8px 12px;
                font-size: 12px;
                border-radius: 6px;
                background-color: {ACCENT_DIM};
                color: {TEXT_BRIGHT};
                border: 1px solid {CARD_BORDER};
            }}
            QComboBox::drop-down {{
                border: none;
                padding-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {CARD_BG};
                color: {TEXT_BRIGHT};
                selection-background-color: {ACCENT_DIM};
                border: 1px solid {CARD_BORDER};
                padding: 4px;
            }}
        """)
        lay.addWidget(self.model_combo)

        btn_row = QHBoxLayout()
        self.load_btn = QPushButton("Load Model")
        self.load_btn.setCursor(Qt.PointingHandCursor)
        self.load_btn.setFixedHeight(38)
        self.load_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #1e66f5;
                color: #ffffff;
                font-size: 13px;
                font-weight: 700;
                border-radius: 6px;
                padding: 0 18px;
            }}
            QPushButton:hover {{
                background-color: #4d8bf7;
            }}
            QPushButton:pressed {{
                background-color: #1450c8;
            }}
        """)
        self.load_btn.clicked.connect(self._load_model)
        btn_row.addWidget(self.load_btn)
        lay.addLayout(btn_row)

        self.model_status = QLabel("No model loaded")
        self.model_status.setStyleSheet(f"font-size: 11px; color: {TEXT_DIM}; padding-top: 2px;")
        lay.addWidget(self.model_status)

    # ── Confidence card ────────────────────────────────────────────────

    def _build_confidence_card(self, parent_layout):
        _, lay = _card("CONFIDENCE THRESHOLD", parent_layout=parent_layout)

        val_row = QHBoxLayout()
        self.conf_value = QLabel("0.50")
        self.conf_value.setStyleSheet(f"""
            font-size: 28px;
            font-weight: 800;
            color: {AMBER};
            font-family: 'JetBrains Mono', 'Consolas', monospace;
        """)
        val_row.addWidget(self.conf_value)
        val_row.addStretch()
        lay.addLayout(val_row)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(5, 95)
        self.conf_slider.setValue(50)
        self.conf_slider.setTickInterval(5)
        self.conf_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 6px;
                background: {ACCENT_DIM};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT};
                width: 18px;
                height: 18px;
                margin: -7px 0;
                border-radius: 9px;
            }}
            QSlider::handle:horizontal:hover {{
                background: #b4d0fb;
            }}
            QSlider::sub-page:horizontal {{
                background: {ACCENT};
                border-radius: 3px;
            }}
        """)
        self.conf_slider.valueChanged.connect(self._on_conf_change)
        lay.addWidget(self.conf_slider)

    # ── Stats card ─────────────────────────────────────────────────────

    def _build_stats_card(self, parent_layout):
        _, lay = _card("LIVE STATS", parent_layout=parent_layout)

        # FPS
        fps_row = QHBoxLayout()
        fps_lbl = QLabel("FPS")
        fps_lbl.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {TEXT_DIM};")
        fps_row.addWidget(fps_lbl)
        fps_row.addStretch()
        self.fps_value = QLabel("—")
        self.fps_value.setStyleSheet(f"""
            font-size: 22px; font-weight: 800; color: {ACCENT};
            font-family: 'JetBrains Mono', 'Consolas', monospace;
        """)
        fps_row.addWidget(self.fps_value)
        lay.addLayout(fps_row)

        # Separator
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background-color: {CARD_BORDER};")
        lay.addWidget(sep)

        # Per-class counts
        self.class_value_labels = {}
        for cls in CLASS_NAMES:
            row = QHBoxLayout()
            clbl = QLabel(cls.upper())
            clbl.setStyleSheet(f"""
                font-size: 12px;
                font-weight: 700;
                color: {CLASS_COLORS_HEX[cls]};
            """)
            row.addWidget(clbl)
            row.addStretch()
            vlbl = QLabel("0")
            vlbl.setStyleSheet(f"""
                font-size: 20px; font-weight: 800; color: {TEXT_BRIGHT};
                font-family: 'JetBrains Mono', 'Consolas', monospace;
            """)
            row.addWidget(vlbl)
            lay.addLayout(row)
            self.class_value_labels[cls] = vlbl

    # ── Controls card ──────────────────────────────────────────────────

    def _build_controls_card(self, parent_layout):
        _, lay = _card("CONTROLS", parent_layout=parent_layout)

        self.start_btn = QPushButton("▶   Start Camera")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.setFixedHeight(42)
        self.start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #40a02b;
                color: #ffffff;
                font-size: 14px;
                font-weight: 700;
                border-radius: 6px;
            }}
            QPushButton:hover {{ background-color: #55b840; }}
            QPushButton:pressed {{ background-color: #338a22; }}
            QPushButton:disabled {{
                background-color: {ACCENT_DIM};
                color: #6c7086;
            }}
        """)
        self.start_btn.clicked.connect(self._start_camera)
        lay.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■   Stop Camera")
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setFixedHeight(42)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #d20f39;
                color: #ffffff;
                font-size: 14px;
                font-weight: 700;
                border-radius: 6px;
            }}
            QPushButton:hover {{ background-color: #e63e5c; }}
            QPushButton:pressed {{ background-color: #b00d30; }}
            QPushButton:disabled {{
                background-color: {ACCENT_DIM};
                color: #6c7086;
            }}
        """)
        self.stop_btn.clicked.connect(self._stop_camera)
        lay.addWidget(self.stop_btn)

        self.screenshot_btn = QPushButton("📷  Screenshot")
        self.screenshot_btn.setCursor(Qt.PointingHandCursor)
        self.screenshot_btn.setFixedHeight(36)
        self.screenshot_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_DIM};
                color: {TEXT_BRIGHT};
                font-size: 13px;
                font-weight: 600;
                border-radius: 6px;
                border: 1px solid {CARD_BORDER};
            }}
            QPushButton:hover {{ background-color: #585b70; }}
            QPushButton:pressed {{ background-color: #3b3e52; }}
        """)
        self.screenshot_btn.clicked.connect(self._take_screenshot)
        lay.addWidget(self.screenshot_btn)

    # ── Detection log ──────────────────────────────────────────────────

    def _build_log(self, parent_layout):
        frame, lay = _card("DETECTION LOG", parent_layout=parent_layout)
        frame.setFixedHeight(180)

        header_row = QHBoxLayout()
        header_row.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.setCursor(Qt.PointingHandCursor)
        clear_btn.setFixedSize(60, 26)
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_DIM};
                color: {TEXT_DIM};
                font-size: 11px;
                font-weight: 600;
                border-radius: 4px;
            }}
            QPushButton:hover {{ background-color: #585b70; }}
        """)
        clear_btn.clicked.connect(self._clear_log)
        header_row.addWidget(clear_btn)
        lay.addLayout(header_row)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: #11111b;
                color: {ACCENT};
                font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                border: 1px solid {CARD_BORDER};
                border-radius: 6px;
                padding: 8px;
            }}
            QScrollBar:vertical {{
                width: 8px;
                background: transparent;
            }}
            QScrollBar::handle:vertical {{
                background: {ACCENT_DIM};
                border-radius: 4px;
                min-height: 20px;
            }}
        """)
        lay.addWidget(self.log_text)

    # ── Actions ────────────────────────────────────────────────────────

    def _load_model(self):
        key = self.model_combo.currentText()
        if key not in self.available_models:
            return
        path = self.available_models[key]
        self.model_status.setText("Loading...")
        self.model_status.setStyleSheet(f"font-size: 11px; color: {AMBER}; padding-top: 2px;")
        QApplication.processEvents()
        try:
            self.model = YOLO(path)
            self.model_name = key
            short = key.split("(")[0].strip()
            self.model_status.setText(f"Active: {short}")
            self.model_status.setStyleSheet(f"font-size: 11px; color: {GREEN_OK}; padding-top: 2px;")
            self._log(f"Loaded: {key}")
        except Exception as e:
            self.model_status.setText(f"Error: {e}")
            self.model_status.setStyleSheet(f"font-size: 11px; color: {RED_ALERT}; padding-top: 2px;")
            self._log(f"Error: {e}")

    def _on_conf_change(self, val):
        self.confidence = val / 100.0
        self.conf_value.setText(f"{self.confidence:.2f}")

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_text.append(f"<span style='color:{TEXT_DIM}'>[{ts}]</span>  {msg}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def _clear_log(self):
        self.log_text.clear()

    # ── Camera logic ───────────────────────────────────────────────────

    def _start_camera(self):
        if self.model is None:
            self._log("⚠  Load a model first.")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self._log("ERROR: Could not open camera.")
            return

        self.running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._log("Camera started.")
        threading.Thread(target=self._video_loop, daemon=True).start()

    def _stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._update_led_standby()
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

            # Emit to main thread
            self._bridge.frame_ready.emit(annotated, fps, counts, has_defect, detections)
            time.sleep(0.016)  # ~60 fps cap

    def _on_frame(self, frame: np.ndarray, fps: float, counts: dict, has_defect: bool, detections: list):
        """Slot: runs in main thread, updates all UI elements."""
        # Convert BGR → RGB → QPixmap
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Scale to fit label while keeping aspect ratio
        label_size = self.video_label.size()
        pixmap = QPixmap.fromImage(q_img).scaled(
            label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        # Paint overlays onto the pixmap
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # FPS overlay
        painter.setPen(QColor(ACCENT))
        painter.setFont(QFont("JetBrains Mono, Consolas, monospace", 14, QFont.Bold))
        painter.drawText(14, 26, f"FPS: {fps:.1f}")

        # Model name overlay
        if self.model_name:
            painter.setPen(QColor(TEXT_DIM))
            painter.setFont(QFont("JetBrains Mono, Consolas, monospace", 10))
            short = self.model_name.split("(")[0].strip()
            painter.drawText(14, 46, short)

        painter.end()

        self.video_label.setPixmap(pixmap)

        # Update LED
        self._update_led(has_defect)

        # Update stats
        self.fps_value.setText(f"{fps:.1f}")
        for cls in CLASS_NAMES:
            n = counts.get(cls, 0)
            lbl = self.class_value_labels[cls]
            lbl.setText(str(n))
            if n > 0:
                lbl.setStyleSheet(f"""
                    font-size: 20px; font-weight: 800;
                    color: {CLASS_COLORS_HEX[cls]};
                    font-family: 'JetBrains Mono', 'Consolas', monospace;
                """)
            else:
                lbl.setStyleSheet(f"""
                    font-size: 20px; font-weight: 800; color: {TEXT_BRIGHT};
                    font-family: 'JetBrains Mono', 'Consolas', monospace;
                """)

        # Log new detections (only when counts change)
        if not hasattr(self, '_prev_counts') or counts != self._prev_counts:
            for d in detections:
                self._log(
                    f"<span style='color:{CLASS_COLORS_HEX[d['class']]}'>"
                    f"DEFECT: {d['class']}</span>  ({d['confidence']:.0%})"
                )
            self._prev_counts = counts.copy()

    def _update_led(self, defect: bool):
        self.led.set_state(defect)
        if defect:
            self.led_label.setText("DEFECT DETECTED")
            self.led_label.setStyleSheet(f"font-size: 14px; font-weight: 700; color: {RED_ALERT}; padding-left: 8px;")
        else:
            self.led_label.setText("ALL CLEAR")
            self.led_label.setStyleSheet(f"font-size: 14px; font-weight: 700; color: {GREEN_OK}; padding-left: 8px;")

    def _update_led_standby(self):
        self.led.set_standby()
        self.led_label.setText("STANDBY")
        self.led_label.setStyleSheet(f"font-size: 14px; font-weight: 700; color: {TEXT_DIM}; padding-left: 8px;")

    def _detect(self, frame):
        """Run YOLO inference and draw bounding boxes."""
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

    def closeEvent(self, event):
        self.running = False
        if self.cap:
            self.cap.release()
        event.accept()


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)

    # Apply material dark theme (teal accent)
    apply_stylesheet(app, theme="dark_teal.xml")

    # Override the material theme's font with something cleaner
    app.setFont(QFont("Segoe UI", 10))

    window = DefectDetectorWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
