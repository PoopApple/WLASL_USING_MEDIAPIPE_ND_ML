"""
ui_main.py
==========
Layout definitions for the ASL recognition UI.

Screens
  0 – Home menu
  1 – Video Testing  (choose / random video, result table, motion energy bar)
  2 – Real Time      (camera feed, prominent top-5 predictions panel)
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QStackedWidget, QDialog, QDialogButtonBox,
    QFormLayout, QSpinBox, QCheckBox, QLineEdit, QFileDialog,
    QSizePolicy, QScrollArea,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import json
import os


# ─────────────────────────────────────────────────────────────────────────────
class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.config = config

        layout      = QVBoxLayout()
        form_layout = QFormLayout()

        # Pose model
        self.pose_model_select = QComboBox()
        self.pose_model_select.addItems(["lite", "full", "heavy"])
        self.pose_model_select.setCurrentText(config.get("pose_model", "lite"))
        form_layout.addRow("Pose Model:", self.pose_model_select)

        # Motion check interval
        self.motion_interval_spin = QSpinBox()
        self.motion_interval_spin.setRange(1, 60)
        self.motion_interval_spin.setValue(config.get("motion_check_interval", 5))
        self.motion_interval_spin.setSuffix(" frames")
        form_layout.addRow("Motion Check Interval:", self.motion_interval_spin)

        # Periodic fallback
        self.fallback_spin = QSpinBox()
        self.fallback_spin.setRange(5, 200)
        self.fallback_spin.setValue(config.get("periodic_fallback_interval", 30))
        self.fallback_spin.setSuffix(" checks")
        form_layout.addRow("Periodic Fallback (checks):", self.fallback_spin)

        # Draw landmarks
        self.draw_landmarks_cb = QCheckBox("Draw Landmarks")
        self.draw_landmarks_cb.setChecked(config.get("draw_landmarks", True))
        form_layout.addRow(self.draw_landmarks_cb)

        # Video file
        vf_layout = QHBoxLayout()
        self.video_file_input = QLineEdit(config.get("video_file", ""))
        browse_vf = QPushButton("Browse")
        browse_vf.clicked.connect(self._browse_video)
        vf_layout.addWidget(self.video_file_input)
        vf_layout.addWidget(browse_vf)
        form_layout.addRow("Video File:", vf_layout)

        # Prediction model
        pm_layout = QHBoxLayout()
        self.pred_model_input = QLineEdit(config.get("prediction_model", ""))
        browse_pm = QPushButton("Browse")
        browse_pm.clicked.connect(self._browse_model)
        pm_layout.addWidget(self.pred_model_input)
        pm_layout.addWidget(browse_pm)
        form_layout.addRow("Prediction Model:", pm_layout)

        layout.addLayout(form_layout)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)
        self.setLayout(layout)

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", "",
                                               "Video Files (*.mp4 *.avi *.mkv *.mov *.webm)")
        if path:
            self.video_file_input.setText(path)

    def _browse_model(self):
        os.makedirs("ASL_Models", exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(self, "Select Prediction Model",
                                               "ASL_Models", "Keras Models (*.keras)")
        if path:
            self.pred_model_input.setText(path)

    def accept(self):
        self.config["pose_model"]                 = self.pose_model_select.currentText()
        self.config["video_file"]                 = self.video_file_input.text()
        self.config["prediction_model"]           = self.pred_model_input.text()
        self.config["motion_check_interval"]      = self.motion_interval_spin.value()
        self.config["periodic_fallback_interval"] = self.fallback_spin.value()
        self.config["draw_landmarks"]             = self.draw_landmarks_cb.isChecked()
        with open("config.json", "w") as f:
            json.dump(self.config, f, indent=4)
        super().accept()


# ─────────────────────────────────────────────────────────────────────────────
def _make_btn(text: str, min_h: int = 40) -> QPushButton:
    btn = QPushButton(text)
    btn.setMinimumHeight(min_h)
    return btn


def _make_label(text: str = "", bold: bool = False, align=Qt.AlignCenter,
                font_size: int = 0) -> QLabel:
    lbl = QLabel(text)
    lbl.setAlignment(align)
    if bold or font_size:
        f = lbl.font()
        if bold:
            f.setBold(True)
        if font_size:
            f.setPointSize(font_size)
        lbl.setFont(f)
    return lbl


# ─────────────────────────────────────────────────────────────────────────────
class MainUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASL Recognition System")
        self.stacked_widget = QStackedWidget()

        self.stacked_widget.addWidget(self._build_home())     # 0
        self.stacked_widget.addWidget(self._build_video())    # 1
        self.stacked_widget.addWidget(self._build_realtime()) # 2

        root = QVBoxLayout()
        root.addWidget(self.stacked_widget)
        self.setLayout(root)

    # ── Screen 0: Home ────────────────────────────────────────────────────────
    def _build_home(self):
        w      = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(16)

        title = _make_label("ASL Recognition System", bold=True, font_size=22)

        self.nav_video_btn    = _make_btn("🎬  Video Testing",  min_h=60)
        self.nav_realtime_btn = _make_btn("📷  Real-Time Demo", min_h=60)
        self.nav_settings_btn = _make_btn("⚙️  Settings",       min_h=60)

        layout.addStretch()
        layout.addWidget(title)
        layout.addSpacing(24)
        layout.addWidget(self.nav_video_btn)
        layout.addWidget(self.nav_realtime_btn)
        layout.addWidget(self.nav_settings_btn)
        layout.addStretch()
        w.setLayout(layout)
        return w

    # ── Screen 1: Video Testing ───────────────────────────────────────────────
    def _build_video(self):
        w      = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Top bar
        top = QHBoxLayout()
        self.video_back_btn   = _make_btn("← Back")
        self.video_choose_btn = _make_btn("Choose Video")
        self.video_random_btn = _make_btn("Random Video")
        top.addWidget(self.video_back_btn)
        top.addWidget(self.video_choose_btn)
        top.addWidget(self.video_random_btn)

        # Video feed
        self.video_feed_label = QLabel()
        self.video_feed_label.setAlignment(Qt.AlignCenter)
        self.video_feed_label.setMinimumHeight(300)
        self.video_feed_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_feed_label.setStyleSheet("background: #111;")

        self.video_name_label = _make_label("No video selected")
        self.video_name_label.setStyleSheet("color: #888; font-style: italic;")

        # Motion energy bar (drawn by main.py as a pixmap)
        energy_title = _make_label("Motion Energy", bold=True)
        energy_title.setStyleSheet("color: #aaa; font-size: 11px;")
        self.motion_energy_label = QLabel()
        self.motion_energy_label.setMinimumHeight(64)
        self.motion_energy_label.setStyleSheet("background: #1a1a1a; border-radius: 4px;")
        self.motion_energy_label.setAlignment(Qt.AlignCenter)
        self.motion_energy_label.setText("Run a video to see motion energy")

        # Results table
        results_title = _make_label("Predictions", bold=True)
        results_title.setStyleSheet("color: #ddd; font-size: 12px; margin-top: 4px;")
        self.video_output_label = QLabel("Select a video to begin...")
        self.video_output_label.setAlignment(Qt.AlignCenter)
        self.video_output_label.setWordWrap(True)
        self.video_output_label.setTextFormat(Qt.RichText)
        self.video_output_label.setStyleSheet("color: #ccc;")

        layout.addLayout(top)
        layout.addWidget(self.video_feed_label, stretch=4)
        layout.addWidget(self.video_name_label)
        layout.addWidget(energy_title)
        layout.addWidget(self.motion_energy_label)
        layout.addWidget(results_title)
        layout.addWidget(self.video_output_label, stretch=2)
        w.setLayout(layout)
        return w

    # ── Screen 2: Real-Time ───────────────────────────────────────────────────
    def _build_realtime(self):
        w      = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(8)

        # Top bar
        top = QHBoxLayout()
        self.rt_back_btn      = _make_btn("← Back")
        self.rt_camera_select = QComboBox()
        self.rt_camera_select.addItems(["Camera 0", "Camera 1", "Camera 2"])
        self.rt_camera_select.setMinimumHeight(36)
        top.addWidget(self.rt_back_btn)
        top.addWidget(self.rt_camera_select)

        # Camera feed
        self.rt_feed_label = QLabel()
        self.rt_feed_label.setAlignment(Qt.AlignCenter)
        self.rt_feed_label.setMinimumHeight(280)
        self.rt_feed_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.rt_feed_label.setStyleSheet("background: #111;")

        self.rt_name_label = _make_label("")
        self.rt_name_label.setStyleSheet("color: #888; font-style: italic; font-size: 11px;")

        # Start / Stop
        ctrl = QHBoxLayout()
        self.rt_start_btn = _make_btn("▶  Start Recognition", min_h=44)
        self.rt_start_btn.setStyleSheet("background: #1e7e34; color: white; font-weight: bold;")
        self.rt_stop_btn  = _make_btn("■  Stop",              min_h=44)
        self.rt_stop_btn.setStyleSheet("background: #7e1e1e; color: white; font-weight: bold;")
        ctrl.addWidget(self.rt_start_btn)
        ctrl.addWidget(self.rt_stop_btn)

        # Motion energy bar (drawn by main.py as a pixmap)
        energy_title = _make_label("Motion Energy", bold=True)
        energy_title.setStyleSheet("color: #aaa; font-size: 11px;")
        self.rt_motion_energy_label = QLabel()
        self.rt_motion_energy_label.setMinimumHeight(64)
        self.rt_motion_energy_label.setStyleSheet("background: #1a1a1a; border-radius: 4px;")
        self.rt_motion_energy_label.setAlignment(Qt.AlignCenter)

        # Predictions panel — scrollable rich-text label
        pred_title = _make_label("PREDICTIONS", bold=True)
        pred_title.setStyleSheet(
            "color: white; background: #222; padding: 6px; font-size: 13px; letter-spacing: 2px;"
        )

        self.rt_output_label = QLabel("Press ▶ Start to begin recognising signs.")
        self.rt_output_label.setAlignment(Qt.AlignCenter)
        self.rt_output_label.setWordWrap(True)
        self.rt_output_label.setTextFormat(Qt.RichText)
        self.rt_output_label.setMinimumHeight(140)
        self.rt_output_label.setStyleSheet(
            "background: #1a1a2e; color: #eee; padding: 8px; border-radius: 6px; font-size: 13px;"
        )

        layout.addLayout(top)
        layout.addWidget(self.rt_feed_label, stretch=4)
        layout.addWidget(self.rt_name_label)
        layout.addLayout(ctrl)
        layout.addWidget(energy_title)
        layout.addWidget(self.rt_motion_energy_label)
        layout.addWidget(pred_title)
        layout.addWidget(self.rt_output_label, stretch=2)
        w.setLayout(layout)
        return w