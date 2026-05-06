"""
SettingsPage — Configuration editor linked to the JSON config file.

Provides an intuitive UI for editing all pipeline parameters with
live validation and hot-reload. Styled with luxury gold aesthetics.
"""

import os
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit,
    QPushButton, QGroupBox, QFileDialog, QScrollArea, QSizePolicy,
    QFrame,
)

from .styles.theme import (
    BG_CARD, BG_SURFACE, BG_DEEP, BORDER_GOLD, BORDER_SUBTLE,
    GOLD, GOLD_DIM, GOLD_BRIGHT, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
    RADIUS, PADDING, SUCCESS, DANGER, IVORY
)
from .widgets.glow_button import LuxuryButton
from ..core.config_manager import ConfigManager


class SettingsPage(QWidget):
    """
    Configuration editor with grouped settings and hot-reload.

    Signals:
        settings_applied  — emitted after Save & Apply

    Live Visual Signals:
        analytics_toggled(bool)
        scanlines_changed(float)
    """

    settings_applied = Signal()
    analytics_toggled = Signal(bool)
    scanlines_changed = Signal(float)

    def __init__(self, config: ConfigManager, parent=None):
        super().__init__(parent)
        self._config = config
        self._inputs: dict = {}

        self._setup_ui()
        self._load_from_config()

    def _setup_ui(self):
        # Scrollable wrapper
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(50, 36, 50, 36)
        layout.setSpacing(24)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # ── Title ───────────────────────────────────────────────────────
        title = QLabel("SYSTEM CONFIGURATION")
        title.setStyleSheet(f"""
            font-size: 18px;
            font-weight: 700;
            letter-spacing: 5px;
            color: {GOLD};
            padding-bottom: 4px;
            font-family: 'Georgia', serif;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Adjust parameters to fine-tune the recognition pipeline")
        subtitle.setStyleSheet(f"""
            font-size: 11px;
            color: {TEXT_DIM};
            letter-spacing: 2px;
            padding-bottom: 8px;
        """)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        # ── Display & Visual Group ───────────────────────────────────────
        visuals_group = QGroupBox("DISPLAY  ·  VISUAL PREFERENCES")
        visuals_grid = QGridLayout(visuals_group)
        visuals_grid.setSpacing(14)

        self._inputs["analytics_mode"] = self._add_checkbox(
            visuals_grid, 0, "Analytics Panel",
            "Show the real-time analytics and logging panels alongside the camera"
        )
        self._inputs["analytics_mode"].toggled.connect(self.analytics_toggled.emit)

        self._inputs["scanline_opacity"] = self._add_double_spinbox(
            visuals_grid, 1, "Scanline Opacity", 0.0, 1.0, 0.15, 0.05,
            "Visibility of the subtle display scanline texture"
        )
        self._inputs["scanline_opacity"].valueChanged.connect(self.scanlines_changed.emit)

        layout.addWidget(visuals_group)

        # ── Camera Group ────────────────────────────────────────────────
        camera_group = QGroupBox("CAMERA  ·  INPUT DEVICE")
        camera_grid = QGridLayout(camera_group)
        camera_grid.setSpacing(14)

        self._inputs["camera_index"] = self._add_spinbox(
            camera_grid, 0, "Camera Index", 0, 10, 0,
            "OpenCV device index (0 = default webcam)"
        )
        self._inputs["fps_limit"] = self._add_spinbox(
            camera_grid, 1, "FPS Limit", 1, 120, 30,
            "Maximum capture frames per second"
        )
        layout.addWidget(camera_group)

        # ── Pipeline Group ──────────────────────────────────────────────
        pipeline_group = QGroupBox("PIPELINE  ·  INFERENCE ENGINE")
        pipeline_grid = QGridLayout(pipeline_group)
        pipeline_grid.setSpacing(14)

        self._inputs["deque_length"] = self._add_spinbox(
            pipeline_grid, 0, "Deque Length", 16, 512, 128,
            "Number of landmark frames in the rolling buffer"
        )
        self._inputs["inference_interval"] = self._add_spinbox(
            pipeline_grid, 1, "Inference Interval", 1, 60, 5,
            "Run ML model every N frames"
        )
        self._inputs["confidence_threshold"] = self._add_double_spinbox(
            pipeline_grid, 2, "Confidence Threshold", 0.0, 1.0, 0.30, 0.05,
            "Minimum confidence required to add a word to the sentence"
        )
        layout.addWidget(pipeline_group)

        # ── Action buttons ──────────────────────────────────────────────
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)
        btn_layout.addStretch()

        reset_btn = LuxuryButton("Reset Defaults", accent_color=GOLD_DIM)
        reset_btn.clicked.connect(self._on_reset)
        btn_layout.addWidget(reset_btn)

        apply_btn = LuxuryButton("Save & Apply", accent_color=GOLD)
        apply_btn.clicked.connect(self._on_apply)
        btn_layout.addWidget(apply_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        layout.addStretch()
        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ── Widget factories ────────────────────────────────────────────────

    def _add_spinbox(
        self, grid, row, label, min_val, max_val, default, tooltip=""
    ) -> QSpinBox:
        lbl = QLabel(label)
        lbl.setToolTip(tooltip)
        lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px;")
        grid.addWidget(lbl, row, 0)

        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setToolTip(tooltip)
        spin.setFixedWidth(120)
        grid.addWidget(spin, row, 1)

        return spin

    def _add_double_spinbox(
        self, grid, row, label, min_val, max_val, default, step, tooltip=""
    ) -> QDoubleSpinBox:
        lbl = QLabel(label)
        lbl.setToolTip(tooltip)
        lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px;")
        grid.addWidget(lbl, row, 0)

        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setSingleStep(step)
        spin.setDecimals(2)
        spin.setToolTip(tooltip)
        spin.setFixedWidth(120)
        grid.addWidget(spin, row, 1)

        return spin

    def _add_checkbox(self, grid, row, label, tooltip="") -> QCheckBox:
        lbl = QLabel(label)
        lbl.setToolTip(tooltip)
        lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px;")
        grid.addWidget(lbl, row, 0)

        cb = QCheckBox()
        cb.setToolTip(tooltip)
        grid.addWidget(cb, row, 1)

        return cb

    # ── Config I/O ──────────────────────────────────────────────────────

    def _load_from_config(self) -> None:
        cfg = self._config.all()

        self._inputs["camera_index"].setValue(cfg.get("camera_index", 0))
        self._inputs["fps_limit"].setValue(cfg.get("fps_limit", 30))
        self._inputs["deque_length"].setValue(cfg.get("deque_length", 128))
        self._inputs["inference_interval"].setValue(cfg.get("inference_interval", 5))
        self._inputs["confidence_threshold"].setValue(cfg.get("confidence_threshold", 0.3))

    def _collect_values(self) -> dict:
        return {
            "camera_index": self._inputs["camera_index"].value(),
            "fps_limit": self._inputs["fps_limit"].value(),
            "deque_length": self._inputs["deque_length"].value(),
            "inference_interval": self._inputs["inference_interval"].value(),
            "confidence_threshold": self._inputs["confidence_threshold"].value(),
        }

    def _on_apply(self):
        values = self._collect_values()
        self._config.update(values)
        self.settings_applied.emit()

    def _on_reset(self):
        self._config.reset_defaults()
        self._load_from_config()
