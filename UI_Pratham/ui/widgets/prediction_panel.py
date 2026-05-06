"""
PredictionPanel — Floating glassmorphic panel for real-time predictions.

Displays the current ASL prediction with an animated confidence bar,
and optionally shows top-K beam search results.  The panel toggles
between single-prediction and beam-search modes.
"""

from PySide6.QtCore import (
    Qt, Slot, QPropertyAnimation, QEasingCurve, Property, QTimer,
)
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QFont, QLinearGradient
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QSizePolicy, QFrame,
)

from ..styles.theme import (
    BG_GLASS, BG_CARD, BG_SURFACE, BORDER_GLOW,
    CYAN, CYAN_DIM, CYAN_GLOW, VIOLET, VIOLET_DIM,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
    RADIUS, PADDING,
)


class ConfidenceBar(QWidget):
    """Animated horizontal bar showing prediction confidence (0–100%)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._display_value = 0.0
        self.setFixedHeight(8)
        self.setMinimumWidth(100)

        self._anim = QPropertyAnimation(self, b"display_value")
        self._anim.setDuration(400)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _get_display_value(self) -> float:
        return self._display_value

    def _set_display_value(self, v: float):
        self._display_value = v
        self.update()

    display_value = Property(float, _get_display_value, _set_display_value)

    def set_value(self, value: float):
        """Animate to a new confidence value (0.0–1.0)."""
        self._value = max(0.0, min(1.0, value))
        self._anim.stop()
        self._anim.setStartValue(self._display_value)
        self._anim.setEndValue(self._value)
        self._anim.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        r = h / 2

        # Background track
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(30, 30, 50))
        painter.drawRoundedRect(0, 0, w, h, r, r)

        # Filled portion with gradient
        fill_w = max(0, int(w * self._display_value))
        if fill_w > 0:
            grad = QLinearGradient(0, 0, fill_w, 0)
            grad.setColorAt(0, QColor(CYAN_DIM))
            grad.setColorAt(1, QColor(CYAN))
            painter.setBrush(grad)
            painter.drawRoundedRect(0, 0, fill_w, h, r, r)

        painter.end()


class PredictionPanel(QWidget):
    """
    Floating panel showing current prediction and optional beam search.

    Layout:
        ┌─────────────────────────────────┐
        │  ◉ PREDICTION          [toggle] │
        │                                 │
        │       HELLO                     │
        │   ████████████░░░  87%          │
        │                                 │
        │  ─── Top Predictions ───        │
        │  1. HELLO     87%               │
        │  2. HELP      5%                │
        │  3. HI        3%                │
        │  ...                            │
        └─────────────────────────────────┘
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(280)
        self.setMaximumWidth(420)

        self._beam_search_visible = False
        self._top_k_widgets: list[tuple] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # ── Header ──────────────────────────────────────────────────────
        header = QHBoxLayout()
        header.setSpacing(8)

        dot = QLabel("◉")
        dot.setStyleSheet(f"color: {CYAN}; font-size: 14px;")
        header.addWidget(dot)

        title = QLabel("PREDICTION")
        title.setStyleSheet(f"""
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 3px;
            color: {TEXT_SECONDARY};
        """)
        header.addWidget(title)
        header.addStretch()

        # Beam search toggle
        from .toggle_switch import LabeledToggle
        self._beam_toggle = LabeledToggle("Beam Search")
        self._beam_toggle.toggle.toggled.connect(self._on_beam_toggled)
        header.addWidget(self._beam_toggle)

        layout.addLayout(header)

        # ── Separator ──────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background-color: rgba(0, 229, 255, 0.1); max-height: 1px;")
        layout.addWidget(sep)

        # ── Main prediction ─────────────────────────────────────────────
        self._word_label = QLabel("—")
        self._word_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._word_label.setStyleSheet(f"""
            font-size: 42px;
            font-weight: 700;
            color: {CYAN};
            padding: 16px 0;
            letter-spacing: 2px;
        """)
        layout.addWidget(self._word_label)

        # ── Confidence bar + percentage ─────────────────────────────────
        conf_layout = QHBoxLayout()
        conf_layout.setSpacing(12)

        self._conf_bar = ConfidenceBar()
        conf_layout.addWidget(self._conf_bar, stretch=1)

        self._conf_label = QLabel("0%")
        self._conf_label.setStyleSheet(f"""
            font-size: 14px;
            font-weight: 600;
            color: {TEXT_SECONDARY};
            min-width: 45px;
        """)
        self._conf_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        conf_layout.addWidget(self._conf_label)

        layout.addLayout(conf_layout)

        # ── Separator 2 ────────────────────────────────────────────────
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"background-color: rgba(0, 229, 255, 0.06); max-height: 1px;")
        layout.addWidget(sep2)

        # ── Top-K predictions container ────────────────────────────────
        self._topk_title = QLabel("TOP PREDICTIONS")
        self._topk_title.setStyleSheet(f"""
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 2px;
            color: {TEXT_DIM};
            padding-top: 4px;
        """)
        layout.addWidget(self._topk_title)

        self._topk_container = QVBoxLayout()
        self._topk_container.setSpacing(6)
        layout.addLayout(self._topk_container)

        # Create 5 slots for top-K predictions
        for i in range(5):
            row = QHBoxLayout()
            row.setSpacing(8)

            rank = QLabel(f"{i + 1}.")
            rank.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px; min-width: 20px;")
            rank.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(rank)

            word = QLabel("—")
            word.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px; font-weight: 500;")
            row.addWidget(word, stretch=1)

            conf = QLabel("")
            conf.setStyleSheet(f"color: {TEXT_DIM}; font-size: 12px; min-width: 40px;")
            conf.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            row.addWidget(conf)

            self._topk_container.addLayout(row)
            self._top_k_widgets.append((word, conf))

        layout.addStretch()

        # Initially show top-K section
        self._set_topk_visible(True)

    # ── Slots ───────────────────────────────────────────────────────────

    @Slot(str, float, list)
    def update_prediction(self, word: str, confidence: float, top_k: list) -> None:
        """
        Update the displayed prediction.

        Args:
            word:       Predicted ASL word
            confidence: Confidence score (0.0–1.0)
            top_k:      List of (word, confidence) tuples
        """
        self._word_label.setText(word.upper())
        self._conf_bar.set_value(confidence)
        self._conf_label.setText(f"{confidence * 100:.0f}%")

        # Update top-K list
        for i, (word_label, conf_label) in enumerate(self._top_k_widgets):
            if i < len(top_k):
                w, c = top_k[i]
                word_label.setText(w.upper())
                conf_label.setText(f"{c * 100:.1f}%")

                # Highlight top prediction
                if i == 0:
                    word_label.setStyleSheet(f"color: {CYAN}; font-size: 13px; font-weight: 600;")
                else:
                    word_label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px; font-weight: 500;")
            else:
                word_label.setText("—")
                conf_label.setText("")

    def set_beam_search(self, enabled: bool) -> None:
        """Programmatically set the beam search toggle."""
        self._beam_toggle.setChecked(enabled)

    def _on_beam_toggled(self, checked: bool):
        self._beam_search_visible = checked
        self._set_topk_visible(True)  # always show top-K, label changes
        self._topk_title.setText(
            "BEAM SEARCH RESULTS" if checked else "TOP PREDICTIONS"
        )

    def _set_topk_visible(self, visible: bool):
        self._topk_title.setVisible(visible)
        for word_lbl, conf_lbl in self._top_k_widgets:
            word_lbl.setVisible(visible)
            conf_lbl.setVisible(visible)

    # ── Glass painting ──────────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Glass background
        painter.setPen(QPen(QColor(255, 255, 255, 25), 1))
        painter.setBrush(QColor(255, 255, 255, 13))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 20, 20)

        # Top glow accent line
        grad = QLinearGradient(0, 0, self.width(), 0)
        grad.setColorAt(0, QColor(0, 229, 255, 0))
        grad.setColorAt(0.3, QColor(0, 229, 255, 60))
        grad.setColorAt(0.7, QColor(179, 136, 255, 60))
        grad.setColorAt(1, QColor(179, 136, 255, 0))
        painter.setPen(QPen(QBrush(grad), 2))
        painter.drawLine(16, 1, self.width() - 16, 1)

        painter.end()
