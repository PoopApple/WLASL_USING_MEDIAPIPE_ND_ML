"""
HUDOverlay — Premium luxury heads-up display for camera metrics.

Positioned in the corner of the camera view, it displays FPS and
environmental luminance. Styled like a luxury watch complication —
subtle, engraved, and authoritative.
"""

from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QFont, QLinearGradient, QBrush
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from ..styles.theme import (
    GOLD, GOLD_DIM, GOLD_BRIGHT, DANGER,
    TEXT_PRIMARY, TEXT_SECONDARY, BG_CARD, BORDER_GOLD, BORDER_SUBTLE
)


class HUDOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(190, 84)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self._fps = 0.0
        self._lumens = 0.0
        self._threshold = 80.0

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)

        style_base = """
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 1px;
            background: transparent;
        """

        self._fps_label = QLabel("FPS: --")
        self._fps_label.setStyleSheet(style_base + f"color: {TEXT_PRIMARY};")
        layout.addWidget(self._fps_label)

        self._lumens_label = QLabel("LUMENS: --")
        self._lumens_label.setStyleSheet(style_base + f"color: {GOLD};")
        layout.addWidget(self._lumens_label)

    @Slot(float)
    def update_fps(self, fps: float) -> None:
        self._fps = fps
        self._fps_label.setText(f"FPS: {fps:.1f}")

    @Slot(float)
    def update_lumens(self, lumens: float) -> None:
        self._lumens = lumens
        if lumens < self._threshold:
            text = f"LUMENS: {lumens:.1f}  ·  LOW"
            color = DANGER
        else:
            text = f"LUMENS: {lumens:.1f}  ·  OPTIMAL"
            color = GOLD

        self._lumens_label.setText(text)
        self._lumens_label.setStyleSheet(
            f"font-size: 11px; font-weight: 600; "
            f"letter-spacing: 1px; background: transparent; color: {color};"
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(0, 0, -1, -1)

        # Dark glass background with warm tint
        painter.setBrush(QColor(12, 10, 8, 220))
        painter.setPen(QPen(QColor(GOLD_DIM), 0.8))
        painter.drawRoundedRect(rect, 8, 8)

        # Top edge — polished gold accent line
        gold = QColor(GOLD)
        gold.setAlpha(80)
        painter.setPen(QPen(gold, 1))
        painter.drawLine(rect.left() + 10, rect.top(), rect.right() - 10, rect.top())

        # Corner ornaments — tiny gold dots (watch dial markers)
        painter.setPen(Qt.PenStyle.NoPen)
        dot_color = QColor(GOLD)
        dot_color.setAlpha(120)
        painter.setBrush(dot_color)
        painter.drawEllipse(rect.left() + 6, rect.top() + 6, 3, 3)
        painter.drawEllipse(rect.right() - 9, rect.top() + 6, 3, 3)

        painter.end()
