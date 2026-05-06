"""
ToggleSwitch — Premium gold toggle switch.

A refined, animated toggle inspired by luxury watch crown selectors.
"""

from PySide6.QtCore import Qt, QPropertyAnimation, Property, QEasingCurve, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QLinearGradient
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel

from ..styles.theme import (
    BG_SURFACE, BG_CARD, GOLD, GOLD_DIM, GOLD_BRIGHT,
    TEXT_PRIMARY, TEXT_SECONDARY, BORDER_GOLD, BORDER_SUBTLE
)


class ToggleSwitch(QWidget):
    """
    A luxury animated toggle switch with gold accents.
    """
    toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(46, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._checked = False
        self._position = 2  # thumb X position

        # Animation for thumb movement
        self._anim = QPropertyAnimation(self, b"position", self)
        self._anim.setDuration(250)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

    def _get_position(self):
        return self._position

    def _set_position(self, pos):
        self._position = pos
        self.update()

    position = Property(float, _get_position, _set_position)

    def isChecked(self):
        return self._checked

    def setChecked(self, checked: bool):
        if self._checked == checked:
            return
        self._checked = checked
        self._anim.stop()
        self._anim.setEndValue(22 if checked else 2)
        self._anim.start()
        self.toggled.emit(checked)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setChecked(not self._checked)

    def toggle_state(self):
        self.setChecked(not self._checked)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        r = h / 2

        # Track — dark with gold border when active
        if self._checked:
            track_color = QColor(GOLD_DIM)
            track_color.setAlpha(120)
            border_color = QColor(GOLD)
            border_color.setAlpha(160)
        else:
            track_color = QColor(BG_SURFACE)
            border_color = QColor(BORDER_SUBTLE)

        painter.setBrush(QBrush(track_color))
        painter.setPen(QPen(border_color, 1))
        painter.drawRoundedRect(0, 0, w, h, r, r)

        # Thumb — gold metallic gradient when active, ivory when off
        thumb_r = h - 4
        thumb_x = int(self._position)

        thumb_grad = QLinearGradient(thumb_x, 2, thumb_x, 2 + thumb_r)
        if self._checked:
            thumb_grad.setColorAt(0, QColor(GOLD_BRIGHT))
            thumb_grad.setColorAt(1, QColor(GOLD))
        else:
            thumb_grad.setColorAt(0, QColor(200, 195, 180))
            thumb_grad.setColorAt(1, QColor(160, 155, 140))

        painter.setBrush(QBrush(thumb_grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(thumb_x, 2, int(thumb_r), int(thumb_r))

        painter.end()


class LabeledToggle(QWidget):
    """A wrapper combining a label and the ToggleSwitch."""
    def __init__(self, label_text: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label_text)
        self.label.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13px;")

        self.toggle = ToggleSwitch()

        layout.addWidget(self.label)
        layout.addStretch()
        layout.addWidget(self.toggle)

    def isChecked(self):
        return self.toggle.isChecked()

    def setChecked(self, checked):
        self.toggle.setChecked(checked)
