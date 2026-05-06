"""
LuxuryButton — Premium gold-bordered button with warm hover glow.

A QPushButton subclass with smooth property animations that creates
a refined gold shimmer effect on hover, inspired by engraved watch buttons.
"""

from PySide6.QtCore import (
    QPropertyAnimation,
    QEasingCurve,
    Property,
    QSize,
)
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QFont, QLinearGradient
from PySide6.QtWidgets import QPushButton

from ..styles.theme import (
    GOLD, GOLD_BRIGHT, GOLD_DIM, GOLD_GLOW,
    BG_SURFACE, BG_CARD, BORDER_GOLD, TEXT_PRIMARY, IVORY
)


class LuxuryButton(QPushButton):
    """
    A premium button with animated gold glow on hover.

    The glow intensity is animated between 0.0 (no glow) and 1.0 (full
    glow) using a QPropertyAnimation, creating a warm, refined effect.
    """

    def __init__(self, text: str = "", parent=None, accent_color: str = GOLD):
        super().__init__(text, parent)
        self._glow_intensity = 0.0
        self._accent = QColor(accent_color)
        self.setMinimumHeight(44)
        self.setCursor(self.cursor())

        # Remove standard background — we custom paint
        self.setStyleSheet("background: transparent; border: none;")

        # ── Hover animation ─────────────────────────────────────────────
        self._anim_glow = QPropertyAnimation(self, b"glow_intensity")
        self._anim_glow.setDuration(300)
        self._anim_glow.setEasingCurve(QEasingCurve.Type.OutQuad)

    def _get_glow(self) -> float:
        return self._glow_intensity

    def _set_glow(self, value: float) -> None:
        self._glow_intensity = value
        self.update()

    glow_intensity = Property(float, _get_glow, _set_glow)

    # ── Events ──────────────────────────────────────────────────────────

    def enterEvent(self, event):
        self._anim_glow.stop()
        self._anim_glow.setEndValue(1.0)
        self._anim_glow.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._anim_glow.stop()
        self._anim_glow.setEndValue(0.0)
        self._anim_glow.start()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(2, 2, -2, -2)
        radius = 8

        # Background — dark glass with subtle warm gradient
        bg_grad = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        base_alpha = int(18 + 14 * self._glow_intensity)
        bg_grad.setColorAt(0, QColor(30, 25, 18, base_alpha + 10))
        bg_grad.setColorAt(1, QColor(15, 13, 10, base_alpha))
        painter.setBrush(QBrush(bg_grad))

        # Border — gold with animated brightness
        gold_alpha = int(60 + 120 * self._glow_intensity)
        border_color = QColor(self._accent)
        border_color.setAlpha(gold_alpha)
        painter.setPen(QPen(border_color, 1))
        painter.drawRoundedRect(rect, radius, radius)

        # Outer gold glow on hover
        if self._glow_intensity > 0.05:
            glow_color = QColor(self._accent)
            glow_color.setAlpha(int(25 * self._glow_intensity))
            painter.setPen(QPen(glow_color, 3))
            painter.setBrush(QBrush())
            painter.drawRoundedRect(rect.adjusted(-1, -1, 1, 1), radius + 1, radius + 1)

        # Top highlight — polished metal reflection
        highlight_alpha = int(15 + 20 * self._glow_intensity)
        painter.setPen(QPen(QColor(255, 248, 220, highlight_alpha), 1))
        painter.drawLine(
            rect.left() + radius, rect.top(),
            rect.right() - radius, rect.top()
        )

        # Text — warm gold
        text_color = QColor(GOLD_BRIGHT) if self._glow_intensity > 0.5 else QColor(GOLD)
        painter.setPen(QPen(text_color))
        font = QFont("Segoe UI", 10)
        font.setWeight(QFont.Weight.DemiBold)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.5)
        painter.setFont(font)
        painter.drawText(rect, 0x0004 | 0x0080, self.text().upper())

        painter.end()

    def sizeHint(self) -> QSize:
        return QSize(170, 48)
