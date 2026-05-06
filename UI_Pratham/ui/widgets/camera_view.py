"""
CameraView — Premium camera feed widget with luxury framing.

Displays the loading video first. Once the video ends, seamlessly
crossfades into the live camera feed. Framed with elegant gold
corner brackets and warm vignette overlay.
"""

import os
from PySide6.QtCore import Qt, Slot, QSize, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QBrush, QLinearGradient
from PySide6.QtWidgets import QLabel, QGridLayout, QWidget, QSizePolicy, QGraphicsOpacityEffect

from ..styles.theme import GOLD, GOLD_DIM, GOLD_BRIGHT, BG_DEEP, BORDER_GOLD, TEXT_DIM
from .loading_screen import LoadingScreen
from .gl_overlay import GLOverlay


class CameraView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 480)

        # ── Layout ────────────────────────────────────────────────────────
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        # ── Loading Screen ────────────────────────────────────────────────
        video_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "Frontend Assets(PyQT)",
            "Loading Screen.mp4"
        )
        self._loading_screen = LoadingScreen(video_path)
        self._layout.addWidget(self._loading_screen, 0, 0)

        # ── Camera Frame (GPU Accelerated) ────────────────────────────────
        self._gl_overlay = GLOverlay(self)

        # Setup opacity effect for crossfade
        self._opacity_effect = QGraphicsOpacityEffect(self._gl_overlay)
        self._opacity_effect.setOpacity(0.0)
        self._gl_overlay.setGraphicsEffect(self._opacity_effect)

        self._layout.addWidget(self._gl_overlay, 0, 0)

        self._has_frame = False

        # ── Animations ────────────────────────────────────────────────────
        self._fade_anim = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_anim.setDuration(1800)  # Slow, luxurious crossfade
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self._video_finished = False
        self._loading_screen.finished.connect(self._on_loading_finished)

    def start_loading(self):
        """Starts the loading video."""
        self._loading_screen.start()

    def stop_loading(self):
        """Stops the loading video if running."""
        self._loading_screen.stop()

    @Slot()
    def _on_loading_finished(self):
        """Called when video ends, begins crossfade to camera."""
        self._video_finished = True
        self._fade_anim.start()

    @Slot(QImage)
    def update_frame(self, qimage: QImage) -> None:
        if qimage.isNull():
            return
        self._has_frame = True
        self._gl_overlay.set_image(qimage)

    @Slot(float)
    def set_scanline_opacity(self, v: float):
        self._gl_overlay.set_scanline_opacity(v)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()

        if self._video_finished:
            # ── Luxury corner brackets ──
            gold = QColor(GOLD)
            gold.setAlpha(180)
            painter.setPen(QPen(gold, 1.5))

            bracket_len = 30
            w = rect.width() - 1
            h = rect.height() - 1

            # Top-left
            painter.drawLine(0, 0, bracket_len, 0)
            painter.drawLine(0, 0, 0, bracket_len)
            # Top-right
            painter.drawLine(w, 0, w - bracket_len, 0)
            painter.drawLine(w, 0, w, bracket_len)
            # Bottom-left
            painter.drawLine(0, h, bracket_len, h)
            painter.drawLine(0, h, 0, h - bracket_len)
            # Bottom-right
            painter.drawLine(w, h, w - bracket_len, h)
            painter.drawLine(w, h, w, h - bracket_len)

            # Corner diamonds — like watch hour markers
            painter.setPen(Qt.PenStyle.NoPen)
            diamond = QColor(GOLD_BRIGHT)
            diamond.setAlpha(140)
            painter.setBrush(diamond)
            # Small 4px diamonds at bracket intersections
            for cx, cy in [(4, 4), (w - 4, 4), (4, h - 4), (w - 4, h - 4)]:
                painter.save()
                painter.translate(cx, cy)
                painter.rotate(45)
                painter.drawRect(-2, -2, 4, 4)
                painter.restore()

        # Outer frame — subtle gold border
        frame_color = QColor(GOLD)
        frame_color.setAlpha(50)
        painter.setPen(QPen(frame_color, 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect.adjusted(0, 0, -1, -1), 6, 6)

        painter.end()

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def sizeHint(self) -> QSize:
        return QSize(800, 600)
