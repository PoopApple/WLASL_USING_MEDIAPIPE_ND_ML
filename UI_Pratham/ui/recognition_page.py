"""
RecognitionPage — Main monitoring view with prediction display.

Camera feed + HUD + prediction bar + sentence builder + quick actions.
"""

from PySide6.QtCore import Qt, Slot, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QFont, QLinearGradient, QBrush
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QScrollArea, QSizePolicy, QGridLayout
)

from .widgets.camera_view import CameraView
from .widgets.hud_overlay import HUDOverlay
from .widgets.analytics_panel import AnalyticsPanel
from .styles.theme import (
    BG_CARD, BG_DEEP, BG_SURFACE, BG_INPUT,
    GOLD, GOLD_DIM, GOLD_BRIGHT, GOLD_SUBTLE,
    BORDER_SUBTLE, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
    IVORY, IVORY_DIM, SUCCESS
)


class PredictionBar(QWidget):
    """Shows the current predicted letter/word with confidence."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(80)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(24, 0, 24, 0)
        lay.setSpacing(20)

        # Current letter
        self._letter = QLabel("—")
        self._letter.setFixedWidth(60)
        self._letter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._letter.setStyleSheet(
            f"color: {GOLD_BRIGHT}; font-size: 36px; font-weight: 700; "
            f"font-family: 'Georgia', serif; background: transparent;"
        )
        lay.addWidget(self._letter)

        # Separator
        sep = QFrame()
        sep.setFixedWidth(1)
        sep.setFixedHeight(40)
        sep.setStyleSheet(f"background: {GOLD_SUBTLE};")
        lay.addWidget(sep)

        # Word / Confidence
        info = QVBoxLayout()
        info.setSpacing(2)
        self._word_label = QLabel("Awaiting Detection...")
        self._word_label.setStyleSheet(
            f"color: {IVORY}; font-size: 16px; font-weight: 600; "
            f"letter-spacing: 2px; background: transparent;"
        )
        self._conf_label = QLabel("Confidence: —")
        self._conf_label.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 11px; letter-spacing: 1px; background: transparent;"
        )
        info.addWidget(self._word_label)
        info.addWidget(self._conf_label)
        lay.addLayout(info)

        lay.addStretch()

        # Status dot
        self._status = QLabel("● IDLE")
        self._status.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 10px; letter-spacing: 2px; background: transparent;"
        )
        lay.addWidget(self._status)

    def update_prediction(self, letter, word, conf):
        self._letter.setText(letter)
        self._word_label.setText(word.upper() if word else "—")
        self._conf_label.setText(f"Confidence: {conf:.0%}")
        self._status.setText("● ACTIVE")
        self._status.setStyleSheet(
            f"color: {SUCCESS}; font-size: 10px; letter-spacing: 2px; background: transparent;"
        )

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(0, 0, -1, -1)
        p.setBrush(QColor(BG_CARD))
        p.setPen(QPen(QColor(BORDER_SUBTLE), 1))
        p.drawRoundedRect(r, 10, 10)
        # Gold top accent
        gold = QColor(GOLD)
        gold.setAlpha(60)
        p.setPen(QPen(gold, 1))
        p.drawLine(r.left() + 14, r.top(), r.right() - 14, r.top())
        p.end()


class SentenceBar(QWidget):
    """Horizontal sentence builder showing detected words as chips."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(52)
        self._words = []

        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(8)

        icon = QLabel("◇")
        icon.setStyleSheet(f"color: {GOLD_DIM}; font-size: 14px; background: transparent;")
        lay.addWidget(icon)

        title = QLabel("SENTENCE")
        title.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 9px; font-weight: 700; "
            f"letter-spacing: 2px; background: transparent;"
        )
        lay.addWidget(title)

        sep = QFrame()
        sep.setFixedWidth(1)
        sep.setFixedHeight(24)
        sep.setStyleSheet(f"background: {GOLD_SUBTLE};")
        lay.addWidget(sep)

        self._text_label = QLabel("—")
        self._text_label.setStyleSheet(
            f"color: {IVORY}; font-size: 14px; font-weight: 500; "
            f"letter-spacing: 1px; background: transparent;"
        )
        lay.addWidget(self._text_label, 1)

        # Clear button
        clear = QPushButton("CLEAR")
        clear.setFixedHeight(28)
        clear.setStyleSheet(
            f"QPushButton {{ background: transparent; border: 1px solid {BORDER_SUBTLE}; "
            f"border-radius: 4px; color: {TEXT_DIM}; font-size: 9px; letter-spacing: 1.5px; padding: 0 10px; }}"
            f"QPushButton:hover {{ border-color: {GOLD}; color: {GOLD}; }}"
        )
        clear.clicked.connect(self._clear)
        lay.addWidget(clear)

    def add_word(self, word):
        self._words.append(word.upper())
        if len(self._words) > 15:
            self._words.pop(0)
        self._text_label.setText("  ".join(self._words))

    def _clear(self):
        self._words.clear()
        self._text_label.setText("—")

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(0, 0, -1, -1)
        p.setBrush(QColor(BG_CARD))
        p.setPen(QPen(QColor(BORDER_SUBTLE), 1))
        p.drawRoundedRect(r, 8, 8)
        p.end()


class QuickActionsBar(QWidget):
    """Row of quick-action buttons."""
    fullscreen_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(42)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        actions = [
            ("COPY TEXT", "📋"), ("EXPORT CSV", "📁"), 
            ("SCREENSHOT", "📷"), ("RESET", "↺"), ("FULLSCREEN", "⛶"),
        ]
        for label, icon in actions:
            btn = QPushButton(f"{icon}  {label}")
            btn.setFixedHeight(36)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(
                f"QPushButton {{ background: {BG_CARD}; border: 1px solid {BORDER_SUBTLE}; "
                f"border-radius: 6px; color: {TEXT_DIM}; font-size: 9px; "
                f"letter-spacing: 1px; padding: 0 12px; }}"
                f"QPushButton:hover {{ border-color: {GOLD}; color: {GOLD}; background: {BG_SURFACE}; }}"
            )
            if label == "FULLSCREEN":
                btn.clicked.connect(self.fullscreen_clicked.emit)
            lay.addWidget(btn)
        lay.addStretch()


class GestureReference(QWidget):
    """Mini reference panel showing ASL alphabet guide."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(60)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 8, 20, 8)
        lay.setSpacing(4)

        title = QLabel("ASL GUIDE  ·")
        title.setStyleSheet(
            f"color: {GOLD_DIM}; font-size: 9px; font-weight: 700; "
            f"letter-spacing: 2px; background: transparent; padding-right: 8px;"
        )
        lay.addWidget(title)

        for letter in "ABCDEFGHIJKLM":
            lbl = QLabel(letter)
            lbl.setFixedSize(28, 28)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(
                f"color: {TEXT_DIM}; font-size: 10px; font-weight: 600; "
                f"background: {BG_SURFACE}; border: 1px solid {BORDER_SUBTLE}; border-radius: 4px;"
            )
            lay.addWidget(lbl)
        lay.addStretch()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor(BG_CARD))
        p.setPen(QPen(QColor(BORDER_SUBTLE), 1))
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 8, 8)
        p.end()


class WordSuggestions(QWidget):
    """Shows predicted next-word suggestions."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(38)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(8)

        hint = QLabel("SUGGESTIONS:")
        hint.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 9px; letter-spacing: 1.5px; background: transparent;"
        )
        lay.addWidget(hint)

        suggestions = ["HELLO", "THANK YOU", "PLEASE", "HELP", "YES"]
        for word in suggestions:
            chip = QPushButton(word)
            chip.setFixedHeight(26)
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.setStyleSheet(
                f"QPushButton {{ background: {BG_SURFACE}; border: 1px solid {GOLD_SUBTLE}; "
                f"border-radius: 13px; color: {IVORY_DIM}; font-size: 10px; "
                f"font-weight: 600; letter-spacing: 1px; padding: 0 12px; }}"
                f"QPushButton:hover {{ border-color: {GOLD}; color: {GOLD}; }}"
            )
            lay.addWidget(chip)
        lay.addStretch()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor(BG_CARD))
        p.setPen(QPen(QColor(BORDER_SUBTLE), 1))
        p.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 8, 8)
        p.end()


class RecognitionPage(QWidget):
    fullscreen_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 16)
        main_layout.setSpacing(10)

        # ── Row 1: Camera + Analytics ──
        self.center_layout = QHBoxLayout()
        self.center_layout.setSpacing(16)

        self.camera_view = CameraView()
        self.center_layout.addWidget(self.camera_view, stretch=10)

        self.analytics_panel = AnalyticsPanel()
        self.center_layout.addWidget(self.analytics_panel, stretch=0)
        self.analytics_panel.hide()

        main_layout.addLayout(self.center_layout, 1)

        # ── Row 2: Prediction Bar ──
        self.prediction_bar = PredictionBar()
        main_layout.addWidget(self.prediction_bar)

        # ── Row 3: Sentence Builder ──
        self.sentence_bar = SentenceBar()
        main_layout.addWidget(self.sentence_bar)

        # ── Row 4: Word Suggestions ──
        self.word_suggestions = WordSuggestions()
        main_layout.addWidget(self.word_suggestions)

        # ── Row 5: Quick Actions ──
        self.quick_actions = QuickActionsBar()
        self.quick_actions.fullscreen_clicked.connect(self.fullscreen_requested.emit)
        main_layout.addWidget(self.quick_actions)

        # ── Row 6: Gesture Reference ──
        self.gesture_ref = GestureReference()
        main_layout.addWidget(self.gesture_ref)

        # HUD overlay
        self.hud_overlay = HUDOverlay(self.camera_view)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cam_rect = self.camera_view.rect()
        overlay_x = cam_rect.width() - self.hud_overlay.width() - 15
        self.hud_overlay.move(max(0, overlay_x), 15)
        self.hud_overlay.raise_()

    @Slot(bool)
    def set_analytics_mode(self, enabled):
        if enabled:
            self.center_layout.setStretch(0, 4)
            self.center_layout.setStretch(1, 6)
            self.analytics_panel.animate_in()
        else:
            self.center_layout.setStretch(0, 10)
            self.center_layout.setStretch(1, 0)
            self.analytics_panel.hide()
