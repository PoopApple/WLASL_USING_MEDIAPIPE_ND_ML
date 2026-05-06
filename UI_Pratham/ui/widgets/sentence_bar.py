"""
SentenceBar — Scrollable sentence builder widget.

Displays predicted ASL words as styled chips in a horizontal scrollable
area.  New words are appended with a subtle slide-in animation and the
view auto-scrolls to keep the latest word visible.
"""

from PySide6.QtCore import (
    Qt, Slot, QPropertyAnimation, QEasingCurve, QTimer, Property,
)
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QFont, QLinearGradient
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QScrollArea,
    QSizePolicy, QPushButton, QFrame,
)

from ..styles.theme import (
    BG_CARD, BG_SURFACE, BORDER_GLOW, CYAN, CYAN_DIM,
    VIOLET, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
)


class WordChip(QLabel):
    """A single word chip in the sentence bar."""

    def __init__(self, word: str, parent=None):
        super().__init__(word.upper(), parent)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: rgba(0, 229, 255, 0.08);
                color: {TEXT_PRIMARY};
                border: 1px solid rgba(0, 229, 255, 0.2);
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 14px;
                font-weight: 600;
                letter-spacing: 1px;
            }}
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Fade-in animation via opacity
        self._opacity = 0.0

    def paintEvent(self, event):
        super().paintEvent(event)


class SentenceBar(QWidget):
    """
    Horizontal scrollable sentence builder.

    Layout:
        ┌────────────────────────────────────────────────────────────┐
        │  ◉ SENTENCE                            [Clear] [⌫ Undo] │
        │  ┌──────┐ ┌──────┐ ┌───────┐ ┌──────┐                   │
        │  │HELLO │ │ HOW  │ │  ARE  │ │ YOU  │  ◀── scrollable   │
        │  └──────┘ └──────┘ └───────┘ └──────┘                   │
        └────────────────────────────────────────────────────────────┘
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._chips: list[WordChip] = []
        self._clear_callback = None
        self._backspace_callback = None

        self._setup_ui()

    def set_callbacks(self, clear_fn, backspace_fn):
        """Set callbacks for clear and backspace actions."""
        self._clear_callback = clear_fn
        self._backspace_callback = backspace_fn

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 10, 20, 10)
        main_layout.setSpacing(8)

        # ── Header row ──────────────────────────────────────────────────
        header = QHBoxLayout()
        header.setSpacing(8)

        dot = QLabel("◉")
        dot.setStyleSheet(f"color: {VIOLET}; font-size: 12px;")
        header.addWidget(dot)

        title = QLabel("SENTENCE")
        title.setStyleSheet(f"""
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 3px;
            color: {TEXT_DIM};
        """)
        header.addWidget(title)
        header.addStretch()

        # Undo button
        undo_btn = QPushButton("⌫")
        undo_btn.setFixedSize(32, 28)
        undo_btn.setToolTip("Remove last word")
        undo_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 6px;
                color: {TEXT_SECONDARY};
                font-size: 14px;
            }}
            QPushButton:hover {{
                border-color: {VIOLET};
                color: {VIOLET};
            }}
        """)
        undo_btn.clicked.connect(self._on_backspace)
        header.addWidget(undo_btn)

        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedHeight(28)
        clear_btn.setToolTip("Clear entire sentence")
        clear_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 6px;
                color: {TEXT_SECONDARY};
                font-size: 11px;
                padding: 0 12px;
            }}
            QPushButton:hover {{
                border-color: {CYAN};
                color: {CYAN};
            }}
        """)
        clear_btn.clicked.connect(self._on_clear)
        header.addWidget(clear_btn)

        main_layout.addLayout(header)

        # ── Scrollable chip area ────────────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
        """)

        self._chip_container = QWidget()
        self._chip_layout = QHBoxLayout(self._chip_container)
        self._chip_layout.setContentsMargins(0, 0, 0, 0)
        self._chip_layout.setSpacing(8)
        self._chip_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._chip_layout.addStretch()

        self._scroll.setWidget(self._chip_container)
        main_layout.addWidget(self._scroll)

    # ── Slots ───────────────────────────────────────────────────────────

    @Slot(list)
    def update_sentence(self, words: list[str]) -> None:
        """
        Replace the entire sentence display with the given word list.
        Called when the pipeline emits sentence_updated.
        """
        # Clear existing chips
        for chip in self._chips:
            self._chip_layout.removeWidget(chip)
            chip.deleteLater()
        self._chips.clear()

        # Remove the stretch
        while self._chip_layout.count():
            item = self._chip_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add new chips
        for word in words:
            chip = WordChip(word)
            self._chips.append(chip)
            self._chip_layout.addWidget(chip)

        self._chip_layout.addStretch()

        # Auto-scroll to the right (latest word)
        QTimer.singleShot(50, self._scroll_to_end)

    def _scroll_to_end(self):
        bar = self._scroll.horizontalScrollBar()
        bar.setValue(bar.maximum())

    def _on_clear(self):
        if self._clear_callback:
            self._clear_callback()

    def _on_backspace(self):
        if self._backspace_callback:
            self._backspace_callback()

    # ── Glass painting ──────────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Glass background (midwam style)
        painter.setPen(QPen(QColor(255, 255, 255, 25), 1))
        painter.setBrush(QColor(255, 255, 255, 13)) # ~0.05 alpha
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 15, 15)

        painter.end()
