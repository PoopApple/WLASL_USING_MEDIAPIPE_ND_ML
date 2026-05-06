"""
TextToVideoPage — Generates an ASL sign sentence video from text input.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFrame, QScrollArea
)

from .styles.theme import (
    BG_CARD, BG_DEEP, BG_SURFACE, BG_INPUT,
    GOLD, GOLD_DIM, GOLD_BRIGHT, GOLD_SUBTLE,
    BORDER_SUBTLE, TEXT_PRIMARY, TEXT_DIM, IVORY
)


class TextToVideoPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(24)

        # ── Header ──
        header = QLabel("TEXT TO ASL VIDEO")
        header.setStyleSheet(
            f"color: {GOLD}; font-size: 18px; font-weight: 700; "
            f"letter-spacing: 5px; font-family: 'Georgia', serif;"
        )
        main_layout.addWidget(header)

        # ── Input Area ──
        input_layout = QHBoxLayout()
        input_layout.setSpacing(16)
        
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Enter sentence to generate ASL sequence...")
        self.text_input.setFixedHeight(48)
        self.text_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {BG_INPUT};
                color: {IVORY};
                border: 1px solid {BORDER_SUBTLE};
                border-radius: 6px;
                padding: 0 16px;
                font-size: 14px;
            }}
            QLineEdit:focus {{
                border-color: {GOLD};
            }}
        """)
        input_layout.addWidget(self.text_input, 1)

        generate_btn = QPushButton("GENERATE VIDEO")
        generate_btn.setFixedHeight(48)
        generate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        generate_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_SURFACE};
                color: {GOLD};
                border: 1px solid {GOLD_DIM};
                border-radius: 6px;
                padding: 0 24px;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 2px;
            }}
            QPushButton:hover {{
                background-color: rgba(212, 175, 55, 0.1);
                border-color: {GOLD_BRIGHT};
                color: {GOLD_BRIGHT};
            }}
        """)
        input_layout.addWidget(generate_btn)
        
        main_layout.addLayout(input_layout)

        # ── Video Player Placeholder ──
        self.video_player = QFrame()
        self.video_player.setStyleSheet(f"""
            QFrame {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER_SUBTLE};
                border-radius: 12px;
            }}
        """)
        vp_layout = QVBoxLayout(self.video_player)
        
        placeholder_text = QLabel("NO VIDEO GENERATED")
        placeholder_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder_text.setStyleSheet(f"""
            color: {TEXT_DIM};
            font-size: 14px;
            font-weight: 600;
            letter-spacing: 4px;
        """)
        vp_layout.addWidget(placeholder_text)
        
        main_layout.addWidget(self.video_player, 1)

        # ── Sequence Timeline ──
        timeline_label = QLabel("GENERATED SEQUENCE")
        timeline_label.setStyleSheet(
            f"color: {GOLD_DIM}; font-size: 10px; font-weight: 700; "
            f"letter-spacing: 2px;"
        )
        main_layout.addWidget(timeline_label)

        self.timeline_area = QScrollArea()
        self.timeline_area.setFixedHeight(80)
        self.timeline_area.setWidgetResizable(True)
        self.timeline_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER_SUBTLE};
                border-radius: 8px;
            }}
        """)
        
        timeline_container = QWidget()
        self.timeline_layout = QHBoxLayout(timeline_container)
        self.timeline_layout.setContentsMargins(16, 0, 16, 0)
        self.timeline_layout.setSpacing(12)
        
        timeline_placeholder = QLabel("Sequence will appear here")
        timeline_placeholder.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        self.timeline_layout.addWidget(timeline_placeholder)
        self.timeline_layout.addStretch()
        
        self.timeline_area.setWidget(timeline_container)
        main_layout.addWidget(self.timeline_area)
