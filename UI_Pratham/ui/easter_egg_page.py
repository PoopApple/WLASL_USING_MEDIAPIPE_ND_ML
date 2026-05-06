"""
EasterEggPage — Hidden credits page for the SignWave team.
Accessed by pressing the '/' key.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QLinearGradient, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFrame, QStackedWidget
)

from .styles.theme import (
    BG_DEEP, BG_CARD, GOLD, GOLD_DIM, GOLD_BRIGHT, GOLD_SUBTLE,
    TEXT_PRIMARY, TEXT_DIM, IVORY, IVORY_DIM, BORDER_SUBTLE
)


class ProfileCard(QFrame):
    """A luxury profile card for a team member."""
    def __init__(self, name, subtitle, desc, bullets, conclusion, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(60, 60, 60, 60)
        self.layout.setSpacing(16)
        
        # Name
        name_lbl = QLabel(name)
        name_lbl.setStyleSheet(
            f"color: {GOLD_BRIGHT}; font-size: 28px; font-weight: 700; "
            f"letter-spacing: 4px; font-family: 'Georgia', serif; background: transparent;"
        )
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(name_lbl)
        
        # Subtitle
        sub_lbl = QLabel(subtitle)
        sub_lbl.setStyleSheet(
            f"color: {IVORY_DIM}; font-size: 18px; font-style: italic; "
            f"font-family: 'Georgia', serif; background: transparent;"
        )
        sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(sub_lbl)
        
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {GOLD_SUBTLE};")
        self.layout.addWidget(sep)
        
        self.layout.addSpacing(10)
        
        # Description
        desc_lbl = QLabel(desc)
        desc_lbl.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: 16px; line-height: 1.6; "
            f"font-family: 'Georgia', serif; background: transparent;"
        )
        desc_lbl.setWordWrap(True)
        desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(desc_lbl)
        
        self.layout.addSpacing(10)
        
        # Bullet intro
        intro_text = "His contributions include:" if "contributions" in desc else "His work enables:" if "enables" in desc else "His work ensures:"
        intro_lbl = QLabel(intro_text)
        intro_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 16px; font-family: 'Georgia', serif; background: transparent;")
        intro_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(intro_lbl)
        
        # Bullets
        for b in bullets:
            b_lbl = QLabel(f"• {b}")
            b_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; font-size: 16px; font-family: 'Georgia', serif; background: transparent;")
            b_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.layout.addWidget(b_lbl)
            
        self.layout.addSpacing(20)
        
        # Conclusion
        conc_lbl = QLabel(conclusion)
        conc_lbl.setStyleSheet(
            f"color: {IVORY}; font-size: 18px; font-style: italic; "
            f"font-family: 'Georgia', serif; background: transparent;"
        )
        conc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(conc_lbl)
        
        self.layout.addStretch()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(20, 20, -21, -21)
        p.setBrush(QColor(BG_CARD))
        p.setPen(QPen(QColor(GOLD_DIM), 2))
        p.drawRoundedRect(r, 12, 12)
        p.setPen(QPen(QColor(BORDER_SUBTLE), 1))
        p.drawRoundedRect(r.adjusted(6, 6, -6, -6), 8, 8)
        p.end()


class EasterEggPage(QWidget):
    exit_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.stack = QStackedWidget()
        
        # ── View 0: Main Intro ──
        intro_view = QWidget()
        intro_layout = QVBoxLayout(intro_view)
        intro_layout.setContentsMargins(60, 60, 60, 60)
        intro_layout.setSpacing(16)
        
        intro_layout.addStretch()
        
        self._add_text(intro_layout, "[ SYSTEM OVERRIDE DETECTED ]\n[ AUTHORIZED ACCESS GRANTED ]", TEXT_DIM, 12, "Consolas", True)
        intro_layout.addSpacing(20)
    
        self._add_text(intro_layout, "✦ TEAM SIGN WAVE ACTIVATED ✦", GOLD_BRIGHT, 28, "Georgia", True, letter_spacing=8)
        intro_layout.addSpacing(50)
        
        self._add_text(intro_layout, "THE ARCHITECTS OF THE SYSTEM", GOLD, 20, "Georgia", True, letter_spacing=4)
        intro_layout.addSpacing(20)
        self._add_text(intro_layout, "➤ Aryan Sethi (Machine Learning)\n➤ Pratham Arora (Frontend Vision)\n➤ Suvraaj Nandwani (Backend Systems)", IVORY, 16, "Georgia", True)
        intro_layout.addSpacing(30)
        
        self._add_text(intro_layout, "[ PRESS 'A', 'P', OR 'S' TO VIEW PROFILES. PRESS '/' TO EXIT ]", TEXT_DIM, 12, "Consolas", True)
        
        intro_layout.addStretch()
        self.stack.addWidget(intro_view)
        
        # ── View 1: Aryan (A) ──
        self.stack.addWidget(ProfileCard(
            "➤ ARYAN SETHI — Machine Learning Architect",
            "The brain behind the intelligence.",
            "Aryan engineered the core learning system that powers real-time sign recognition. From designing model pipelines to optimizing prediction accuracy, he transformed raw data into intelligent understanding.",
            ["High-precision gesture detection", "Real-time inference performance", "Adaptive learning capability"],
            "He didn’t just build a model — he built the mind of the system."
        ))

        # ── View 2: Pratham (P) ──
        self.stack.addWidget(ProfileCard(
            "➤ PRATHAM ARORA — Frontend Vision Engineer",
            "The one who turned code into experience.",
            "Pratham designed and developed the entire user interface, transforming a technical system into a visually immersive and intuitive dashboard.",
            ["Full UI/UX architecture", "Cyberpunk → Luxury visual transformation", "Interactive controls and real-time feedback system"],
            "He made the system not just usable — but unforgettable."
        ))

        # ── View 3: Suvraaj (S) ──
        self.stack.addWidget(ProfileCard(
            "➤ SUVRAAJ NANDWANI — Backend Systems Engineer",
            "The backbone of the system.",
            "Suvraaj built the infrastructure that keeps everything running seamlessly. From managing data flow to ensuring system stability, he made sure every component communicates flawlessly.",
            ["Efficient data handling", "Smooth integration between ML and UI", "Reliable system performance"],
            "He ensured the system doesn’t just work — it works perfectly."
        ))
        
        main_layout.addWidget(self.stack)

    def _add_text(self, layout, text, color, size, font, center, italic=False, letter_spacing=0):
        lbl = QLabel(text)
        align = "center" if center else "left"
        italic_str = "italic" if italic else "normal"
        lbl.setStyleSheet(f"""
            color: {color};
            font-size: {size}px;
            font-family: '{font}', serif;
            font-style: {italic_str};
            letter-spacing: {letter_spacing}px;
            background: transparent;
        """)
        lbl.setWordWrap(True)
        if center:
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_A:
            self.stack.setCurrentIndex(1)
        elif key == Qt.Key.Key_P:
            self.stack.setCurrentIndex(2)
        elif key == Qt.Key.Key_S:
            self.stack.setCurrentIndex(3)
        elif key == Qt.Key.Key_Escape or key == Qt.Key.Key_Slash:
            if self.stack.currentIndex() != 0:
                self.stack.setCurrentIndex(0)
            else:
                self.exit_requested.emit()
        else:
            super().keyPressEvent(event)

    def showEvent(self, event):
        self.stack.setCurrentIndex(0)
        self.setFocus()
        super().showEvent(event)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect()
        
        # Deep luxury gradient background
        grad = QLinearGradient(r.topLeft(), r.bottomRight())
        grad.setColorAt(0, QColor(BG_DEEP))
        grad.setColorAt(0.5, QColor(15, 12, 8))
        grad.setColorAt(1, QColor(BG_DEEP))
        p.fillRect(r, grad)
        
        # Corner gold accents
        p.setPen(QPen(QColor(GOLD_DIM), 2))
        L = 40
        w, h = r.width(), r.height()
        # Top-left
        p.drawLine(10, 10, 10 + L, 10)
        p.drawLine(10, 10, 10, 10 + L)
        # Top-right
        p.drawLine(w - 10, 10, w - 10 - L, 10)
        p.drawLine(w - 10, 10, w - 10, 10 + L)
        # Bottom-left
        p.drawLine(10, h - 10, 10 + L, h - 10)
        p.drawLine(10, h - 10, 10, h - 10 - L)
        # Bottom-right
        p.drawLine(w - 10, h - 10, w - 10 - L, h - 10)
        p.drawLine(w - 10, h - 10, w - 10, h - 10 - L)
        p.end()
