"""
AnalyticsPanel — Expanded instrument dashboard with 12+ cards.
"""

from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, Slot, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QScrollArea, QGraphicsOpacityEffect, QGridLayout, QProgressBar
)
from PySide6.QtGui import (
    QColor, QPainter, QPen, QBrush, QLinearGradient,
    QConicalGradient, QFont
)
import random, math, time

from ..styles.theme import (
    BG_CARD, BG_DEEP, BG_SURFACE, BG_INPUT,
    GOLD, GOLD_DIM, GOLD_BRIGHT, GOLD_SUBTLE,
    BORDER_GOLD, BORDER_SUBTLE,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_DIM,
    IVORY, IVORY_DIM, SUCCESS, DANGER
)

import pyqtgraph as pg


# ── Instrument Card Base ──
class InstrumentCard(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0.0)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 14, 16, 14)
        self.layout.setSpacing(8)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(
            f"color: {GOLD}; font-size: 10px; font-weight: 700; "
            f"letter-spacing: 2.5px; border: none; background: transparent; "
            f"font-family: 'Georgia', serif;"
        )
        self.layout.addWidget(self.title_label)
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet(f"background-color: {GOLD_SUBTLE}; border: none;")
        self.layout.addWidget(line)

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(0, 0, -1, -1)
        p.setBrush(QColor(BG_CARD))
        p.setPen(QPen(QColor(BORDER_SUBTLE), 1))
        p.drawRoundedRect(r, 10, 10)
        gold = QColor(GOLD)
        gold.setAlpha(100)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(gold)
        p.drawRoundedRect(r.left(), r.top() + 12, 3, r.height() - 24, 1, 1)
        p.end()


# ── Radial Gauge Widget ──
class RadialGauge(QWidget):
    def __init__(self, label="", size=90, parent=None):
        super().__init__(parent)
        self._label = label
        self._value = 0.0
        self._display = 0.0
        self.setFixedSize(size, size)
        t = QTimer(self)
        t.timeout.connect(self._animate)
        t.start(30)

    def set_value(self, v):
        self._value = max(0.0, min(1.0, v))

    def _animate(self):
        self._display += (self._value - self._display) * 0.12
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = self.width() // 2, self.height() // 2
        r = min(cx, cy) - 8

        # Track
        p.setPen(QPen(QColor(BG_SURFACE), 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
        p.drawArc(cx - r, cy - r, 2 * r, 2 * r, 0, 360 * 16)

        # Value arc
        span = int(-360 * 16 * self._display)
        if span != 0:
            grad = QConicalGradient(cx, cy, 90)
            grad.setColorAt(0.0, QColor(GOLD))
            grad.setColorAt(0.5, QColor(GOLD_BRIGHT))
            grad.setColorAt(1.0, QColor(GOLD))
            p.setPen(QPen(QBrush(grad), 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(cx - r, cy - r, 2 * r, 2 * r, 90 * 16, span)

        # Center text
        p.setPen(QColor(IVORY))
        p.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        p.drawText(self.rect().adjusted(0, -8, 0, 0), Qt.AlignmentFlag.AlignCenter, f"{int(self._display * 100)}")
        p.setFont(QFont("Georgia", 6))
        p.setPen(QColor(GOLD_DIM))
        p.drawText(self.rect().adjusted(0, 18, 0, 0), Qt.AlignmentFlag.AlignCenter, self._label.upper())
        p.end()


# ── Letter Heatmap ──
class LetterHeatmap(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.scores = {l: random.uniform(0, 0.3) for l in self.letters}
        self.setFixedHeight(100)
        t = QTimer(self)
        t.timeout.connect(self._sim)
        t.start(1200)

    def _sim(self):
        for l in self.letters:
            self.scores[l] = max(0, min(1, self.scores[l] + random.uniform(-0.04, 0.04)))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect()
        p.fillRect(r, QColor(BG_DEEP))
        cols = 13
        rows = 2
        cw = r.width() / cols
        ch = (r.height() - 4) / rows
        pad = 2
        for i, letter in enumerate(self.letters):
            col, row = i % cols, i // cols
            x, y = int(col * cw + pad), int(row * ch + pad + 2)
            w, h = int(cw - pad * 2), int(ch - pad * 2)
            score = self.scores.get(letter, 0)
            alpha = int(40 + 200 * score)
            c = QColor(GOLD)
            c.setAlpha(alpha)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(c)
            p.drawRoundedRect(x, y, w, h, 3, 3)
            p.setPen(QColor(IVORY) if score > 0.5 else QColor(TEXT_DIM))
            p.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            from PySide6.QtCore import QRect
            p.drawText(QRect(x, y, w, h), Qt.AlignmentFlag.AlignCenter, letter)
        p.end()


# ── Mini Stat Label ──
def _stat_label(title, value, color=GOLD):
    w = QWidget()
    w.setStyleSheet("background: transparent;")
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)
    t = QLabel(title)
    t.setStyleSheet(f"color: {TEXT_DIM}; font-size: 8px; letter-spacing: 1.5px; background: transparent;")
    v = QLabel(str(value))
    v.setObjectName("val")
    v.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold; background: transparent;")
    lay.addWidget(t)
    lay.addWidget(v)
    return w


# ── Main Analytics Panel ──
class AnalyticsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(420)
        self._cards = []
        self._setup_ui()
        self._start_sim()
        self.hide()

    @Slot()
    def animate_in(self):
        self.show()
        for i, card in enumerate(self._cards):
            card.opacity_effect.setOpacity(0.0)
            fade = QPropertyAnimation(card.opacity_effect, b"opacity")
            fade.setDuration(400 + i * 80)
            fade.setStartValue(0.0)
            fade.setEndValue(1.0)
            fade.setEasingCurve(QEasingCurve.Type.OutQuad)
            fade.start()
            setattr(card, f"_anim_{i}", fade)

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        container = QWidget()
        self.cl = QVBoxLayout(container)
        self.cl.setContentsMargins(8, 0, 8, 0)
        self.cl.setSpacing(12)

        pg.setConfigOption('antialias', True)
        pg.setConfigOption('background', 'transparent')
        pg.setConfigOption('foreground', TEXT_SECONDARY)

        # 1. SESSION STATS
        c1 = InstrumentCard("SESSION STATISTICS")
        self._cards.append(c1)
        stats_row = QHBoxLayout()
        stats_row.setSpacing(8)
        self._stat_detections = _stat_label("DETECTIONS", "0")
        self._stat_accuracy = _stat_label("ACCURACY", "—")
        self._stat_wpm = _stat_label("WORDS/MIN", "0")
        self._stat_uptime = _stat_label("UPTIME", "00:00")
        for s in [self._stat_detections, self._stat_accuracy, self._stat_wpm, self._stat_uptime]:
            stats_row.addWidget(s)
        c1.layout.addLayout(stats_row)
        self.cl.addWidget(c1)

        # 2. CONFIDENCE GAUGES
        c2 = InstrumentCard("CONFIDENCE INSTRUMENTS")
        self._cards.append(c2)
        gauge_row = QHBoxLayout()
        gauge_row.setSpacing(4)
        self._g_conf = RadialGauge("CONF", 85)
        self._g_acc = RadialGauge("ACC", 85)
        self._g_speed = RadialGauge("SPEED", 85)
        self._g_model = RadialGauge("MODEL", 85)
        for g in [self._g_conf, self._g_acc, self._g_speed, self._g_model]:
            gauge_row.addWidget(g, alignment=Qt.AlignmentFlag.AlignCenter)
        c2.layout.addLayout(gauge_row)
        self.cl.addWidget(c2)

        # 3. FPS PERFORMANCE GRAPH
        c3 = InstrumentCard("FRAME RATE MONITOR")
        self._cards.append(c3)
        self.plot_fps = pg.PlotWidget()
        self.plot_fps.setFixedHeight(120)
        self.plot_fps.getAxis('left').setPen(pg.mkPen(color=TEXT_DIM, width=1))
        self.plot_fps.getAxis('bottom').setPen(pg.mkPen(color=TEXT_DIM, width=1))
        self.plot_fps.showGrid(x=False, y=True, alpha=0.04)
        self.fps_data = []
        self.fps_curve = self.plot_fps.plot(pen=pg.mkPen(color=GOLD, width=2), fillLevel=0, fillBrush=QColor(212, 175, 55, 25))
        self.fps_glow = self.plot_fps.plot(pen=pg.mkPen(color=QColor(212, 175, 55, 50), width=5))
        c3.layout.addWidget(self.plot_fps)
        self.cl.addWidget(c3)

        # 4. LATENCY MONITOR
        c4 = InstrumentCard("INFERENCE LATENCY")
        self._cards.append(c4)
        self.plot_latency = pg.PlotWidget()
        self.plot_latency.setFixedHeight(100)
        self.plot_latency.getAxis('left').setPen(pg.mkPen(color=TEXT_DIM, width=1))
        self.plot_latency.getAxis('bottom').setPen(pg.mkPen(color=TEXT_DIM, width=1))
        self.plot_latency.showGrid(x=False, y=True, alpha=0.04)
        self._latency_data = []
        self._latency_curve = self.plot_latency.plot(pen=pg.mkPen(color=GOLD_DIM, width=2), fillLevel=0, fillBrush=QColor(139, 117, 54, 20))
        c4.layout.addWidget(self.plot_latency)
        self.cl.addWidget(c4)

        # 5. LETTER HEATMAP
        c5 = InstrumentCard("LETTER ACTIVATION HEATMAP")
        self._cards.append(c5)
        self._heatmap = LetterHeatmap()
        c5.layout.addWidget(self._heatmap)
        self.cl.addWidget(c5)

        # 6. TOP-K PREDICTIONS
        c6 = InstrumentCard("TOP PREDICTIONS")
        self._cards.append(c6)
        self._topk_bars = []
        for i in range(5):
            row = QHBoxLayout()
            row.setSpacing(8)
            rank = QLabel(f"{i+1}.")
            rank.setFixedWidth(18)
            rank.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; background: transparent;")
            lbl = QLabel("—")
            lbl.setFixedWidth(50)
            lbl.setStyleSheet(f"color: {IVORY}; font-size: 12px; font-weight: 600; background: transparent;")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(8)
            bar.setTextVisible(False)
            bar.setStyleSheet(
                f"QProgressBar {{ border: none; background: {BG_SURFACE}; border-radius: 4px; }}"
                f"QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {GOLD}, stop:1 {GOLD_DIM}); border-radius: 4px; }}"
            )
            pct = QLabel("0%")
            pct.setFixedWidth(32)
            pct.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px; background: transparent;")
            row.addWidget(rank)
            row.addWidget(lbl)
            row.addWidget(bar, 1)
            row.addWidget(pct)
            c6.layout.addLayout(row)
            self._topk_bars.append((lbl, bar, pct))
        self.cl.addWidget(c6)

        # 7. HAND DETECTION STATUS
        c7 = InstrumentCard("HAND DETECTION STATUS")
        self._cards.append(c7)
        hand_row = QHBoxLayout()
        hand_row.setSpacing(16)
        self._left_hand = QLabel("LEFT HAND\n● Not Detected")
        self._left_hand.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; background: transparent; letter-spacing: 1px;")
        self._right_hand = QLabel("RIGHT HAND\n● Not Detected")
        self._right_hand.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; background: transparent; letter-spacing: 1px;")
        self._landmarks_count = QLabel("LANDMARKS\n0 / 21")
        self._landmarks_count.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; background: transparent; letter-spacing: 1px;")
        for w in [self._left_hand, self._right_hand, self._landmarks_count]:
            hand_row.addWidget(w)
        c7.layout.addLayout(hand_row)
        self.cl.addWidget(c7)

        # 8. CONFIDENCE DISTRIBUTION
        c8 = InstrumentCard("CONFIDENCE DISTRIBUTION")
        self._cards.append(c8)
        self.plot_hist = pg.PlotWidget()
        self.plot_hist.setFixedHeight(100)
        self.plot_hist.getAxis('left').setPen(pg.mkPen(color=TEXT_DIM, width=1))
        self.plot_hist.getAxis('bottom').setPen(pg.mkPen(color=TEXT_DIM, width=1))
        self._hist_data = [0] * 10
        self._hist_bars = pg.BarGraphItem(x=list(range(10)), height=self._hist_data, width=0.7, brush=QColor(212, 175, 55, 80), pen=pg.mkPen(color=GOLD, width=1))
        self.plot_hist.addItem(self._hist_bars)
        c8.layout.addWidget(self.plot_hist)
        self.cl.addWidget(c8)

        # 9. DETECTION RATE
        c9 = InstrumentCard("DETECTION RATE (per min)")
        self._cards.append(c9)
        self.plot_rate = pg.PlotWidget()
        self.plot_rate.setFixedHeight(100)
        self.plot_rate.getAxis('left').setPen(pg.mkPen(color=TEXT_DIM, width=1))
        self.plot_rate.getAxis('bottom').setPen(pg.mkPen(color=TEXT_DIM, width=1))
        self.plot_rate.showGrid(x=False, y=True, alpha=0.04)
        self._rate_data = []
        self._rate_curve = self.plot_rate.plot(pen=pg.mkPen(color=GOLD_BRIGHT, width=2))
        c9.layout.addWidget(self.plot_rate)
        self.cl.addWidget(c9)

        # 10. MODEL STATUS
        c10 = InstrumentCard("MODEL STATUS")
        self._cards.append(c10)
        model_grid = QGridLayout()
        model_grid.setSpacing(6)
        model_items = [
            ("Model", "ASL-v2.1 Full"), ("Status", "● Loaded"),
            ("Parameters", "2.4M"), ("Input Shape", "128 × 63"),
            ("Backend", "TensorFlow"), ("Precision", "FP32"),
        ]
        for i, (k, v) in enumerate(model_items):
            kl = QLabel(k.upper())
            kl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9px; letter-spacing: 1.5px; background: transparent;")
            vl = QLabel(v)
            color = SUCCESS if "Loaded" in v else IVORY
            vl.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: 600; background: transparent;")
            model_grid.addWidget(kl, i, 0)
            model_grid.addWidget(vl, i, 1)
        c10.layout.addLayout(model_grid)
        self.cl.addWidget(c10)

        # 11. CAMERA HEALTH
        c11 = InstrumentCard("CAMERA HEALTH")
        self._cards.append(c11)
        cam_row = QHBoxLayout()
        cam_row.setSpacing(8)
        self._cam_res = _stat_label("RESOLUTION", "1280×720")
        self._cam_fps = _stat_label("CAM FPS", "30")
        self._cam_exp = _stat_label("EXPOSURE", "AUTO")
        self._cam_wb = _stat_label("WHITE BAL", "AUTO")
        for w in [self._cam_res, self._cam_fps, self._cam_exp, self._cam_wb]:
            cam_row.addWidget(w)
        c11.layout.addLayout(cam_row)
        self.cl.addWidget(c11)

        # 12. SYSTEM RESOURCES
        c12 = InstrumentCard("SYSTEM RESOURCES")
        self._cards.append(c12)
        res_row = QHBoxLayout()
        res_row.setSpacing(8)
        self._res_cpu = _stat_label("CPU", "12%")
        self._res_mem = _stat_label("MEMORY", "340 MB")
        self._res_gpu = _stat_label("GPU", "N/A")
        self._res_temp = _stat_label("TEMP", "42°C")
        for w in [self._res_cpu, self._res_mem, self._res_gpu, self._res_temp]:
            res_row.addWidget(w)
        c12.layout.addLayout(res_row)
        self.cl.addWidget(c12)

        # 13. PREDICTION LOG
        c13 = InstrumentCard("DETECTION LOG")
        self._cards.append(c13)
        self.log_label = QLabel("> System initialized\n> Neural link ready")
        self.log_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px; font-family: 'Consolas'; background: transparent; border: none;")
        c13.layout.addWidget(self.log_label)
        self.cl.addWidget(c13)

        self.cl.addStretch()
        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    # ── Simulation ──
    def _start_sim(self):
        self._sim_timer = QTimer(self)
        self._sim_timer.timeout.connect(self._sim_tick)
        self._sim_timer.start(500)
        self._det_count = 0
        self._sim_t = 0

    def _sim_tick(self):
        self._sim_t += 1
        if not self.isVisible():
            return
        self._g_conf.set_value(random.uniform(0.6, 0.99))
        self._g_acc.set_value(random.uniform(0.8, 0.98))
        self._g_speed.set_value(random.uniform(0.5, 0.95))
        self._g_model.set_value(random.uniform(0.85, 0.99))
        # Latency
        self._latency_data.append(random.uniform(8, 35))
        if len(self._latency_data) > 60:
            self._latency_data.pop(0)
        self._latency_curve.setData(self._latency_data)
        # Rate
        self._rate_data.append(random.uniform(5, 20))
        if len(self._rate_data) > 40:
            self._rate_data.pop(0)
        self._rate_curve.setData(self._rate_data)
        # Histogram
        self._hist_data = [random.randint(0, 15) for _ in range(10)]
        self._hist_bars.setOpts(height=self._hist_data)

    # ── Public API ──
    def update_fps(self, fps):
        self.fps_data.append(fps)
        if len(self.fps_data) > 50:
            self.fps_data.pop(0)
        self.fps_curve.setData(self.fps_data)
        self.fps_glow.setData(self.fps_data)

    def add_log(self, text):
        current = self.log_label.text()
        lines = current.split('\n')
        lines.append(f"> {text}")
        if len(lines) > 6:
            lines.pop(0)
        self.log_label.setText("\n".join(lines))

    def update_prediction(self, word, conf):
        self._det_count += 1
        v = self._stat_detections.findChild(QLabel, "val")
        if v:
            v.setText(str(self._det_count))
