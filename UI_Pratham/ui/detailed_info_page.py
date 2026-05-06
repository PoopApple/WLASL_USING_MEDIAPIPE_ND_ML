"""
DetailedInfoPage — Comprehensive full-screen analytics dashboard.
"""

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QColor, QPainter, QPen, QFont, QBrush
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QScrollArea, QFrame, QSizePolicy, QProgressBar
)

import pyqtgraph as pg
import random
import time
import math

from .styles.theme import (
    BG_CARD, BG_DEEP, BG_SURFACE,
    GOLD, GOLD_DIM, GOLD_BRIGHT, GOLD_SUBTLE,
    TEXT_PRIMARY, TEXT_DIM, IVORY, IVORY_DIM, SUCCESS, BORDER_SUBTLE
)
from .widgets.analytics_panel import LetterHeatmap, InstrumentCard, RadialGauge, _stat_label


class KPICard(QFrame):
    """A small card for a single KPI metric."""
    def __init__(self, title, default_val="0", parent=None):
        super().__init__(parent)
        self.setFixedHeight(100)
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self._title = QLabel(title)
        self._title.setStyleSheet(f"color: {GOLD_DIM}; font-size: 11px; font-weight: 700; letter-spacing: 2px; background: transparent;")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self._val = QLabel(default_val)
        self._val.setStyleSheet(f"color: {IVORY}; font-size: 28px; font-weight: 700; font-family: 'Georgia', serif; background: transparent;")
        self._val.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.layout.addWidget(self._title)
        self.layout.addWidget(self._val)

    def set_value(self, val_str):
        self._val.setText(val_str)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(0, 0, -1, -1)
        p.setBrush(QColor(BG_CARD))
        p.setPen(QPen(QColor(BORDER_SUBTLE), 1))
        p.drawRoundedRect(r, 8, 8)
        p.end()


class DetailedInfoPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._start_simulation()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Scroll Area ──
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{ background: {BG_DEEP}; border: none; }}
            QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
            QScrollBar::handle:vertical {{ background: {GOLD_SUBTLE}; border-radius: 5px; min-height: 20px; }}
            QScrollBar::handle:vertical:hover {{ background: {GOLD_DIM}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
        """)

        content = QWidget()
        content.setObjectName("DetailedContent")
        content.setStyleSheet("QWidget#DetailedContent { background: transparent; }")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(40, 40, 40, 60)
        layout.setSpacing(30)

        # ── Header ──
        header = QLabel("SYSTEM PERFORMANCE & COMPREHENSIVE METRICS")
        header.setStyleSheet(f"color: {GOLD}; font-size: 22px; font-weight: 700; letter-spacing: 6px; font-family: 'Georgia', serif;")
        layout.addWidget(header)

        # ── KPI Grid ──
        kpi_grid = QGridLayout()
        kpi_grid.setSpacing(15)
        
        self.kpi_fps = KPICard("AVG FPS", "0.0")
        self.kpi_frames = KPICard("AVG FRAMES", "0")
        self.kpi_videos = KPICard("AVG VIDEOS / VOCAB", "0.0")
        self.kpi_f1 = KPICard("AVG F1 SCORE", "0.00")
        self.kpi_precision = KPICard("PRECISION", "0.00%")
        self.kpi_top5 = KPICard("TOP 5 SCORE", "0.00%")
        self.kpi_top7 = KPICard("TOP 7 SCORE", "0.00%")
        self.kpi_top10 = KPICard("TOP 10 SCORE", "0.00%")

        kpi_grid.addWidget(self.kpi_fps, 0, 0)
        kpi_grid.addWidget(self.kpi_frames, 0, 1)
        kpi_grid.addWidget(self.kpi_videos, 0, 2)
        kpi_grid.addWidget(self.kpi_f1, 0, 3)
        kpi_grid.addWidget(self.kpi_precision, 1, 0)
        kpi_grid.addWidget(self.kpi_top5, 1, 1)
        kpi_grid.addWidget(self.kpi_top7, 1, 2)
        kpi_grid.addWidget(self.kpi_top10, 1, 3)

        layout.addLayout(kpi_grid)

        # ── Charts Configuration ──
        pg.setConfigOption('antialias', True)
        pg.setConfigOption('background', 'transparent')
        pg.setConfigOption('foreground', TEXT_DIM)

        charts_layout = QGridLayout()
        charts_layout.setSpacing(20)

        # 1. FPS Graph (Large)
        card_fps = InstrumentCard("FRAME RATE STABILITY (LONG TERM)")
        card_fps.setMinimumHeight(250)
        self.plot_fps = pg.PlotWidget()
        self.plot_fps.showGrid(x=False, y=True, alpha=0.1)
        self._data_fps = [30] * 100
        self._curve_fps = self.plot_fps.plot(pen=pg.mkPen(color=GOLD, width=3), fillLevel=0, fillBrush=QColor(212, 175, 55, 30))
        card_fps.layout.addWidget(self.plot_fps)
        charts_layout.addWidget(card_fps, 0, 0, 1, 2)

        # 2. Model Accuracy Trend (NEW)
        card_acc = InstrumentCard("MODEL ACCURACY TREND")
        card_acc.setMinimumHeight(250)
        self.plot_acc = pg.PlotWidget()
        self.plot_acc.showGrid(x=False, y=True, alpha=0.1)
        self._data_acc = [85.0] * 100
        self._curve_acc = self.plot_acc.plot(pen=pg.mkPen(color=SUCCESS, width=2), fillLevel=80, fillBrush=QColor(0, 255, 128, 20))
        card_acc.layout.addWidget(self.plot_acc)
        charts_layout.addWidget(card_acc, 0, 2)

        # 3. Latency Graph
        card_lat = InstrumentCard("INFERENCE LATENCY (ms)")
        card_lat.setMinimumHeight(200)
        self.plot_lat = pg.PlotWidget()
        self.plot_lat.showGrid(x=False, y=True, alpha=0.1)
        self._data_lat = [40] * 100
        self._curve_lat = self.plot_lat.plot(pen=pg.mkPen(color=GOLD_BRIGHT, width=2), fillLevel=0, fillBrush=QColor(212, 175, 55, 20))
        card_lat.layout.addWidget(self.plot_lat)
        charts_layout.addWidget(card_lat, 1, 0)

        # 4. Detection Rate
        card_rate = InstrumentCard("DETECTION RATE (per min)")
        card_rate.setMinimumHeight(200)
        self.plot_rate = pg.PlotWidget()
        self.plot_rate.showGrid(x=False, y=True, alpha=0.1)
        self._data_rate = [0] * 100
        self._curve_rate = self.plot_rate.plot(pen=pg.mkPen(color=IVORY, width=2))
        card_rate.layout.addWidget(self.plot_rate)
        charts_layout.addWidget(card_rate, 1, 1)

        # 5. Pipeline Latency Breakdown (NEW)
        card_pipe = InstrumentCard("PIPELINE STAGE LATENCY (ms)")
        card_pipe.setMinimumHeight(200)
        self.plot_pipe = pg.PlotWidget()
        self.plot_pipe.showGrid(x=False, y=True, alpha=0.1)
        self._data_pipe1 = [15] * 50 # Capture
        self._data_pipe2 = [25] * 50 # MP
        self._data_pipe3 = [10] * 50 # Model
        self._curve_pipe1 = self.plot_pipe.plot(pen=pg.mkPen(color=QColor(212, 175, 55, 50), width=2))
        self._curve_pipe2 = self.plot_pipe.plot(pen=pg.mkPen(color=QColor(212, 175, 55, 150), width=2))
        self._curve_pipe3 = self.plot_pipe.plot(pen=pg.mkPen(color=GOLD_BRIGHT, width=2))
        card_pipe.layout.addWidget(self.plot_pipe)
        charts_layout.addWidget(card_pipe, 1, 2)

        # 6. Confidence Histogram
        card_hist = InstrumentCard("CONFIDENCE DISTRIBUTION")
        card_hist.setMinimumHeight(250)
        self.plot_hist = pg.PlotWidget()
        self._data_hist = [random.randint(5, 50) for _ in range(10)]
        self._bar_hist = pg.BarGraphItem(x=list(range(10)), height=self._data_hist, width=0.6, brush=QColor(212, 175, 55, 120), pen=pg.mkPen(color=GOLD, width=1))
        self.plot_hist.addItem(self._bar_hist)
        card_hist.layout.addWidget(self.plot_hist)
        charts_layout.addWidget(card_hist, 2, 0)

        # 7. Heatmap
        card_heat = InstrumentCard("LETTER ACTIVATION HEATMAP")
        card_heat.setMinimumHeight(250)
        self._heatmap = LetterHeatmap()
        card_heat.layout.addWidget(self._heatmap)
        charts_layout.addWidget(card_heat, 2, 1)
        
        # 8. Top Predictions
        c6 = InstrumentCard("TOP PREDICTIONS")
        c6.setMinimumHeight(250)
        self._topk_bars = []
        for i in range(7):
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
        charts_layout.addWidget(c6, 2, 2)

        # 9. Gauges
        c2 = InstrumentCard("CONFIDENCE INSTRUMENTS")
        c2.setFixedHeight(160)
        gauge_row = QHBoxLayout()
        gauge_row.setSpacing(4)
        self._g_conf = RadialGauge("CONF", 100)
        self._g_acc = RadialGauge("ACC", 100)
        self._g_speed = RadialGauge("SPEED", 100)
        self._g_model = RadialGauge("MODEL", 100)
        for g in [self._g_conf, self._g_acc, self._g_speed, self._g_model]:
            gauge_row.addWidget(g, alignment=Qt.AlignmentFlag.AlignCenter)
        c2.layout.addLayout(gauge_row)
        charts_layout.addWidget(c2, 3, 0, 1, 2)

        # 10. Status/Info Group
        info_group = QVBoxLayout()
        info_group.setSpacing(15)
        
        c7 = InstrumentCard("HAND DETECTION STATUS")
        hand_row = QHBoxLayout()
        self._left_hand = QLabel("LEFT HAND\n● Detected")
        self._left_hand.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; background: transparent; letter-spacing: 1px;")
        self._right_hand = QLabel("RIGHT HAND\n● Not Detected")
        self._right_hand.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; background: transparent; letter-spacing: 1px;")
        self._landmarks_count = QLabel("LANDMARKS\n21 / 42")
        self._landmarks_count.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px; background: transparent; letter-spacing: 1px;")
        for w in [self._left_hand, self._right_hand, self._landmarks_count]:
            hand_row.addWidget(w)
        c7.layout.addLayout(hand_row)
        info_group.addWidget(c7)

        c11 = InstrumentCard("CAMERA & SYSTEM HEALTH")
        cam_row = QHBoxLayout()
        self._cam_res = _stat_label("RESOLUTION", "1280×720")
        self._res_cpu = _stat_label("CPU", "12%")
        self._res_mem = _stat_label("MEMORY", "340 MB")
        self._res_temp = _stat_label("TEMP", "42°C")
        for w in [self._cam_res, self._res_cpu, self._res_mem, self._res_temp]:
            cam_row.addWidget(w)
        c11.layout.addLayout(cam_row)
        info_group.addWidget(c11)

        c10 = InstrumentCard("MODEL STATUS")
        model_grid = QGridLayout()
        model_items = [
            ("Model", "ASL-v2.1 Full"), ("Status", "● Loaded"),
            ("Parameters", "2.4M"), ("Input Shape", "128 × 63"),
        ]
        for i, (k, v) in enumerate(model_items):
            kl = QLabel(k.upper())
            kl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 9px; letter-spacing: 1.5px; background: transparent;")
            vl = QLabel(v)
            color = SUCCESS if "Loaded" in v else IVORY
            vl.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: 600; background: transparent;")
            model_grid.addWidget(kl, i//2, (i%2)*2)
            model_grid.addWidget(vl, i//2, (i%2)*2+1)
        c10.layout.addLayout(model_grid)
        info_group.addWidget(c10)

        charts_layout.addLayout(info_group, 3, 2)

        layout.addLayout(charts_layout)
        scroll.setWidget(content)
        main_layout.addWidget(scroll)

        # Fix opacity for all imported InstrumentCards
        for card in self.findChildren(InstrumentCard):
            card.opacity_effect.setOpacity(1.0)

    # ── Entry Points for Backend API ──

    @Slot(float)
    def update_avg_fps(self, val):
        self.kpi_fps.set_value(f"{val:.1f}")

    @Slot(int)
    def update_avg_frames(self, val):
        self.kpi_frames.set_value(str(val))

    @Slot(float)
    def update_avg_videos_per_vocab(self, val):
        self.kpi_videos.set_value(f"{val:.1f}")

    @Slot(float)
    def update_avg_f1_score(self, val):
        self.kpi_f1.set_value(f"{val:.2f}")

    @Slot(float)
    def update_precision(self, val):
        self.kpi_precision.set_value(f"{val:.2f}%")

    @Slot(float)
    def update_top5_score(self, val):
        self.kpi_top5.set_value(f"{val:.2f}%")

    @Slot(float)
    def update_top7_score(self, val):
        self.kpi_top7.set_value(f"{val:.2f}%")

    @Slot(float)
    def update_top10_score(self, val):
        self.kpi_top10.set_value(f"{val:.2f}%")

    # ── Simulation (Skeletal Data) ──

    def _start_simulation(self):
        self._sim_timer = QTimer(self)
        self._sim_timer.timeout.connect(self._simulate_tick)
        self._sim_timer.start(50)  # 20hz update

        self._kpi_timer = QTimer(self)
        self._kpi_timer.timeout.connect(self._simulate_kpis)
        self._kpi_timer.start(2000)

        self._t = 0

    def _simulate_tick(self):
        self._t += 0.05
        
        # FPS
        fps = 30 + math.sin(self._t) * 2 + random.uniform(-1, 1)
        self._data_fps.pop(0)
        self._data_fps.append(fps)
        self._curve_fps.setData(self._data_fps)

        # Accuracy
        acc = min(99.0, max(80.0, 92 + math.log(self._t + 1) * 2 + random.uniform(-1, 1)))
        self._data_acc.pop(0)
        self._data_acc.append(acc)
        self._curve_acc.setData(self._data_acc)

        # Latency
        lat = 42 + math.cos(self._t * 0.5) * 5 + random.uniform(-2, 2)
        self._data_lat.pop(0)
        self._data_lat.append(lat)
        self._curve_lat.setData(self._data_lat)

        # Rate
        rate = max(0, 15 + math.sin(self._t * 0.2) * 10 + random.uniform(-3, 3))
        self._data_rate.pop(0)
        self._data_rate.append(rate)
        self._curve_rate.setData(self._data_rate)

        # Pipeline breakdown
        self._data_pipe1.pop(0); self._data_pipe1.append(15 + random.uniform(-1, 1))
        self._data_pipe2.pop(0); self._data_pipe2.append(25 + math.sin(self._t)*2 + random.uniform(-1, 1))
        self._data_pipe3.pop(0); self._data_pipe3.append(10 + random.uniform(-0.5, 0.5))
        self._curve_pipe1.setData(self._data_pipe1)
        self._curve_pipe2.setData(self._data_pipe2)
        self._curve_pipe3.setData(self._data_pipe3)

        # Histogram (slower update)
        if random.random() < 0.1:
            self._data_hist = [max(0, h + random.randint(-5, 5)) for h in self._data_hist]
            self._bar_hist.setOpts(height=self._data_hist)

        # Gauges
        if random.random() < 0.2:
            self._g_conf.set_value(random.uniform(0.7, 0.99))
            self._g_acc.set_value(random.uniform(0.8, 0.95))
            self._g_speed.set_value(random.uniform(0.6, 0.9))
            self._g_model.set_value(random.uniform(0.9, 1.0))

        # Top Predictions
        if random.random() < 0.1:
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            top = random.sample(letters, 7)
            vals = sorted([random.uniform(10, 95) for _ in range(7)], reverse=True)
            for i, (lbl, bar, pct) in enumerate(self._topk_bars):
                lbl.setText(top[i])
                bar.setValue(int(vals[i]))
                pct.setText(f"{int(vals[i])}%")

    def _simulate_kpis(self):
        self.update_avg_fps(random.uniform(28.5, 31.2))
        self.update_avg_frames(random.randint(120, 150))
        self.update_avg_videos_per_vocab(random.uniform(40.0, 45.0))
        self.update_avg_f1_score(random.uniform(0.85, 0.94))
        self.update_precision(random.uniform(88.5, 96.2))
        self.update_top5_score(random.uniform(94.0, 98.5))
        self.update_top7_score(random.uniform(96.0, 99.1))
        self.update_top10_score(random.uniform(98.0, 99.8))
