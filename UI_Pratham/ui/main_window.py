"""
MainWindow — Root window with premium luxury navigation.
"""

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QLabel, QPushButton, QStatusBar
)

from .recognition_page import RecognitionPage
from .settings_page import SettingsPage
from .text_to_video_page import TextToVideoPage
from .easter_egg_page import EasterEggPage
from .detailed_info_page import DetailedInfoPage
from .styles.theme import (
    BG_CARD, GOLD, GOLD_DIM, GOLD_BRIGHT,
    TEXT_DIM, TEXT_PRIMARY, BORDER_SUBTLE,
    get_app_stylesheet
)
from ..core.config_manager import ConfigManager
from ..core.pipeline import Pipeline


class NavButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._active = False
        self.setFixedHeight(48)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_style()

    def set_active(self, active):
        self._active = active
        self._update_style()

    def _update_style(self):
        if self._active:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgba(212,175,55,0.06);
                    color: {GOLD};
                    border: none;
                    border-bottom: 2px solid {GOLD};
                    border-radius: 0;
                    font-size: 12px; font-weight: 700;
                    letter-spacing: 2.5px; padding: 0 24px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background-color: transparent;
                    color: {TEXT_DIM};
                    border: none;
                    border-bottom: 2px solid transparent;
                    border-radius: 0;
                    font-size: 12px; font-weight: 600;
                    letter-spacing: 2.5px; padding: 0 24px;
                }}
                QPushButton:hover {{
                    color: {GOLD_DIM};
                    border-bottom: 2px solid {GOLD_DIM};
                    background-color: rgba(212,175,55,0.04);
                }}
            """)


class MainWindow(QMainWindow):
    def __init__(self, config: ConfigManager):
        super().__init__()
        self._config = config
        self.setWindowTitle("SignWave \u2014 Premium Edition")
        self.setMinimumSize(1100, 700)
        self.resize(1400, 850)
        self.setStyleSheet(get_app_stylesheet())

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        nav_bar = self._create_nav_bar()
        main_layout.addWidget(nav_bar)

        self._stack = QStackedWidget()
        self._recognition_page = RecognitionPage()
        self._settings_page = SettingsPage(config)
        self._text_to_video_page = TextToVideoPage()
        self._easter_egg_page = EasterEggPage()
        self._detailed_info_page = DetailedInfoPage()
        
        self._stack.addWidget(self._recognition_page)
        self._stack.addWidget(self._settings_page)
        self._stack.addWidget(self._text_to_video_page)
        self._stack.addWidget(self._easter_egg_page)
        self._stack.addWidget(self._detailed_info_page)
        main_layout.addWidget(self._stack, stretch=1)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("SYSTEM INITIALIZING...")

        self._pipeline = Pipeline(config)
        self._wire_pipeline()
        self._setup_shortcuts()
        self._switch_page(0)
        self._settings_page.settings_applied.connect(self._on_settings_applied)
        self._settings_page.analytics_toggled.connect(
            self._recognition_page.set_analytics_mode
        )
        self._settings_page.scanlines_changed.connect(
            self._recognition_page.camera_view.set_scanline_opacity
        )
        
        self._recognition_page.fullscreen_requested.connect(self._toggle_fullscreen)
        self._easter_egg_page.exit_requested.connect(self._exit_easter_egg)
        self._previous_page_index = 0

    def start_loading(self):
        self._pipeline.start()
        self._recognition_page.camera_view.start_loading()

    def _create_nav_bar(self):
        nav = QWidget()
        nav.setFixedHeight(64)
        nav.setStyleSheet(f"""
            QWidget {{
                background-color: {BG_CARD};
                border-bottom: 1px solid {BORDER_SUBTLE};
            }}
        """)
        layout = QHBoxLayout(nav)
        layout.setContentsMargins(30, 0, 30, 0)
        layout.setSpacing(0)

        logo = QLabel("SIGNWAVE")
        logo.setStyleSheet(f"""
            font-size: 18px;
            font-family: 'Georgia', serif;
            font-weight: 700; letter-spacing: 6px;
            color: {GOLD}; padding-right: 40px;
        """)
        layout.addWidget(logo)

        dot = QLabel("\u00b7")
        dot.setStyleSheet(f"color: {GOLD_DIM}; font-size: 24px; padding: 0 12px;")
        layout.addWidget(dot)

        self._nav_recognition = NavButton("MONITOR")
        self._nav_recognition.clicked.connect(lambda: self._switch_page(0))
        layout.addWidget(self._nav_recognition)

        self._nav_txt2video = NavButton("TEXT TO ASL")
        self._nav_txt2video.clicked.connect(lambda: self._switch_page(2))
        layout.addWidget(self._nav_txt2video)

        self._nav_detailed_info = NavButton("DETAILED INFO")
        self._nav_detailed_info.clicked.connect(lambda: self._switch_page(4))
        layout.addWidget(self._nav_detailed_info)

        self._nav_settings = NavButton("SETTINGS")
        self._nav_settings.clicked.connect(lambda: self._switch_page(1))
        layout.addWidget(self._nav_settings)

        self._nav_toggle = NavButton("ANALYTICS")
        self._nav_toggle.clicked.connect(self._toggle_mode)
        layout.addWidget(self._nav_toggle)

        layout.addStretch()

        # Session timer
        self._session_timer_label = QLabel("00:00:00")
        self._session_timer_label.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 11px; letter-spacing: 2px; "
            f"font-family: 'Consolas', monospace; padding-right: 20px;"
        )
        layout.addWidget(self._session_timer_label)

        # Session timer tick
        from PySide6.QtCore import QTimer, QElapsedTimer
        self._elapsed = QElapsedTimer()
        self._elapsed.start()
        self._session_tick = QTimer(self)
        self._session_tick.timeout.connect(self._update_session_timer)
        self._session_tick.start(1000)

        ver = QLabel("ASL DETECTOR")
        ver.setStyleSheet(
            f"color: {GOLD}; font-size: 11px; letter-spacing: 3px; "
            f"font-weight: 700; font-family: 'Georgia', serif;"
        )
        layout.addWidget(ver)
        return nav

    def _update_session_timer(self):
        ms = self._elapsed.elapsed()
        s = ms // 1000
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        self._session_timer_label.setText(f"{h:02d}:{m:02d}:{sec:02d}")

    def _switch_page(self, index):
        if self._stack.currentWidget() == self._easter_egg_page:
            return  # Prevent nav buttons from working while in easter egg mode
        self._stack.setCurrentIndex(index)
        self._nav_recognition.set_active(index == 0)
        self._nav_settings.set_active(index == 1)
        self._nav_txt2video.set_active(index == 2)
        if hasattr(self, '_nav_detailed_info'):
            self._nav_detailed_info.set_active(index == 4)

    def _exit_easter_egg(self):
        self._stack.setCurrentIndex(self._previous_page_index)

    def _toggle_mode(self):
        cb = self._settings_page._inputs["analytics_mode"]
        cb.setChecked(not cb.isChecked())

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _wire_pipeline(self):
        p = self._pipeline
        rp = self._recognition_page
        p.display_frame.connect(rp.camera_view.update_frame)
        p.fps_updated.connect(rp.hud_overlay.update_fps)
        p.fps_updated.connect(rp.analytics_panel.update_fps)
        p.lumens_updated.connect(rp.hud_overlay.update_lumens)
        p.prediction_updated.connect(
            lambda word, conf, top: rp.analytics_panel.update_prediction(word, conf)
        )
        p.status_message.connect(self._status_bar.showMessage)
        p.status_message.connect(rp.analytics_panel.add_log)

    def _setup_shortcuts(self):
        for key, fn in [("Ctrl+1", lambda: self._switch_page(0)),
                        ("Ctrl+2", lambda: self._switch_page(1)),
                        ("Ctrl+Q", self.close)]:
            act = QAction(self)
            act.setShortcut(QKeySequence(key))
            act.triggered.connect(fn)
            self.addAction(act)

    @Slot()
    def _on_settings_applied(self):
        self._status_bar.showMessage("PREFERENCES UPDATED  \u2713", 3000)

    def closeEvent(self, event):
        self._pipeline.stop()
        self._recognition_page.camera_view.stop_loading()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Slash:
            if self._stack.currentWidget() != self._easter_egg_page:
                self._previous_page_index = self._stack.currentIndex()
                self._stack.setCurrentWidget(self._easter_egg_page)
            else:
                self._exit_easter_egg()
        else:
            super().keyPressEvent(event)
