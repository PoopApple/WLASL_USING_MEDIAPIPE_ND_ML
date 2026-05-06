"""
SignWave — Real-Time ASL Recognition Desktop Application.

Entry point for the PyQt6/PySide6 desktop application.  Initialises the
Qt application, loads configuration, and launches the main window with
the full processing pipeline.

Usage:
    python main.py                     # uses default config.json
    python main.py --config my.json    # uses custom config file

Architecture Overview:
    ┌──────────────────────────────────────────────────────────────┐
    │                     UI THREAD (Qt)                          │
    │  MainWindow → RecognitionPage + SettingsPage                │
    │       ▲              ▲              ▲                       │
    │       │ signals      │ signals      │ signals               │
    │       │              │              │                       │
    │  ┌────┴─────┐  ┌────┴─────┐  ┌────┴──────────┐           │
    │  │ Camera   │  │MediaPipe │  │ ML Inference  │           │
    │  │ Thread 1 │  │ Thread 2 │  │ Thread 3     │           │
    │  │ 30 FPS   │──│ ~20ms    │──│ every Nth    │           │
    │  └──────────┘  └──────────┘  └──────────────┘           │
    │                                                           │
    │  ConfigManager ←→ config.json                             │
    └──────────────────────────────────────────────────────────────┘
"""

import sys
import os
import argparse

# ── Ensure project root is on sys.path so imports work ──────────────────
# When running `python App/main.py` from the project root, we need
# the parent directory (ASL_recognition-main/) on the path.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

# ── Unbuffered output so print() appears immediately on Windows ─────────
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── Suppress noisy logs before any imports ──────────────────────────────
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_LOGGING_THRESHOLD"] = "FATAL"

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from core.config_manager import ConfigManager
from ui.main_window import MainWindow


def parse_args():
    parser = argparse.ArgumentParser(
        description="SignWave — Real-Time ASL Recognition"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(_APP_DIR, "config.json"),
        help="Path to configuration JSON file (default: App/config.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Qt Application ──────────────────────────────────────────────────
    app = QApplication(sys.argv)
    app.setApplicationName("SignWave Premium")
    app.setOrganizationName("SignWave")

    # High-DPI scaling (PySide6 enables this by default, but be explicit)
    app.setStyle("Fusion")  # Cross-platform consistent look

    # Default font — refined, warm, non-techy
    font = QFont("Segoe UI", 10)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    app.setFont(font)

    # ── Configuration ───────────────────────────────────────────────────
    config = ConfigManager(args.config)
    print(f"[SignWave] Config loaded from: {args.config}")
    print(f"[SignWave] Model path: {config.get('model_path', '(not set)')}")

    # ── Main Window ─────────────────────────────────────────────────────
    window = MainWindow(config)
    window.show()

    # Start the loading sequence (video -> fade in -> pipeline)
    window.start_loading()

    # ── Event loop ──────────────────────────────────────────────────────
    exit_code = app.exec()

    print("[SignWave] Application closed.")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
