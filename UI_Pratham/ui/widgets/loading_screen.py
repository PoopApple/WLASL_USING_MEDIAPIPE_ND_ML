"""
LoadingScreen — Plays a startup video before revealing the main UI.

Uses OpenCV and QTimer to render the video frame-by-frame, ensuring
compatibility without needing QtMultimedia codecs.
"""

import os
import cv2
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel


class LoadingScreen(QWidget):
    """
    Plays an MP4 video and emits `finished` when playback completes.
    """

    finished = Signal()

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self._video_path = video_path

        # Setup UI
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self._video_label = QLabel(self)
        self._video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._video_label.setStyleSheet("background-color: black;")
        self._layout.addWidget(self._video_label)

        # OpenCV Capture
        self._cap = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_frame)

    def start(self):
        """Begin video playback."""
        if not os.path.exists(self._video_path):
            print(f"[LoadingScreen] Video not found: {self._video_path}")
            self.finished.emit()
            return

        self._cap = cv2.VideoCapture(self._video_path)
        if not self._cap.isOpened():
            print("[LoadingScreen] Failed to open video.")
            self.finished.emit()
            return

        # Attempt to get video FPS to set the timer interval correctly
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        interval = int(1000 / fps) if fps > 0 else 33

        self._timer.start(interval)

    def _update_frame(self):
        if self._cap is None:
            return

        ret, frame = self._cap.read()
        if not ret:
            # Video ended
            self._timer.stop()
            self._cap.release()
            self._cap = None
            self.finished.emit()
            return

        # Convert BGR to QImage
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

        # Scale to fit window while keeping aspect ratio
        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation
        )
        self._video_label.setPixmap(scaled_pixmap)

    def stop(self):
        """Force stop playback."""
        if self._timer.isActive():
            self._timer.stop()
        if self._cap:
            self._cap.release()
            self._cap = None

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)
