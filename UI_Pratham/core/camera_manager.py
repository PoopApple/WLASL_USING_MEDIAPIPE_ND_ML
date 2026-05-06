"""
CameraManager — Thread 1 of the processing pipeline.

Captures frames from an OpenCV VideoCapture device at a configurable FPS.
Runs in its own QThread and communicates with the rest of the application
exclusively through Qt signals.

Data flow:
    Camera hardware  ──capture──▶  CameraManager (QThread)
                                        │
                                        ├─ frame_ready(ndarray, int)  ──▶  MediaPipeProcessor
                                        └─ fps_updated(float)         ──▶  FPS overlay

The frame_ready signal carries:
    - BGR numpy array   (h, w, 3)  — raw camera frame
    - timestamp in ms   (int)      — monotonic timestamp for MediaPipe
"""

import time
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, QElapsedTimer


class CameraManager(QThread):
    """
    Captures camera frames in a dedicated thread at a target FPS.

    Signals:
        frame_ready(np.ndarray, int)  — BGR frame + timestamp_ms
        fps_updated(float)            — actual measured FPS (updated ~1/sec)
        camera_error(str)             — emitted on capture failure
    """

    frame_ready = Signal(np.ndarray, int)
    fps_updated = Signal(float)
    lumens_updated = Signal(float)
    camera_error = Signal(str)

    def __init__(self, camera_index: int = 0, fps_limit: int = 30, parent=None):
        super().__init__(parent)
        self._camera_index = camera_index
        self._fps_limit = max(1, fps_limit)
        self._running = False

    # ── configuration hot-reload ────────────────────────────────────────

    def set_fps_limit(self, fps: int) -> None:
        self._fps_limit = max(1, fps)

    def set_camera_index(self, index: int) -> None:
        """Change camera device.  Takes effect on next start()."""
        self._camera_index = index

    # ── lifecycle ───────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Main capture loop — called automatically by QThread.start().

        Uses a simple frame-time budget to enforce the FPS cap while
        keeping latency as low as possible.
        """
        self._running = True

        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            self.camera_error.emit(
                f"Cannot open camera at index {self._camera_index}"
            )
            return

        # Try to hint the backend to use our target resolution / FPS.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, self._fps_limit)

        frame_count = 0
        fps_timer = time.perf_counter()

        # Monotonic timestamp counter (ms) for MediaPipe VIDEO mode.
        # MediaPipe requires strictly increasing timestamps.
        timestamp_ms = 0

        while self._running:
            loop_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret or frame is None:
                # Retry once — some webcams drop frames occasionally.
                ret, frame = cap.read()
                if not ret or frame is None:
                    self.camera_error.emit("Camera read failed")
                    break

            # Mirror the frame horizontally for intuitive interaction
            frame = cv2.flip(frame, 1)

            # Advance the monotonic timestamp by the ideal frame period.
            timestamp_ms += int(1000 / self._fps_limit)

            self.frame_ready.emit(frame, timestamp_ms)
            
            # ── Lumens calculation ──
            # Calculate brightness every few frames to save CPU
            if frame_count % 3 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lumens = float(np.mean(gray))
                self.lumens_updated.emit(lumens)

            frame_count += 1

            # ── FPS measurement (every ~1 second) ──
            now = time.perf_counter()
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                self.fps_updated.emit(frame_count / elapsed)
                frame_count = 0
                fps_timer = now

            # ── FPS limiter ──
            frame_budget = 1.0 / self._fps_limit
            work_time = time.perf_counter() - loop_start
            sleep_time = frame_budget - work_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        # ── cleanup ──
        cap.release()

    def stop(self) -> None:
        """Request the capture loop to stop and wait for the thread to finish."""
        self._running = False
        self.quit()
        self.wait(3000)  # wait up to 3 seconds
