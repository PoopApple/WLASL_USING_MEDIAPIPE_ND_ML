"""
Pipeline — Orchestrator that wires CameraManager, MediaPipeProcessor,
and MLInferenceEngine together with thread-safe queues and Qt signals.

Architecture:
    ┌─────────────┐   frame_ready    ┌──────────────────┐
    │ Camera      │ ────────────────▶│ MediaPipe        │
    │ (QThread)   │                  │ (Worker QThread)  │
    └─────────────┘                  └──────┬───────────┘
                                            │
                                   landmarks_extracted
                                            │
                                            ▼
                                    ┌──────────────┐
                                    │  SafeDeque   │◀── thread-safe buffer
                                    └──────┬───────┘
                                            │  every Nth frame
                                            ▼
                                    ┌──────────────────┐
                                    │ ML Inference     │
                                    │ (Worker QThread)  │
                                    └──────┬───────────┘
                                            │
                                   prediction_ready
                                            │
                                            ▼
                                    ┌──────────────┐
                                    │   UI Thread   │
                                    └──────────────┘

All cross-thread communication uses Qt signals/slots (auto-queued).
The deque is the only shared mutable state, protected by a lock.
"""

import numpy as np
import cv2
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QImage

from .camera_manager import CameraManager
from .mediapipe_processor import MediaPipeProcessor
from .ml_inference import MLInferenceEngine
from .config_manager import ConfigManager
from ..utils.threading_utils import SafeDeque


import os as _os

class Pipeline(QObject):
    """
    Central orchestrator connecting all processing threads.

    Signals forwarded to the UI:
        display_frame(QImage)                — annotated camera frame
        prediction_updated(str, float, list) — word, confidence, top-K
        sentence_updated(list)               — full sentence history
        fps_updated(float)                   — camera FPS
        mp_latency(float)                    — MediaPipe processing time
        ml_latency(float)                    — ML inference time
        status_message(str)                  — human-readable status
    """

    # Project root for resolving relative config paths
    _PROJECT_ROOT = _os.path.dirname(
        _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    )

    # ── signals forwarded to UI ─────────────────────────────────────────
    display_frame = Signal(QImage)
    prediction_updated = Signal(str, float, list)
    sentence_updated = Signal(list)
    fps_updated = Signal(float)
    lumens_updated = Signal(float)
    mp_latency = Signal(float)
    ml_latency = Signal(float)
    status_message = Signal(str)

    def __init__(self, config: ConfigManager, parent=None):
        super().__init__(parent)
        self._config = config

        # ── state ───────────────────────────────────────────────────────
        self._frame_counter = 0
        self._sentence: list[str] = []
        self._last_prediction = ""
        self._last_confidence = 0.0

        # ── deque buffer ────────────────────────────────────────────────
        self._deque = SafeDeque(maxlen=config.get("deque_length", 128))

        # ── Thread 1: Camera ────────────────────────────────────────────
        self._camera = CameraManager(
            camera_index=config.get("camera_index", 0),
            fps_limit=config.get("fps_limit", 30),
        )

        # ── Thread 2: MediaPipe (QObject in a worker thread) ────────────
        self._mp_thread = QThread()
        self._mp_processor = MediaPipeProcessor(
            pose_model_path=config.get("mediapipe_pose_model", ""),
            hand_model_path=config.get("mediapipe_hand_model", ""),
        )
        self._mp_processor.moveToThread(self._mp_thread)

        # ── Thread 3: ML Inference (QObject in a worker thread) ─────────
        self._ml_thread = QThread()
        self._ml_engine = MLInferenceEngine()
        self._ml_engine.moveToThread(self._ml_thread)

        # ── Wire signals ────────────────────────────────────────────────
        self._connect_signals()

        # ── Config hot-reload ───────────────────────────────────────────
        self._config.config_changed.connect(self._on_config_changed)

    # ── signal wiring ───────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        """Connect all inter-thread signals."""
        # Bypass MediaPipe for zero-latency instant preview
        self._camera.frame_ready.connect(self._on_raw_frame)

        # Camera → UI (FPS & Lumens)
        self._camera.fps_updated.connect(self.fps_updated)
        self._camera.lumens_updated.connect(self.lumens_updated)
        self._camera.camera_error.connect(
            lambda msg: self.status_message.emit(f"⚠ Camera: {msg}")
        )

        # (MediaPipe to UI is bypassed)

        # MediaPipe → deque + inference trigger
        self._mp_processor.landmarks_extracted.connect(self._on_landmarks)

        # MediaPipe → UI (latency)
        self._mp_processor.processing_time.connect(self.mp_latency)

        # ML → UI
        self._ml_engine.prediction_ready.connect(self._on_prediction)
        self._ml_engine.inference_time.connect(self.ml_latency)
        self._ml_engine.model_loaded.connect(self._on_model_loaded)

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start all processing threads."""
        self.status_message.emit("Starting pipeline…")

        # Bypass worker threads to save CPU and eliminate lag
        # self._mp_thread.start()
        # self._ml_thread.start()

        # Load ML model (runs in ml_thread)
        model_path = self._config.get("model_path", "")
        label_map_path = self._config.get("label_map_path", "")

        # Resolve relative paths against project root
        if model_path and not _os.path.isabs(model_path):
            model_path = _os.path.normpath(
                _os.path.join(self._PROJECT_ROOT, model_path)
            )
        if label_map_path and not _os.path.isabs(label_map_path):
            label_map_path = _os.path.normpath(
                _os.path.join(self._PROJECT_ROOT, label_map_path)
            )

        # if model_path:
        #     self._ml_engine.load_model(model_path, label_map_path)

        # Start camera capture
        self._camera.start()
        self.status_message.emit("Pipeline running")

    def stop(self) -> None:
        """Gracefully shut down all threads."""
        self.status_message.emit("Stopping pipeline…")

        # Stop camera first (source of data)
        self._camera.stop()

        # Stop worker threads
        self._mp_thread.quit()
        self._mp_thread.wait(1000)

        self._ml_thread.quit()
        self._ml_thread.wait(1000)

        self._deque.clear()
        self.status_message.emit("Pipeline stopped")

    # ── slots ───────────────────────────────────────────────────────────

    @Slot(np.ndarray, int)
    def _on_raw_frame(self, bgr_frame: np.ndarray, timestamp_ms: int) -> None:
        """Convert BGR to QImage instantly and display it without MediaPipe."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        self.display_frame.emit(qimg)

    @Slot(np.ndarray)
    def _on_landmarks(self, landmarks: np.ndarray) -> None:
        """
        Called when MediaPipe produces a normalised landmark frame.
        Appends to the deque and triggers ML inference every N frames.
        """
        self._deque.append(landmarks)
        self._frame_counter += 1

        interval = self._config.get("inference_interval", 5)
        if self._frame_counter % interval == 0:
            snapshot = self._deque.snapshot()
            top_k = self._config.get("display_top_k", 5)
            # Trigger inference in the ML thread
            self._ml_engine.run_inference(snapshot, top_k)

    @Slot(str, float, list)
    def _on_prediction(
        self, word: str, confidence: float, top_k: list
    ) -> None:
        """
        Called when ML inference produces a prediction.
        Updates sentence history and forwards to UI.
        """
        threshold = self._config.get("confidence_threshold", 0.3)

        self._last_prediction = word
        self._last_confidence = confidence

        # Forward to UI regardless of threshold (UI can dim low-confidence)
        self.prediction_updated.emit(word, confidence, top_k)

        # Only append to sentence if above threshold and different from last
        if confidence >= threshold:
            if not self._sentence or self._sentence[-1] != word:
                self._sentence.append(word)
                self.sentence_updated.emit(list(self._sentence))

    @Slot(bool)
    def _on_model_loaded(self, success: bool) -> None:
        if success:
            self.status_message.emit("✓ Model loaded — ready for inference")
        else:
            self.status_message.emit("⚠ Model failed to load — check settings")

    # ── config hot-reload ───────────────────────────────────────────────

    @Slot(dict)
    def _on_config_changed(self, cfg: dict) -> None:
        """Apply configuration changes without restarting."""
        # Update camera FPS
        self._camera.set_fps_limit(cfg.get("fps_limit", 30))

        # Update deque length
        new_maxlen = cfg.get("deque_length", 128)
        if new_maxlen != self._deque.maxlen:
            self._deque.resize(new_maxlen)

    # ── public state access ─────────────────────────────────────────────

    def clear_sentence(self) -> None:
        """Clear the sentence history."""
        self._sentence.clear()
        self.sentence_updated.emit([])

    def get_sentence(self) -> list[str]:
        return list(self._sentence)

    def backspace_sentence(self) -> None:
        """Remove the last word from the sentence."""
        if self._sentence:
            self._sentence.pop()
            self.sentence_updated.emit(list(self._sentence))
