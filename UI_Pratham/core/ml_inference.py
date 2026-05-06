"""
MLInferenceEngine — Thread 3 of the processing pipeline.

Loads a trained TensorFlow / Keras ASL recognition model and runs
inference on the landmark deque buffer at configurable intervals.

Data flow:
    SafeDeque snapshot  ──▶  normalise_lm_arr_temporally()
                                    │
                                    ▼
                             model.predict()
                                    │
                                    ├─ prediction_ready(str, float, list)
                                    └─ inference_time(float)

Backend functions used (NOT modified):
    • normalise_lm_arr_temporally()  from ExtractLandmarks/normalise_data.py
    • TensorFlow model loaded via tf.keras.models.load_model()
"""

import os
import sys
import json
import time
import numpy as np

from PySide6.QtCore import QObject, Signal, Slot

# ── Import the existing temporal normalisation function ─────────────────
_EXTRACT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ExtractLandmarks")
)
if _EXTRACT_DIR not in sys.path:
    sys.path.insert(0, _EXTRACT_DIR)

from normalise_data import normalise_lm_arr_temporally  # noqa: E402


class MLInferenceEngine(QObject):
    """
    Runs ML model inference on landmark buffer snapshots.

    Lives in a dedicated QThread (moved via moveToThread).
    Triggered by the pipeline orchestrator every N frames.

    Signals:
        prediction_ready(str, float, list)
            - predicted word (str)
            - confidence (float 0-1)
            - top-K results as list of (word, confidence) tuples
        inference_time(float)
            - wall-clock time in ms for the inference call
        model_loaded(bool)
            - True if model loaded successfully
    """

    prediction_ready = Signal(str, float, list)
    inference_time = Signal(float)
    model_loaded = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = None
        self._ind_to_word: dict[int, str] = {}
        self._num_classes: int = 0
        self._loaded = False

    # ── model loading ───────────────────────────────────────────────────

    @Slot(str, str)
    def load_model(self, model_path: str, label_map_path: str) -> None:
        """
        Load a trained .keras model and its word-to-index JSON mapping.

        This is called from the pipeline orchestrator after the thread
        starts.  TensorFlow is imported lazily here so the UI thread
        never pays the TF import cost.
        """
        try:
            import tensorflow as tf

            # Suppress TF warnings during load
            tf.get_logger().setLevel("ERROR")

            if not os.path.isfile(model_path):
                print(f"[MLInference] Model file not found: {model_path}")
                self.model_loaded.emit(False)
                return

            self._model = tf.keras.models.load_model(model_path)
            print(f"[MLInference] Model loaded: {model_path}")
            print(f"[MLInference] Input shapes: {[i.shape for i in self._model.inputs]}")

            # Load label mapping
            if os.path.isfile(label_map_path):
                with open(label_map_path, "r", encoding="utf-8") as f:
                    word_to_ind = json.load(f)
                self._ind_to_word = {int(v): k for k, v in word_to_ind.items()}
                self._num_classes = len(self._ind_to_word)
                print(f"[MLInference] Loaded {self._num_classes} class labels")
            else:
                print(f"[MLInference] Label map not found: {label_map_path}")
                # Create generic labels
                self._num_classes = self._model.output_shape[-1]
                self._ind_to_word = {i: f"class_{i}" for i in range(self._num_classes)}

            self._loaded = True
            self.model_loaded.emit(True)

        except Exception as exc:
            print(f"[MLInference] Failed to load model: {exc}")
            self._loaded = False
            self.model_loaded.emit(False)

    # ── inference ───────────────────────────────────────────────────────

    @Slot(list, int)
    def run_inference(self, landmark_frames: list, top_k: int = 5) -> None:
        """
        Run model prediction on a snapshot of the landmark deque.

        Args:
            landmark_frames:  list of np.ndarray, each shape (64, 4).
                              This is a snapshot from SafeDeque.
            top_k:            Number of top predictions to return.
        """
        if not self._loaded or self._model is None:
            return

        if len(landmark_frames) == 0:
            return

        t0 = time.perf_counter()

        try:
            import tensorflow as tf

            # ── Stack frames into (num_frames, 64, 4) ───────────────────
            frame_array = np.stack(landmark_frames, axis=0)  # (N, 64, 4)

            # ── Temporal normalisation (wrapping existing backend) ───────
            # normalise_lm_arr_temporally expects (N, 64, 4) and returns
            # (128, 64, 4) padded/sampled array + (128,) boolean mask.
            normalised, mask = normalise_lm_arr_temporally(frame_array)

            # ── Prepare batch dimension ─────────────────────────────────
            # Model expects: input_data (1, 128, 64, 4), input_mask (1, 128)
            input_data = np.expand_dims(normalised, axis=0).astype(np.float32)
            input_mask = np.expand_dims(mask, axis=0).astype(np.bool_)

            # ── Run prediction ──────────────────────────────────────────
            predictions = self._model.predict(
                {"input_data": input_data, "input_mask": input_mask},
                verbose=0,
            )

            # predictions shape: (1, num_classes) — softmax probabilities
            probs = predictions[0]

            # ── Extract results ─────────────────────────────────────────
            top_indices = np.argsort(probs)[::-1][:top_k]
            top_word = self._ind_to_word.get(top_indices[0], "???")
            top_confidence = float(probs[top_indices[0]])

            top_k_results = [
                (self._ind_to_word.get(int(idx), "???"), float(probs[idx]))
                for idx in top_indices
            ]

            elapsed_ms = (time.perf_counter() - t0) * 1000
            self.inference_time.emit(elapsed_ms)
            self.prediction_ready.emit(top_word, top_confidence, top_k_results)

        except Exception as exc:
            print(f"[MLInference] Inference error: {exc}")

    # ── state ───────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded
