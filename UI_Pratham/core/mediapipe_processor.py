"""
MediaPipeProcessor — Thread 2 of the processing pipeline.

Wraps the existing landmark extraction logic from ExtractLandmarks/
extract_all_landmarks.py without modifying any backend code.  Runs
MediaPipe Pose + Hand landmarkers in VIDEO mode on each camera frame and
outputs normalised landmark arrays.

Data flow:
    CameraManager.frame_ready  ──▶  process_frame()
                                        │
                                        ├─ annotated_frame_ready(QImage)  ──▶  CameraView
                                        ├─ landmarks_extracted(ndarray)   ──▶  deque buffer
                                        └─ processing_time(float)        ──▶  latency overlay

Backend functions used (NOT modified):
    • normalise_lm_arr_spatially()  from ExtractLandmarks/normalise_data.py
"""

import os
import sys
import time
import numpy as np
import cv2

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage

# ── Suppress noisy MediaPipe / TF logs ──────────────────────────────────
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_LOGGING_THRESHOLD"] = "FATAL"

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles

# ── Import the existing spatial normalisation function ──────────────────
# We add the ExtractLandmarks directory to sys.path so we can import
# normalise_data without moving or copying the file.
_EXTRACT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ExtractLandmarks")
)
if _EXTRACT_DIR not in sys.path:
    sys.path.insert(0, _EXTRACT_DIR)

from normalise_data import normalise_lm_arr_spatially  # noqa: E402


# Connection maps for drawing
_MP_HANDS_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections
_MP_DRAWING = mp.tasks.vision.drawing_utils
_MP_DRAWING_STYLES = mp.tasks.vision.drawing_styles


def _bgr_to_qimage(frame: np.ndarray) -> QImage:
    """Convert an OpenCV BGR frame to a QImage (RGB888).

    We call .copy() to decouple the QImage lifetime from numpy's buffer.
    At 720p this is ~2.7 MB per copy — negligible at 30 FPS.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


class MediaPipeProcessor(QObject):
    """
    Processes camera frames through MediaPipe Pose + Hand landmarkers.

    Lives in a worker thread (moved via moveToThread).  Receives frames
    through the process_frame slot and emits results as signals.

    Signals:
        annotated_frame_ready(QImage)   — frame with landmarks drawn on it
        landmarks_extracted(np.ndarray) — spatially-normalised (64, 4) array
        processing_time(float)          — time in ms for the full pipeline
    """

    annotated_frame_ready = Signal(QImage)
    landmarks_extracted = Signal(np.ndarray)
    processing_time = Signal(float)

    def __init__(
        self,
        pose_model_path: str,
        hand_model_path: str,
        parent=None,
    ):
        super().__init__(parent)

        # Resolve relative paths against the project root (App/../)
        _app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _project_root = os.path.dirname(_app_dir)

        if not os.path.isabs(pose_model_path):
            self._pose_model_path = os.path.normpath(
                os.path.join(_project_root, pose_model_path)
            )
        else:
            self._pose_model_path = pose_model_path

        if not os.path.isabs(hand_model_path):
            self._hand_model_path = os.path.normpath(
                os.path.join(_project_root, hand_model_path)
            )
        else:
            self._hand_model_path = hand_model_path

        # MediaPipe landmarkers — initialised lazily on first frame
        self._pose_landmarker = None
        self._hand_landmarker = None
        self._initialised = False
        self._init_failed = False  # prevent error spam

    # ── lazy init (must happen in the worker thread) ────────────────────

    def _ensure_init(self) -> bool:
        """Create MediaPipe landmarkers.  Called once from the worker thread.
        Returns True if initialised successfully, False otherwise."""
        if self._initialised:
            return True
        if self._init_failed:
            return False

        try:
            print(f"[MediaPipe] Loading pose model: {self._pose_model_path}")
            print(f"[MediaPipe] Loading hand model: {self._hand_model_path}")

            BaseOptions = mp.tasks.BaseOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            pose_opts = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self._pose_model_path),
                running_mode=VisionRunningMode.VIDEO,
            )
            hand_opts = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self._hand_model_path),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=2,
            )

            self._pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)
            self._hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)
            self._initialised = True
            print("[MediaPipe] Models loaded successfully")
            return True

        except Exception as exc:
            print(f"[MediaPipe] Failed to initialise: {exc}")
            self._init_failed = True
            return False

    # ── main processing slot ────────────────────────────────────────────

    @Slot(np.ndarray, int)
    def process_frame(self, bgr_frame: np.ndarray, timestamp_ms: int) -> None:
        """
        Process a single camera frame through the full MediaPipe pipeline.

        This method mirrors the per-frame logic inside the existing doshit()
        function from extract_all_landmarks.py, extracting 75 landmarks
        (33 pose + 21 left hand + 21 right hand) with 4 channels each.

        After extraction, applies normalise_lm_arr_spatially() to reduce
        to a (64, 4) array — one normalised frame ready for the deque.
        """
        if not self._ensure_init():
            # Still emit the raw frame so the camera view isn't blank
            qimg = _bgr_to_qimage(bgr_frame)
            self.annotated_frame_ready.emit(qimg)
            return
        t0 = time.perf_counter()

        # ── Prepare image for MediaPipe (needs RGB) ─────────────────────
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # ── Landmark array for this frame: (75, 4) ──────────────────────
        # Layout matches existing doshit():
        #   0-32  → pose landmarks
        #   33-53 → left hand landmarks
        #   54-74 → right hand landmarks
        frame_landmarks = np.zeros((75, 4), dtype=np.float32)

        # ── Pose detection ──────────────────────────────────────────────
        pose_result = self._pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        # Prepare an annotated copy for display
        display_frame = bgr_frame.copy()

        if pose_result.pose_landmarks:
            pose_lms = pose_result.pose_landmarks[0]
            for i in range(33):
                frame_landmarks[i] = [
                    pose_lms[i].x,
                    pose_lms[i].y,
                    pose_lms[i].z,
                    pose_lms[i].visibility,
                ]

            # Draw pose landmarks on the display frame
            display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            annotated_rgb = self._draw_pose(display_rgb, pose_result)
        else:
            annotated_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

        # ── Hand detection ──────────────────────────────────────────────
        hand_result = self._hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        if hand_result.hand_landmarks:
            for idx, handedness in enumerate(hand_result.handedness):
                label = handedness[0].display_name
                hand_lms = hand_result.hand_landmarks[idx]

                # Left hand → indices 33-53, Right hand → indices 54-74
                # (matches existing doshit() logic exactly)
                landmark_offset = 33 if label == "Left" else 54

                for i in range(21):
                    frame_landmarks[landmark_offset + i] = [
                        hand_lms[i].x,
                        hand_lms[i].y,
                        hand_lms[i].z,
                        1.0,
                    ]

            # Draw hand landmarks on the display frame
            annotated_rgb = self._draw_hands(annotated_rgb, hand_result)

        # ── Spatial normalisation (wrapping existing backend) ───────────
        # The existing function expects shape (N+1, 75, 4) where row 0
        # is metadata [total_frames, fps, ...].  For a single live frame
        # we construct a 2-row array: [metadata_row, data_row].
        raw_arr = np.zeros((2, 75, 4), dtype=np.float32)
        raw_arr[0, 0, 0] = 1  # total_num_frames = 1
        raw_arr[1] = frame_landmarks

        try:
            normalised = normalise_lm_arr_spatially(raw_arr)
            # normalised shape: (1, 64, 4) — squeeze to (64, 4)
            single_frame = normalised[0]
            self.landmarks_extracted.emit(single_frame)
        except Exception:
            # If normalisation fails (e.g. no body detected → zero division),
            # emit a zero frame so the deque stays in sync.
            self.landmarks_extracted.emit(np.zeros((64, 4), dtype=np.float32))

        # ── Emit annotated frame for UI display ─────────────────────────
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        qimg = _bgr_to_qimage(annotated_bgr)
        self.annotated_frame_ready.emit(qimg)

        # ── Timing ──────────────────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.processing_time.emit(elapsed_ms)

    # ── drawing helpers (mirrors existing draw functions) ────────────────

    @staticmethod
    def _draw_pose(rgb_image: np.ndarray, detection_result) -> np.ndarray:
        """Draw pose landmarks — wraps the existing draw_pose_landmarks_on_image."""
        annotated = np.copy(rgb_image)
        pose_landmarks_list = detection_result.pose_landmarks
        pose_style = drawing_styles.get_default_pose_landmarks_style()
        conn_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

        for pose_lms in pose_landmarks_list:
            drawing_utils.draw_landmarks(
                image=annotated,
                landmark_list=pose_lms,
                connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=pose_style,
                connection_drawing_spec=conn_style,
            )
        return annotated

    @staticmethod
    def _draw_hands(rgb_image: np.ndarray, detection_result) -> np.ndarray:
        """Draw hand landmarks — wraps the existing draw_hands_landmarks_on_image."""
        annotated = np.copy(rgb_image)
        for idx in range(len(detection_result.hand_landmarks)):
            hand_lms = detection_result.hand_landmarks[idx]
            _MP_DRAWING.draw_landmarks(
                annotated,
                hand_lms,
                _MP_HANDS_CONNECTIONS.HAND_CONNECTIONS,
                _MP_DRAWING_STYLES.get_default_hand_landmarks_style(),
                _MP_DRAWING_STYLES.get_default_hand_connections_style(),
            )
        return annotated

    # ── cleanup ─────────────────────────────────────────────────────────

    def release(self) -> None:
        """Close MediaPipe landmarkers and free GPU/CPU resources."""
        if self._pose_landmarker:
            self._pose_landmarker.close()
            self._pose_landmarker = None
        if self._hand_landmarker:
            self._hand_landmarker.close()
            self._hand_landmarker = None
        self._initialised = False
