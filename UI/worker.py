"""
worker.py
=========
VideoWorker pipeline:

  1. Every frame → MediaPipe landmark extraction → spatial normalisation
  2. Raw (75, 4) frames pushed into a rolling deque of `buffer_size` frames.
  3. Every `motion_check_interval` frames, run motion-energy segmentation
     on the entire buffer using the proven ExtractLandmarks/motion_detection.py
     algorithm.
  4. If a segment is found, try inferring from multiple temporal windows
     (e.g. 256 → 128 → 64 frames) to handle slow/fast signers:
       - Window ≥ segment length → subsample to 64 (speedup handles slow signers)
       - Window < segment length → crop the most-motion-dense part
     The *first* window that contains the full segment is used.
  5. Sends the best (64, 64, 4) + mask to InferenceWorker.
"""

import sys
import os
import cv2
import time
from collections import deque

import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

from preprocess import process_landmarks, normalise_lm_arr_temporally
from get_prediction import load_prediction_model, run_inference

# ── Motion detection import ───────────────────────────────────────────────────
# The motion_detection module lives in ExtractLandmarks; add it to path.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EL_DIR    = os.path.join(_REPO_ROOT, "ExtractLandmarks")
if _EL_DIR not in sys.path:
    sys.path.insert(0, _EL_DIR)

from motion_detection import (
    compute_motion_per_frame,
    find_motion_segments,
    find_longest_segment,
    DEFAULTS as MD_DEFAULTS,
)


# ─────────────────────────────────────────────────────────────────────────────
class InferenceWorker(QThread):
    result_signal           = pyqtSignal(str)
    predicting_status_signal = pyqtSignal(bool)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.running = False
        self.frames_to_process = None
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.model = None
        self.loaded_model_path = None
        self.new_model_path = config.get("prediction_model", "")

    def update_frames(self, frames, mode="real_time"):
        self.mutex.lock()
        self.frames_to_process = frames
        self.current_mode = mode
        self.condition.wakeOne()
        self.mutex.unlock()

    def update_model_path(self, new_path):
        self.mutex.lock()
        self.new_model_path = new_path
        self.condition.wakeOne()
        self.mutex.unlock()

    def run(self):
        self.running = True
        while self.running:
            self.mutex.lock()
            self.condition.wait(self.mutex, 100)
            frames = self.frames_to_process
            self.frames_to_process = None
            pending_model_path = self.new_model_path
            self.mutex.unlock()

            if pending_model_path and pending_model_path != self.loaded_model_path:
                self.model = load_prediction_model(pending_model_path)
                self.loaded_model_path = pending_model_path

            if frames is not None and self.running and self.model is not None:
                self.predicting_status_signal.emit(True)
                print_preds = self.config.get("print_predictions", False)
                include_confidence = getattr(self, "current_mode", "real_time") == "video_testing"
                result = run_inference(self.model, frames, print_preds=print_preds,
                                       include_confidence=include_confidence)
                self.predicting_status_signal.emit(False)
                if result:
                    self.result_signal.emit(result)

    def stop(self):
        self.running = False
        self.mutex.lock()
        self.condition.wakeOne()
        self.mutex.unlock()
        self.wait()


# ─────────────────────────────────────────────────────────────────────────────
# Landmark drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_pose_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    pose_landmark_style    = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style  = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
    for pose_landmarks in pose_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style,
        )
    return annotated_image


mp_hands          = mp.tasks.vision.HandLandmarksConnections
mp_drawing        = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN               = 10
FONT_SIZE            = 1
FONT_THICKNESS       = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)


def draw_hands_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list     = detection_result.handedness
    annotated_image     = np.copy(rgb_image)
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness     = handedness_list[idx]
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
        height, width, _ = annotated_image.shape
        x_coordinates = [lm.x for lm in hand_landmarks]
        y_coordinates = [lm.y for lm in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    return annotated_image


# ─────────────────────────────────────────────────────────────────────────────
# Main VideoWorker
# ─────────────────────────────────────────────────────────────────────────────

class VideoWorker(QThread):
    frame_signal = pyqtSignal(object)
    text_signal  = pyqtSignal(str)

    def __init__(self, source, config):
        super().__init__()
        self.source  = source
        self.config  = config
        self.running = False
        self.processing = False
        self.new_source = None
        self.draw_landmarks_flag = config.get("draw_landmarks", True)

        # ── Target sequence length (model input)
        self.target_frames = config.get("target_frames", 64)

        # ── Rolling raw-landmark buffer (holds (75,4) arrays, unprocessed)
        self.buffer_size = config.get("buffer_size", 256)
        self.raw_buffer  = deque(maxlen=self.buffer_size)   # raw (75,4) frames
        self.proc_buffer = deque(maxlen=self.buffer_size)   # processed (64,4) frames (spatially normalised)

        # ── How often to run the motion detector (every N new frames)
        self.motion_check_interval = config.get("motion_check_interval", 5)
        self._frames_since_check   = 0

        # ── Temporal windows to try (descending order: slow→fast signers)
        self.temporal_windows = config.get("temporal_windows", [256, 128, 64])

        # ── Motion detection parameters
        self.md_start_thresh  = config.get("motion_start_threshold",  MD_DEFAULTS["start_thresh"])
        self.md_end_thresh    = config.get("motion_end_threshold",    MD_DEFAULTS["end_thresh"])
        self.md_cooldown      = config.get("motion_cooldown_frames",  MD_DEFAULTS["cooldown"])
        self.md_min_frames    = config.get("motion_min_sign_frames",  MD_DEFAULTS["min_frames"])
        self.md_method        = config.get("motion_method",           MD_DEFAULTS["method"])
        self.md_top_k         = config.get("motion_top_k",            MD_DEFAULTS["top_k"])
        self.md_tip_weight    = config.get("motion_tip_weight",       MD_DEFAULTS["tip_weight"])
        self.md_head_pad      = config.get("motion_head_pad",         MD_DEFAULTS["head_pad"])
        self.md_vis_thresh    = config.get("motion_vis_thresh",       MD_DEFAULTS["vis_thresh"])

        # ── Latest landmark results (async callbacks)
        self.latest_pose_result = None
        self.latest_hand_result = None
        self.frame_count        = 0
        self.is_predicting      = False
        self.mode               = "real_time"
        self.has_predicted      = False

        # ── Inference worker
        self.inference_worker = InferenceWorker(config)
        self.inference_worker.result_signal.connect(self.text_signal.emit)
        self.inference_worker.predicting_status_signal.connect(self.set_predicting_status)
        self.inference_worker.start()

        # ── MediaPipe model paths
        self.pose_model_type     = config.get("pose_model", "lite")
        self.pose_model_path_video = f"./mp_models/pose_landmarker_{self.pose_model_type}.task"
        self.pose_model_path_live  = "./mp_models/pose_landmarker_lite.task"
        self.hand_model_path       = "./mp_models/hand_landmarker.task"

        self.setup_landmarkers()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def set_predicting_status(self, status):
        self.is_predicting = status

    def change_source(self, source):
        self.new_source = source
        self.stop_processing()

    def set_mode(self, mode):
        self.mode = mode

    def pose_callback(self, result, output_image, timestamp_ms):
        self.latest_pose_result = result

    def hand_callback(self, result, output_image, timestamp_ms):
        self.latest_hand_result = result

    # ── MediaPipe setup ────────────────────────────────────────────────────────
    def setup_landmarkers(self):
        if hasattr(self, "pose_landmarker"):
            self.pose_landmarker.close()
        if hasattr(self, "hand_landmarker"):
            self.hand_landmarker.close()

        BaseOptions    = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        self.is_live   = isinstance(self.source, int)
        mode           = VisionRunningMode.LIVE_STREAM if self.is_live else VisionRunningMode.VIDEO
        pose_model_path = self.pose_model_path_live if self.is_live else self.pose_model_path_video

        PoseLandmarker        = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        HandLandmarker        = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        if self.is_live:
            pose_options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=pose_model_path),
                running_mode=mode, result_callback=self.pose_callback)
            hand_options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.hand_model_path),
                running_mode=mode, num_hands=2, result_callback=self.hand_callback)
        else:
            pose_options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=pose_model_path),
                running_mode=mode)
            hand_options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.hand_model_path),
                running_mode=mode, num_hands=2)

        self.pose_landmarker = PoseLandmarker.create_from_options(pose_options)
        self.hand_landmarker = HandLandmarker.create_from_options(hand_options)

    # ── Landmark extraction ────────────────────────────────────────────────────
    def extract_landmarks(self, pose_result, hand_result):
        """Returns raw (75, 4) array."""
        curr_frame_data = np.zeros((75, 4), dtype=np.float32)

        if pose_result and getattr(pose_result, "pose_landmarks", None) and len(pose_result.pose_landmarks) > 0:
            arr = pose_result.pose_landmarks[0]
            for i in range(33):
                curr_frame_data[i] = [arr[i].x, arr[i].y, arr[i].z, arr[i].visibility]

        if hand_result and getattr(hand_result, "hand_landmarks", None) and len(hand_result.hand_landmarks) > 0:
            for idx, handedd in enumerate(hand_result.handedness):
                lbl    = handedd[0].display_name
                handLMs = hand_result.hand_landmarks[idx]
                landmark_index = 33 if lbl == "Left" else 54
                for i in range(21):
                    curr_frame_data[landmark_index + i] = [
                        handLMs[i].x, handLMs[i].y, handLMs[i].z, 1.0
                    ]

        return curr_frame_data

    def start_processing(self):
        self.processing = True

    def stop_processing(self):
        self.processing = False
        self.raw_buffer.clear()
        self.proc_buffer.clear()
        self.frame_count = 0
        self._frames_since_check = 0

    # ── Motion-aware inference trigger ────────────────────────────────────────
    def _try_motion_inference(self):
        """
        Run motion segmentation on the full raw buffer, then extract and
        normalise the best segment for inference.

        Window selection strategy
        ─────────────────────────
        We want the SMALLEST window that fully contains the sign + head_pad.
        Larger windows cause more idle frames to be subsampled into the 64-frame
        target, effectively burying the sign in background noise.

          Segment 50 frames → try 64 first  → fits → subsample 64→64 (×1)
          Segment 90 frames → try 128 first → fits → subsample 128→64 (×2 speedup)
          Segment 200 frames→ try 256 first → fits → subsample 256→64 (×4 speedup)

        Duplicate-fire guard
        ────────────────────
        Because this is called every `motion_check_interval` frames, the same
        completed sign could fire multiple times.  We track `_last_fired_seg_end`
        and skip if the new segment's end is within `motion_check_interval` frames
        of the last fired one.
        """
        raw_arr  = np.array(self.raw_buffer)   # (T, 75, 4)
        proc_arr = np.array(self.proc_buffer)  # (T, 64, 4)
        T = len(raw_arr)
        if T < self.md_min_frames:
            return

        energies = compute_motion_per_frame(
            raw_arr,
            method=self.md_method,
            top_k=self.md_top_k,
            tip_weight=self.md_tip_weight,
            vis_thresh=self.md_vis_thresh,
        )
        segments = find_motion_segments(
            energies,
            start_thresh=self.md_start_thresh,
            end_thresh=self.md_end_thresh,
            cooldown_frames=self.md_cooldown,
            min_sign_frames=self.md_min_frames,
        )
        seg = find_longest_segment(segments)
        if seg is None:
            return

        seg_start, seg_end = seg

        # ── Duplicate-fire guard ──────────────────────────────────────────────
        last_end = getattr(self, "_last_fired_seg_end", -999)
        if abs(seg_end - last_end) <= self.motion_check_interval * 2:
            return  # same sign, skip

        # ── Head-pad: step back before first visible hand ─────────────────────
        padded_start = max(0, seg_start - self.md_head_pad)
        seg_needed   = seg_end - padded_start  # total frames required

        # ── Window selection: SMALLEST window that contains the sign ──────────
        chosen_window = None
        for window in sorted(self.temporal_windows):  # ascending: 64 → 128 → 256
            if window >= seg_needed:
                chosen_window = window
                break

        if chosen_window is None:
            # Sign longer than all windows — use largest, crop to densest part
            chosen_window = max(self.temporal_windows)

        # ── Extract frames: anchor to padded_start, extend chosen_window ──────
        extract_start = padded_start
        extract_end   = min(T, extract_start + chosen_window)

        # If we hit the buffer end, shift start backwards to get full window
        if (extract_end - extract_start) < chosen_window:
            extract_start = max(0, extract_end - chosen_window)

        window_frames = proc_arr[extract_start:extract_end]

        if len(window_frames) < self.md_min_frames:
            return

        # ── Temporal normalisation → 64 frames ───────────────────────────────
        arr_padded, mask = normalise_lm_arr_temporally(window_frames, self.target_frames)

        print(f"[Motion] seg={seg_start}:{seg_end}  pad_start={padded_start}"
              f"  window={chosen_window}  extract={extract_start}:{extract_end}"
              f"  → {self.target_frames} frames")

        self._last_fired_seg_end = seg_end
        self.inference_worker.update_frames((arr_padded, mask))

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(self.source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        self.running = True

        while self.running:
            # ── Source change ─────────────────────────────────────────────────
            if self.new_source is not None:
                if self.new_source == self.source:
                    self.new_source = None
                    continue
                cap.release()
                self.source    = self.new_source
                self.new_source = None
                self.has_predicted = False
                self.raw_buffer.clear()
                self.proc_buffer.clear()
                self.frame_count = 0
                self._frames_since_check = 0
                self.setup_landmarkers()
                cap = cv2.VideoCapture(self.source)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30

                # ── Video-testing mode: process whole file at once ────────────
                if self.mode == "video_testing" and isinstance(self.source, str):
                    self.text_signal.emit("<tr><td>Processing... Please wait.</td></tr>")
                    all_frames  = []
                    all_raw     = []
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                    emit_every  = max(1, total_frames // 60)
                    frame_idx   = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        curr_frame_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                        if curr_frame_ms <= getattr(self, "last_timestamp_ms", -1):
                            curr_frame_ms = getattr(self, "last_timestamp_ms", -1) + 1
                        self.last_timestamp_ms = curr_frame_ms
                        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        pose_result = self.pose_landmarker.detect_for_video(mp_image, curr_frame_ms)
                        hand_result = self.hand_landmarker.detect_for_video(mp_image, curr_frame_ms)
                        raw = self.extract_landmarks(pose_result, hand_result)
                        processed = process_landmarks(raw)
                        if processed is not None:
                            all_raw.append(raw)
                            all_frames.append(processed)
                        if frame_idx % emit_every == 0:
                            if self.draw_landmarks_flag:
                                annotated = np.copy(rgb)
                                if pose_result and getattr(pose_result, "pose_landmarks", None) and len(pose_result.pose_landmarks) > 0:
                                    annotated = draw_pose_landmarks_on_image(annotated, pose_result)
                                if hand_result and getattr(hand_result, "hand_landmarks", None) and len(hand_result.hand_landmarks) > 0:
                                    annotated = draw_hands_landmarks_on_image(annotated, hand_result)
                                self.frame_signal.emit(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                            else:
                                self.frame_signal.emit(frame)
                        frame_idx += 1

                    if all_frames:
                        arr_padded, mask = normalise_lm_arr_temporally(
                            np.array(all_frames), self.target_frames)
                        self.inference_worker.update_frames((arr_padded, mask), self.mode)
                        self.has_predicted = True
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # ── Normal frame read ─────────────────────────────────────────────
            ret, frame = cap.read()
            if not ret:
                if isinstance(self.source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    self.text_signal.emit("Error: Camera unavailable. Please select another.")
                    time.sleep(0.5)
                continue

            if self.mode == "video_testing" and isinstance(self.source, str):
                self.frame_signal.emit(frame)
                time.sleep(1 / fps)
                continue

            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            curr_frame_ms = int(time.time() * 1000) if self.is_live else int(cap.get(cv2.CAP_PROP_POS_MSEC))

            if not self.processing:
                self.frame_signal.emit(frame)
                if not self.is_live:
                    time.sleep(1 / fps)
                continue

            if curr_frame_ms <= getattr(self, "last_timestamp_ms", -1):
                curr_frame_ms = getattr(self, "last_timestamp_ms", -1) + 1
            self.last_timestamp_ms = curr_frame_ms

            # ── Landmark detection ────────────────────────────────────────────
            if self.is_live:
                self.pose_landmarker.detect_async(mp_image, curr_frame_ms)
                self.hand_landmarker.detect_async(mp_image, curr_frame_ms)
                pose_result = self.latest_pose_result
                hand_result = self.latest_hand_result
            else:
                pose_result = self.pose_landmarker.detect_for_video(mp_image, curr_frame_ms)
                hand_result = self.hand_landmarker.detect_for_video(mp_image, curr_frame_ms)

            # ── Draw landmarks + border ───────────────────────────────────────
            annotated_image = np.copy(rgb)
            if self.draw_landmarks_flag:
                if pose_result and getattr(pose_result, "pose_landmarks", None) and len(pose_result.pose_landmarks) > 0:
                    annotated_image = draw_pose_landmarks_on_image(annotated_image, pose_result)
                if hand_result and getattr(hand_result, "hand_landmarks", None) and len(hand_result.hand_landmarks) > 0:
                    annotated_image = draw_hands_landmarks_on_image(annotated_image, hand_result)

            frame_to_emit = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            border_color  = (56, 207, 19) if self.is_predicting else (224, 160, 22)
            h_img, w_img  = frame_to_emit.shape[:2]
            cv2.rectangle(frame_to_emit, (0, 0), (w_img, h_img), border_color, 30)
            self.frame_signal.emit(frame_to_emit)

            # ── Append to rolling buffers ─────────────────────────────────────
            raw_frame = self.extract_landmarks(pose_result, hand_result)
            processed = process_landmarks(raw_frame)

            if processed is not None:
                self.raw_buffer.append(raw_frame)
                self.proc_buffer.append(processed)
                self.frame_count += 1
                self._frames_since_check += 1

                # ── Periodic motion-based inference ──────────────────────────
                if self.mode == "real_time" and self._frames_since_check >= self.motion_check_interval:
                    self._frames_since_check = 0
                    self._try_motion_inference()

            if not self.is_live:
                time.sleep(1 / fps)

        cap.release()
        if hasattr(self, "pose_landmarker"):
            self.pose_landmarker.close()
        if hasattr(self, "hand_landmarker"):
            self.hand_landmarker.close()

    def stop(self):
        self.running = False
        self.inference_worker.stop()
        self.wait()