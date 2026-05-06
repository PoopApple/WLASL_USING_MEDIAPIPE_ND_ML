"""
worker.py
=========
Pipeline for real-time and video-testing ASL inference.

Real-time flow
──────────────
  1. Every camera frame → MediaPipe extraction → spatial normalise
  2. Push raw (75,4) + processed (64,4) into rolling deques (buffer_size frames)
  3. Every motion_check_interval frames → run motion segmentation on raw buffer
       • If a complete sign segment is found → temporally normalise + infer
       • If NOT fired for periodic_fallback_interval consecutive checks
         → force inference on the last 64 processed frames
  4. InferenceWorker runs model.predict() in a background thread (non-blocking)

Video-testing flow
──────────────────
  • Process ALL video frames at once (landmark extraction + spatial normalise)
  • Run motion detection on the full raw array to find the best sign segment
  • Temporally normalise that segment to 64 frames
  • Run TTA inference (offline, speed not critical)
  • Emit motion_energy_signal with energies + segment info for the UI bar chart
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
    result_signal            = pyqtSignal(str)
    predicting_status_signal = pyqtSignal(bool)

    def __init__(self, config):
        super().__init__()
        self.config            = config
        self.running           = False
        self.frames_to_process = None
        self.current_mode      = "real_time"
        self.mutex             = QMutex()
        self.condition         = QWaitCondition()
        self.model             = None
        self.loaded_model_path = None
        self.new_model_path    = config.get("prediction_model", "")

    def update_frames(self, frames, mode: str = "real_time"):
        self.mutex.lock()
        self.frames_to_process = frames
        self.current_mode      = mode
        self.condition.wakeOne()
        self.mutex.unlock()

    def update_model_path(self, new_path: str):
        self.mutex.lock()
        self.new_model_path = new_path
        self.condition.wakeOne()
        self.mutex.unlock()

    def run(self):
        self.running = True
        while self.running:
            self.mutex.lock()
            self.condition.wait(self.mutex, 100)
            frames             = self.frames_to_process
            self.frames_to_process = None
            pending_path       = self.new_model_path
            mode               = self.current_mode
            self.mutex.unlock()

            if pending_path and pending_path != self.loaded_model_path:
                self.model             = load_prediction_model(pending_path)
                self.loaded_model_path = pending_path

            if frames is not None and self.running and self.model is not None:
                self.predicting_status_signal.emit(True)
                print_preds = self.config.get("print_predictions", False)
                result = run_inference(self.model, frames, mode=mode, print_preds=print_preds)
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
    annotated   = np.copy(rgb_image)
    style       = drawing_styles.get_default_pose_landmarks_style()
    conn_style  = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
    for lms in detection_result.pose_landmarks:
        drawing_utils.draw_landmarks(
            image=annotated, landmark_list=lms,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=style, connection_drawing_spec=conn_style,
        )
    return annotated


mp_hands          = mp.tasks.vision.HandLandmarksConnections
mp_drawing        = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN                = 10
FONT_SIZE             = 1
FONT_THICKNESS        = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)


def draw_hands_landmarks_on_image(rgb_image, detection_result):
    annotated = np.copy(rgb_image)
    for idx in range(len(detection_result.hand_landmarks)):
        lms       = detection_result.hand_landmarks[idx]
        handedness = detection_result.handedness[idx]
        mp_drawing.draw_landmarks(
            annotated, lms, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
        height, width, _ = annotated.shape
        xs = [lm.x for lm in lms]; ys = [lm.y for lm in lms]
        cv2.putText(
            annotated, handedness[0].display_name,
            (int(min(xs) * width), int(min(ys) * height) - MARGIN),
            cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS, cv2.LINE_AA,
        )
    return annotated


# ─────────────────────────────────────────────────────────────────────────────
class VideoWorker(QThread):
    frame_signal        = pyqtSignal(object)
    text_signal         = pyqtSignal(str)
    motion_energy_signal = pyqtSignal(object)   # emits dict with energies + segments

    def __init__(self, source, config):
        super().__init__()
        self.source  = source
        self.config  = config
        self.running    = False
        self.processing = False
        self.new_source = None

        self.draw_landmarks_flag = config.get("draw_landmarks", True)
        self.target_frames       = config.get("target_frames", 64)

        # ── Rolling buffers ───────────────────────────────────────────────────
        self.buffer_size  = config.get("buffer_size", 256)
        self.raw_buffer   = deque(maxlen=self.buffer_size)   # (75, 4)
        self.proc_buffer  = deque(maxlen=self.buffer_size)   # (64, 4)

        # ── Trigger cadences ──────────────────────────────────────────────────
        self.motion_check_interval    = config.get("motion_check_interval", 5)
        self.periodic_fallback_interval = config.get("periodic_fallback_interval", 30)
        self._frames_since_check      = 0
        self._no_fire_count           = 0   # consecutive motion-check misses

        # ── Temporal windows ──────────────────────────────────────────────────
        self.temporal_windows = sorted(config.get("temporal_windows", [64, 128, 256]))

        # ── Motion detection params ───────────────────────────────────────────
        self.md_start_thresh = config.get("motion_start_threshold", MD_DEFAULTS["start_thresh"])
        self.md_end_thresh   = config.get("motion_end_threshold",   MD_DEFAULTS["end_thresh"])
        self.md_cooldown     = config.get("motion_cooldown_frames",  MD_DEFAULTS["cooldown"])
        self.md_min_frames   = config.get("motion_min_sign_frames",  MD_DEFAULTS["min_frames"])
        self.md_method       = config.get("motion_method",           MD_DEFAULTS["method"])
        self.md_top_k        = config.get("motion_top_k",            MD_DEFAULTS["top_k"])
        self.md_tip_weight   = config.get("motion_tip_weight",       MD_DEFAULTS["tip_weight"])
        self.md_head_pad     = config.get("motion_head_pad",         MD_DEFAULTS["head_pad"])
        self.md_vis_thresh   = config.get("motion_vis_thresh",       MD_DEFAULTS["vis_thresh"])

        # ── Misc state ────────────────────────────────────────────────────────
        self.latest_pose_result = None
        self.latest_hand_result = None
        self.frame_count        = 0
        self.is_predicting      = False
        self.mode               = "real_time"
        self.has_predicted      = False
        self._last_fired_abs_end = -999

        # ── MediaPipe paths ───────────────────────────────────────────────────
        pose_type = config.get("pose_model", "lite")
        self.pose_model_path_video = f"./mp_models/pose_landmarker_{pose_type}.task"
        self.pose_model_path_live  = "./mp_models/pose_landmarker_lite.task"
        self.hand_model_path       = "./mp_models/hand_landmarker.task"

        self.inference_worker = InferenceWorker(config)
        self.inference_worker.result_signal.connect(self.text_signal.emit)
        self.inference_worker.predicting_status_signal.connect(self._set_predicting)
        self.inference_worker.start()

        self.setup_landmarkers()

    # ── Status helpers ────────────────────────────────────────────────────────
    def _set_predicting(self, status: bool):
        self.is_predicting = status

    def set_mode(self, mode: str):
        self.mode = mode

    def start_processing(self):
        self.processing = True

    def stop_processing(self):
        self.processing = False
        self.raw_buffer.clear()
        self.proc_buffer.clear()
        self.frame_count        = 0
        self._frames_since_check = 0
        self._no_fire_count     = 0

    def change_source(self, source):
        self.new_source = source
        self.stop_processing()

    # ── Async callbacks ───────────────────────────────────────────────────────

    # ── MediaPipe setup ───────────────────────────────────────────────────────
    def setup_landmarkers(self):
        for attr in ("pose_landmarker", "hand_landmarker"):
            if hasattr(self, attr):
                getattr(self, attr).close()

        BaseOptions       = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        self.is_live      = isinstance(self.source, int)
        mode_mp           = VisionRunningMode.VIDEO
        pose_path         = self.pose_model_path_live if self.is_live else self.pose_model_path_video

        PoseLandmarker        = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        HandLandmarker        = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        pose_opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_path),
            running_mode=mode_mp,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4)
        hand_opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.hand_model_path),
            running_mode=mode_mp, 
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4)

        self.pose_landmarker = PoseLandmarker.create_from_options(pose_opts)
        self.hand_landmarker = HandLandmarker.create_from_options(hand_opts)

    # ── Landmark extraction ───────────────────────────────────────────────────
    def extract_landmarks(self, pose_result, hand_result) -> np.ndarray:
        """Returns raw (75, 4) float32 array."""
        out = np.zeros((75, 4), dtype=np.float32)
        if pose_result and getattr(pose_result, "pose_landmarks", None) and pose_result.pose_landmarks:
            for i, lm in enumerate(pose_result.pose_landmarks[0]):
                out[i] = [lm.x, lm.y, lm.z, lm.visibility]
        if hand_result and getattr(hand_result, "hand_landmarks", None) and hand_result.hand_landmarks:
            for idx, handedness in enumerate(hand_result.handedness):
                lbl   = handedness[0].display_name          # "Left" | "Right"
                start = 33 if lbl == "Left" else 54
                for i, lm in enumerate(hand_result.hand_landmarks[idx]):
                    out[start + i] = [lm.x, lm.y, lm.z, 1.0]
        return out

    # ── Motion inference ──────────────────────────────────────────────────────
    def _run_motion_segmentation(self, raw_arr):
        """Run segmentation; return (energies, segments, best_seg) or (None,None,None)."""
        if len(raw_arr) < self.md_min_frames:
            return None, None, None
        energies = compute_motion_per_frame(
            raw_arr, method=self.md_method, top_k=self.md_top_k,
            tip_weight=self.md_tip_weight, vis_thresh=self.md_vis_thresh,
        )

        segments = find_motion_segments(
            energies, start_thresh=self.md_start_thresh, end_thresh=self.md_end_thresh,
            cooldown_frames=self.md_cooldown, min_sign_frames=self.md_min_frames,
        )
        best = find_longest_segment(segments)
        return energies, segments, best

    def _try_motion_inference(self) -> bool:
        """
        Attempt to detect a completed sign segment and fire inference.

        Returns True if inference was dispatched, False otherwise.

        Window strategy: pick the SMALLEST temporal_window that fully contains
        the sign+head_pad.  This minimises idle frames included in the 64-frame
        normalised input.

        Duplicate-fire guard: skip if the detected segment ends within
        2 × motion_check_interval frames of the last fired segment.
        """
        raw_arr  = np.array(self.raw_buffer)   # (T, 75, 4)
        proc_arr = np.array(self.proc_buffer)  # (T, 64, 4)
        T = len(raw_arr)

        energies, segments, _ = self._run_motion_segmentation(raw_arr)
        
        # In real-time, emit energy for visualizer regardless of trigger
        if self.mode == "real_time" and energies is not None:
            self.motion_energy_signal.emit({
                "energies":    energies.tolist(),
                "segments":    segments or [],
                "best_seg":    None,
                "total_frames": T,
                "start_thresh": self.md_start_thresh,
            })

        if not segments:
            return False

        # Find the LATEST segment that hasn't been fired yet.
        # Absolute index of buffer start = total frames added - current buffer size
        abs_buffer_start = self.frame_count - T
        
        best_seg = None
        for start, end in reversed(segments):
            # Enforce CLOSED segments: skip if the segment reaches the very end of the buffer
            if self.mode == "real_time" and end == T:
                continue

            abs_end = abs_buffer_start + end
            if abs_end > self._last_fired_abs_end + self.motion_check_interval:
                best_seg = (start, end)
                break

        if best_seg is None:
            return False

        seg_start, seg_end = best_seg
        abs_end = abs_buffer_start + seg_end

        # Head-pad cleanly, do NOT jump to arbitrary temporal windows.
        # Extract the exact detected segment and rely strictly on Temporal Normalization 
        # to apply zero-padding (matching training behavior).
        padded_start = max(0, seg_start - self.md_head_pad)
        window_frames = proc_arr[padded_start:seg_end]

        if len(window_frames) < self.md_min_frames:
            return False

        arr_padded, mask = normalise_lm_arr_temporally(window_frames, self.target_frames)
        print(f"[Motion] Closed seg={seg_start}:{seg_end} (abs {abs_end})  pad={padded_start}"
              f"  → frames={len(window_frames)}  → {self.target_frames}f zero-padded")
        self._last_fired_abs_end = abs_end
        abs_start = abs_buffer_start + padded_start
        self.inference_worker.update_frames((arr_padded, mask, abs_start, abs_end), "real_time")
        return True

    def _fire_periodic_inference(self):
        """
        Fallback: take the last target_frames processed frames and infer.
        This guarantees output even if motion detection never triggers.
        """
        proc_arr = np.array(self.proc_buffer)
        if len(proc_arr) < self.md_min_frames:
            return
        # Take the latest target_frames frames
        frames = proc_arr[-self.target_frames:]
        arr_padded, mask = normalise_lm_arr_temporally(frames, self.target_frames)
        abs_end = self.frame_count
        abs_start = max(0, abs_end - self.target_frames)
        print(f"[Fallback] forcing inference on last {len(frames)} proc frames (abs {abs_start}:{abs_end})")
        self.inference_worker.update_frames((arr_padded, mask, abs_start, abs_end), "real_time")

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Prevent OpenCV from queuing delayed frames
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.running = True

        while self.running:
            # ── Source change ─────────────────────────────────────────────────
            if self.new_source is not None:
                if self.new_source == self.source:
                    self.new_source = None
                    continue
                cap.release()
                self.source     = self.new_source
                self.new_source = None
                self.has_predicted = False
                self.raw_buffer.clear()
                self.proc_buffer.clear()
                self.frame_count          = 0
                self._frames_since_check  = 0
                self._no_fire_count       = 0
                self._last_fired_abs_end  = -999
                self.setup_landmarkers()
                cap = cv2.VideoCapture(self.source)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

                # ── Video testing: process whole file at once ─────────────────
                if self.mode == "video_testing" and isinstance(self.source, str):
                    self.text_signal.emit(
                        '{"mode":"video_testing","slices":{"Full":[["Processing…",0]],'
                        '"First Half":[],"Second Half":[]}}'
                    )
                    all_raw  = []
                    all_proc = []
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                    emit_every   = max(1, total_frames // 60)
                    frame_idx    = 0
                    last_ts      = -1

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        curr_ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                        if curr_ts <= last_ts:
                            curr_ts = last_ts + 1
                        last_ts = curr_ts

                        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if self.is_live:
                            rgb = cv2.resize(rgb, (640, 480))
                        
                        mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        p_res    = self.pose_landmarker.detect_for_video(mp_img, curr_ts)
                        h_res    = self.hand_landmarker.detect_for_video(mp_img, curr_ts)
                        raw      = self.extract_landmarks(p_res, h_res)
                        proc     = process_landmarks(raw)
                        if proc is not None:
                            all_raw.append(raw)
                            all_proc.append(proc)

                        if frame_idx % emit_every == 0:
                            if self.draw_landmarks_flag:
                                ann = np.copy(rgb)
                                if p_res and getattr(p_res, "pose_landmarks", None) and p_res.pose_landmarks:
                                    ann = draw_pose_landmarks_on_image(ann, p_res)
                                if h_res and getattr(h_res, "hand_landmarks", None) and h_res.hand_landmarks:
                                    ann = draw_hands_landmarks_on_image(ann, h_res)
                                self.frame_signal.emit(cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))
                            else:
                                self.frame_signal.emit(frame)
                        frame_idx += 1

                    if all_proc:
                        raw_arr  = np.array(all_raw)
                        proc_arr = np.array(all_proc)
                        N        = len(proc_arr)

                        # Motion detection on whole video
                        energies, segments, best_seg = self._run_motion_segmentation(raw_arr)

                        # Emit motion energy for UI bar
                        if energies is not None:
                            self.motion_energy_signal.emit({
                                "energies":    energies.tolist(),
                                "segments":    segments or [],
                                "best_seg":    best_seg,
                                "total_frames": N,
                                "start_thresh": self.md_start_thresh,
                            })

                        # Choose best segment or fall back to full video
                        if best_seg is not None:
                            s_start, s_end = best_seg
                            padded_start   = max(0, s_start - self.md_head_pad)
                            frames_to_use  = proc_arr[padded_start:s_end]
                            abs_start, abs_end = padded_start, s_end
                            print(f"[VideoTest] using motion seg {padded_start}:{s_end}"
                                  f"  ({len(frames_to_use)} frames)")
                        else:
                            frames_to_use = proc_arr
                            abs_start, abs_end = 0, N
                            print(f"[VideoTest] no segment found, using all {N} frames")

                        arr_padded, mask = normalise_lm_arr_temporally(
                            frames_to_use, self.target_frames)
                        self.inference_worker.update_frames((arr_padded, mask, abs_start, abs_end), "video_testing")
                        self.has_predicted = True

                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # ── Normal per-frame loop ─────────────────────────────────────────
            frame_skip = self.config.get("frame_skip", 0)
            if self.is_live and frame_skip > 0:
                for _ in range(frame_skip):
                    cap.grab()
            ret, frame = cap.read()
            if not ret:
                if isinstance(self.source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    self.text_signal.emit('{"mode":"error","message":"Camera unavailable."}')
                    time.sleep(0.5)
                continue

            if self.mode == "video_testing" and isinstance(self.source, str):
                self.frame_signal.emit(frame)
                time.sleep(1 / fps)
                continue

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.is_live:
                rgb = cv2.resize(rgb, (640, 480))
            
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms  = int(time.time() * 1000) if self.is_live else int(cap.get(cv2.CAP_PROP_POS_MSEC))

            if not self.processing:
                self.frame_signal.emit(frame)
                if not self.is_live:
                    time.sleep(1 / fps)
                continue

            if ts_ms <= getattr(self, "_last_ts", -1):
                ts_ms = getattr(self, "_last_ts", -1) + 1
            self._last_ts = ts_ms

            # Landmark detection
            p_res = self.pose_landmarker.detect_for_video(mp_img, ts_ms)
            h_res = self.hand_landmarker.detect_for_video(mp_img, ts_ms)

            # Draw + border overlay
            ann = np.copy(rgb)
            if self.draw_landmarks_flag:
                if p_res and getattr(p_res, "pose_landmarks", None) and p_res.pose_landmarks:
                    ann = draw_pose_landmarks_on_image(ann, p_res)
                if h_res and getattr(h_res, "hand_landmarks", None) and h_res.hand_landmarks:
                    ann = draw_hands_landmarks_on_image(ann, h_res)
            frame_out    = cv2.cvtColor(ann, cv2.COLOR_RGB2BGR)
            border_color = (56, 207, 19) if self.is_predicting else (224, 160, 22)
            h_img, w_img = frame_out.shape[:2]
            cv2.rectangle(frame_out, (0, 0), (w_img, h_img), border_color, 30)
            self.frame_signal.emit(frame_out)

            # Buffer append
            raw_frame = self.extract_landmarks(p_res, h_res)
            proc      = process_landmarks(raw_frame)
            if proc is not None:
                self.raw_buffer.append(raw_frame)
                self.proc_buffer.append(proc)
                self.frame_count         += 1
                self._frames_since_check += 1

                if self.mode == "real_time" and self._frames_since_check >= self.motion_check_interval:
                    self._frames_since_check = 0
                    fired = self._try_motion_inference()
                    if fired:
                        self._no_fire_count = 0
                    else:
                        self._no_fire_count += 1
                        if self._no_fire_count >= self.periodic_fallback_interval:
                            self._no_fire_count = 0
                            self._fire_periodic_inference()

            if not self.is_live:
                time.sleep(1 / fps)

        cap.release()
        for attr in ("pose_landmarker", "hand_landmarker"):
            if hasattr(self, attr):
                getattr(self, attr).close()

    def stop(self):
        self.running = False
        self.inference_worker.stop()
        self.wait()