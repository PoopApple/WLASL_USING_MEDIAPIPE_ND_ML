import cv2
import time
from collections import deque
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

import numpy as np

from preprocess import process_landmarks
from get_prediction import load_prediction_model, run_inference

class InferenceWorker(QThread):
    result_signal = pyqtSignal(str)
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
            # Wait for new frames or timeout to check running status (100 ms)
            self.condition.wait(self.mutex, 100)
            frames = self.frames_to_process
            self.frames_to_process = None
            pending_model_path = self.new_model_path
            self.mutex.unlock()
            
            # Load or reload the model immediately if the path has changed
            if pending_model_path and pending_model_path != self.loaded_model_path:
                self.model = load_prediction_model(pending_model_path)
                self.loaded_model_path = pending_model_path
            
            if frames is not None and self.running and self.model is not None:
                self.predicting_status_signal.emit(True)
                print_preds = self.config.get("print_predictions", False)
                include_confidence = getattr(self, 'current_mode', "real_time") == "video_testing"
                result = run_inference(self.model, frames, print_preds=print_preds, include_confidence=include_confidence)
                self.predicting_status_signal.emit(False)
                if result:
                    self.result_signal.emit(result)

    def stop(self):
        self.running = False
        self.mutex.lock()
        self.condition.wakeOne()
        self.mutex.unlock()
        self.wait()


def draw_pose_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    for pose_landmarks in pose_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style,
        )

    return annotated_image

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

def draw_hands_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image



class VideoWorker(QThread):
    frame_signal = pyqtSignal(object)
    text_signal = pyqtSignal(str)

    def __init__(self, source, config):
        super().__init__()
        self.source = source
        self.running = False
        self.processing = False
        self.new_source = None
        self.draw_landmarks_flag = config.get("draw_landmarks", True)

        self.seq_len = config["sequence_length"]
        self.interval = config["inference_interval"]

        self.latest_pose_result = None
        self.latest_hand_result = None

        self.frames = deque(maxlen=self.seq_len)
        self.frame_count = 0
        self.is_predicting = False
        self.mode = "real_time"
        self.has_predicted = False

        # --- Motion Energy Segmentation ---
        self.motion_energy_enabled = config.get("motion_energy_segmentation", False)
        self.me_start_thresh = config.get("motion_energy_start_threshold", 0.015)
        self.me_end_thresh = config.get("motion_energy_end_threshold", 0.008)
        self.me_cooldown_frames = config.get("motion_energy_cooldown_frames", 10)
        self.me_min_sign_frames = config.get("motion_energy_min_sign_frames", 15)
        # State: 'idle' | 'signing' | 'cooldown'
        self.me_state = "idle"
        self.me_cooldown_counter = 0
        self.me_sign_frames = deque(maxlen=self.seq_len)  # separate buffer for current sign
        self.me_prev_landmarks = None  # previous frame's key landmark positions

        self.inference_worker = InferenceWorker(config)
        self.inference_worker.result_signal.connect(self.text_signal.emit)
        self.inference_worker.predicting_status_signal.connect(self.set_predicting_status)
        self.inference_worker.start()

        self.pose_model_type = config.get("pose_model", "lite")
        self.pose_model_path_video = f"./mp_models/pose_landmarker_{self.pose_model_type}.task"
        self.pose_model_path_live  = "./mp_models/pose_landmarker_lite.task"  # always lite for speed

        self.hand_model_path = "./mp_models/hand_landmarker.task"
        
        self.setup_landmarkers()

    def set_predicting_status(self, status):
        self.is_predicting = status

    def change_source(self, new_source):
        self.new_source = new_source

    def set_mode(self, mode):
        self.mode = mode

    def pose_callback(self, result, output_image, timestamp_ms):
        self.latest_pose_result = result

    def hand_callback(self, result, output_image, timestamp_ms):
        self.latest_hand_result = result

    def setup_landmarkers(self):
        if hasattr(self, 'pose_landmarker'):
            self.pose_landmarker.close()
        if hasattr(self, 'hand_landmarker'):
            self.hand_landmarker.close()

        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.is_live = isinstance(self.source, int)
        mode = VisionRunningMode.LIVE_STREAM if self.is_live else VisionRunningMode.VIDEO

        # Use lite model for real-time (speed), config model for video testing (accuracy)
        pose_model_path = self.pose_model_path_live if self.is_live else self.pose_model_path_video

        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        if self.is_live:
            hand_options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.hand_model_path),
                running_mode=mode,
                num_hands=2,
                result_callback=self.hand_callback
            )
            pose_options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=pose_model_path),
                running_mode=mode,
                result_callback=self.pose_callback
            )
        else:
            hand_options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.hand_model_path),
                running_mode=mode,
                num_hands=2
            )
            pose_options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=pose_model_path),
                running_mode=mode
            )

        self.pose_landmarker = PoseLandmarker.create_from_options(pose_options)
        self.hand_landmarker = HandLandmarker.create_from_options(hand_options) 
        
    def extract_landmarks(self, pose_result, hand_result):
        curr_frame_data = np.zeros(shape=(75, 4), dtype=np.float32)

        if pose_result and getattr(pose_result, 'pose_landmarks', None) and len(pose_result.pose_landmarks) > 0:
            arr = pose_result.pose_landmarks[0]
            for i in range(33):
                curr_frame_data[i] = [arr[i].x, arr[i].y, arr[i].z, arr[i].visibility]

        if hand_result and getattr(hand_result, 'hand_landmarks', None) and len(hand_result.hand_landmarks) > 0:
            for idx, handedd in enumerate(hand_result.handedness):
                handedd = handedd[0]
                lbl = handedd.display_name
                handLMs = hand_result.hand_landmarks[idx]
                landmark_index = 33 if lbl == "Left" else 54
                for i in range(21):
                    curr_frame_data[landmark_index + i] = [
                        handLMs[i].x,
                        handLMs[i].y,
                        handLMs[i].z,
                        1.0,
                    ]

        return curr_frame_data

    def change_source(self, source):
        self.new_source = source
        self.stop_processing()

    def start_processing(self):
        self.processing = True

    def stop_processing(self):
        self.processing = False
        self.frames.clear()
        self.frame_count = 0

    def run(self):
        cap = cv2.VideoCapture(self.source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        self.running = True
        
        while self.running:
            if self.new_source is not None:
                if self.new_source == self.source:
                    self.new_source = None
                    continue
                cap.release()
                self.source = self.new_source
                self.new_source = None
                self.has_predicted = False
                self.frames.clear()
                self.frame_count = 0
                self.setup_landmarkers()
                cap = cv2.VideoCapture(self.source)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30
                
                if self.mode == "video_testing" and isinstance(self.source, str):
                    self.text_signal.emit("<tr><td>Processing... Please wait.</td></tr>")
                    all_frames = []
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                    emit_every = max(1, total_frames // 60)  # emit ~60 preview frames across whole video
                    frame_idx = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret: break
                        curr_frame_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                        if curr_frame_ms <= getattr(self, 'last_timestamp_ms', -1):
                            curr_frame_ms = getattr(self, 'last_timestamp_ms', -1) + 1
                        self.last_timestamp_ms = curr_frame_ms
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        pose_result = self.pose_landmarker.detect_for_video(mp_image, curr_frame_ms)
                        hand_result = self.hand_landmarker.detect_for_video(mp_image, curr_frame_ms)
                        curr_frame_data = self.extract_landmarks(pose_result, hand_result)
                        processed = process_landmarks(curr_frame_data)
                        if processed is not None:
                            all_frames.append(processed)
                        # Emit every Nth frame so the UI shows the video scrubbing during processing
                        if frame_idx % emit_every == 0:
                            if self.draw_landmarks_flag:
                                annotated = np.copy(rgb)
                                if pose_result and getattr(pose_result, 'pose_landmarks', None) and len(pose_result.pose_landmarks) > 0:
                                    annotated = draw_pose_landmarks_on_image(annotated, pose_result)
                                if hand_result and getattr(hand_result, 'hand_landmarks', None) and len(hand_result.hand_landmarks) > 0:
                                    annotated = draw_hands_landmarks_on_image(annotated, hand_result)
                                self.frame_signal.emit(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                            else:
                                self.frame_signal.emit(frame)
                        frame_idx += 1

                    if all_frames:
                        from preprocess import normalise_lm_arr_temporally
                        arr_padded, mask = normalise_lm_arr_temporally(np.array(all_frames))
                        self.inference_worker.update_frames((arr_padded, mask), self.mode)
                        self.has_predicted = True

                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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

            # Convert for mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            curr_frame_ms = int(time.time() * 1000) if self.is_live else int(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            if not self.processing:
                self.frame_signal.emit(frame)
                if not self.is_live:
                    time.sleep(1 / fps)
                continue

            if curr_frame_ms <= getattr(self, 'last_timestamp_ms', -1):
                curr_frame_ms = getattr(self, 'last_timestamp_ms', -1) + 1
            self.last_timestamp_ms = curr_frame_ms

            if self.is_live:
                # Live mode: use detect_async
                self.pose_landmarker.detect_async(mp_image, curr_frame_ms)
                self.hand_landmarker.detect_async(mp_image, curr_frame_ms)
                
                pose_result = self.latest_pose_result
                hand_result = self.latest_hand_result
            else:
                # Video mode: use detect_for_video
                pose_result = self.pose_landmarker.detect_for_video(mp_image, curr_frame_ms)
                hand_result = self.hand_landmarker.detect_for_video(mp_image, curr_frame_ms)

            annotated_image = np.copy(rgb)
            if getattr(self, 'draw_landmarks_flag', True):
                if pose_result and getattr(pose_result, 'pose_landmarks', None) and len(pose_result.pose_landmarks) > 0:
                    annotated_image = draw_pose_landmarks_on_image(annotated_image, pose_result)
                if hand_result and getattr(hand_result, 'hand_landmarks', None) and len(hand_result.hand_landmarks) > 0:
                    annotated_image = draw_hands_landmarks_on_image(annotated_image, hand_result)
            
            frame_to_emit = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
            if getattr(self, 'is_predicting', False):
                color = (56, 207, 19) # BGR for #13cf38
            else:
                color = (224, 160, 22) # BGR for #16a0e0
                
            h_img, w_img = frame_to_emit.shape[:2]
            cv2.rectangle(frame_to_emit, (0, 0), (w_img, h_img), color, 30) # 30 thickness means 15px inside
            
            self.frame_signal.emit(frame_to_emit)

            curr_frame_data = self.extract_landmarks(pose_result, hand_result)
            processed = process_landmarks(curr_frame_data)

            if processed is not None:
                self.frames.append(processed)
                self.frame_count += 1

                if self.mode == "real_time":
                    if self.motion_energy_enabled:
                        self._update_motion_energy(processed, curr_frame_data)
                    else:
                        # Interval-based fallback
                        if self.frame_count != 0 and self.frame_count % self.interval == 0:
                            from preprocess import normalise_lm_arr_temporally
                            frames_arr = np.array(list(self.frames))
                            arr_padded, mask = normalise_lm_arr_temporally(frames_arr)
                            self.inference_worker.update_frames((arr_padded, mask))

            if not self.is_live:
                time.sleep(1 / fps)

        cap.release()
        if hasattr(self, 'pose_landmarker'):
            self.pose_landmarker.close()
        if hasattr(self, 'hand_landmarker'):
            self.hand_landmarker.close()



    def _compute_motion_energy(self, curr_frame_data):
        """
        Compute mean L2 displacement of key landmarks between this frame and the last.
        Uses pose wrists (idx 15, 16) + all detected hand joints (idx 33-74).
        Landmarks are in normalised 0-1 image coords, so thresholds are scale-independent.
        Returns 0.0 on the first call.
        """
        key_indices = [15, 16] + list(range(33, 75))
        curr_pos = curr_frame_data[key_indices, :2]  # (N, 2) — x, y only

        if self.me_prev_landmarks is None:
            self.me_prev_landmarks = curr_pos
            return 0.0

        energy = float(np.mean(np.linalg.norm(curr_pos - self.me_prev_landmarks, axis=1)))
        self.me_prev_landmarks = curr_pos
        return energy

    def _update_motion_energy(self, processed, curr_frame_data):
        """State machine: IDLE → SIGNING → COOLDOWN → fire inference → IDLE."""
        energy = self._compute_motion_energy(curr_frame_data)

        if self.me_state == "idle":
            if energy >= self.me_start_thresh:
                self.me_state = "signing"
                self.me_sign_frames.clear()
                self.me_sign_frames.append(processed)
                self.me_cooldown_counter = 0

        elif self.me_state == "signing":
            self.me_sign_frames.append(processed)
            if energy < self.me_end_thresh:
                self.me_state = "cooldown"
                self.me_cooldown_counter = 1
            elif len(self.me_sign_frames) >= self.seq_len:
                self._fire_motion_inference()

        elif self.me_state == "cooldown":
            self.me_sign_frames.append(processed)
            if energy >= self.me_start_thresh:
                # Motion resumed — still signing
                self.me_state = "signing"
                self.me_cooldown_counter = 0
            else:
                self.me_cooldown_counter += 1
                if self.me_cooldown_counter >= self.me_cooldown_frames:
                    self._fire_motion_inference()

    def _fire_motion_inference(self):
        """Normalize the collected sign frames and send to inference, then reset."""
        from preprocess import normalise_lm_arr_temporally
        if len(self.me_sign_frames) >= self.me_min_sign_frames:
            frames_arr = np.array(list(self.me_sign_frames))
            arr_padded, mask = normalise_lm_arr_temporally(frames_arr)
            self.inference_worker.update_frames((arr_padded, mask))
        # Reset state regardless
        self.me_state = "idle"
        self.me_sign_frames.clear()
        self.me_cooldown_counter = 0
        self.me_prev_landmarks = None

    def stop(self):
        self.running = False
        self.inference_worker.stop()
        self.wait()