import os
import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
VIDEO_FOLDER = "/run/media/aryan/PoopiDrive/projects_linux/microsoft asl citizen/ASL_Citizen/videos/"
POSE_MODEL_PATH = "/home/aryan/projects_linux/WLASL_USING_MEDIAPIPE_ND_ML/UI/mp_models/pose_landmarker_lite.task"
HAND_MODEL_PATH = "/home/aryan/projects_linux/WLASL_USING_MEDIAPIPE_ND_ML/UI/mp_models/hand_landmarker.task"
NUM_VIDEOS = 5
WARMUP_FRAMES = 50
# ─────────────────────────────────────────────────────────────────────────────

def setup_landmarkers():
    base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    base_options_hand = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    hand_options = vision.HandLandmarkerOptions(
        base_options=base_options_hand,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
    
    return pose_landmarker, hand_landmarker

def main():
    print(f"--- MediaPipe Benchmarking ---")
    print(f"Loading models...")
    pose_landmarker, hand_landmarker = setup_landmarkers()
    
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.webm'))]
    if not video_files:
        print(f"No videos found in {VIDEO_FOLDER}")
        return
    
    video_files = video_files[:NUM_VIDEOS]
    
    latencies = []
    total_frames = 0
    first_frame_latency = None
    global_frame_idx = 0
    
    for v_file in video_files:
        v_path = os.path.join(VIDEO_FOLDER, v_file)
        cap = cv2.VideoCapture(v_path)
        print(f"Processing: {v_file}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Use ms timestamp (must be increasing)
            ts_ms = global_frame_idx * 33 # Assume 30fps for timestamp gap
            
            start_time = time.perf_counter()
            
            # Run Pose
            pose_landmarker.detect_for_video(mp_image, ts_ms)
            # Run Hands
            hand_landmarker.detect_for_video(mp_image, ts_ms)
            
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            
            if first_frame_latency is None:
                first_frame_latency = latency_ms
                print(f"  First Frame (Model Load): {latency_ms:.2f} ms")
            
            latencies.append(latency_ms)
            total_frames += 1
            global_frame_idx += 1
            
        cap.release()

    if latencies:
        # We consider "loaded" after the first frame, but let's see the distribution
        avg_latency = np.mean(latencies)
        avg_subsequent = np.mean(latencies[1:]) if len(latencies) > 1 else avg_latency
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        
        print(f"\n--- Final Results ---")
        print(f"  Initial Load Latency (1st frame): {first_frame_latency:.2f} ms")
        print(f"  Average Latency (All frames):     {avg_latency:.2f} ms")
        print(f"  Average Latency (Subsequent):     {avg_subsequent:.2f} ms")
        print(f"  P50 Latency:                      {p50:.2f} ms")
        print(f"  P95 Latency:                      {p95:.2f} ms")
        print(f"  Total Frames Measured:            {len(latencies)}")
    else:
        print("No frames processed.")

    pose_landmarker.close()
    hand_landmarker.close()

if __name__ == "__main__":
    main()
