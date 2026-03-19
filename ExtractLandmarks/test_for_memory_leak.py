import mediapipe as mp
print(mp.__version__)
import cv2
import numpy as np
import time
import psutil
import os
import gc

# Global counter to track if results are being processed
processed_count = 0

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def result_callback(result: mp.tasks.vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global processed_count
    if result.segmentation_masks:
        _ = result.segmentation_masks[0].numpy_view()
    
    processed_count += 1

def run_async_leak_test(image_path='pose.jpeg', iterations=1000):
    global processed_count
    
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='./vision_models/pose_landmarker_heavy.task',delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=VisionRunningMode.LIVE_STREAM, # Required for detect_async
        output_segmentation_masks=True,
        result_callback=result_callback
    )

    raw_image = cv2.imread(image_path)
    raw_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    
    start_mem = get_memory_usage()
    print(f"Initial Memory: {start_mem:.2f} MB")

    with PoseLandmarker.create_from_options(options) as landmarker:
        for i in range(iterations):
            # Create a unique mp.Image for every call to simulate a camera stream
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_rgb)
            
            # timestamp_ms must be monotonically increasing
            timestamp = int(time.time() * 1000) + i 
            
            landmarker.detect_async(mp_image, timestamp)

            if i % 100 == 0:
                gc.collect()
                print(f"Sent {i} frames | Processed {processed_count} | Memory: {get_memory_usage():.2f} MB")
            
            # Small sleep to simulate real-time and prevent the C++ queue from exploding
            time.sleep(0.01)
    gc.collect()
    print(f"\nFinal Memory: {get_memory_usage():.2f} MB")
    print(f"Total processed: {processed_count}")

if __name__ == "__main__":
    run_async_leak_test()
    run_async_leak_test()
    run_async_leak_test()
