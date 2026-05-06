import os
import sys
import time
import numpy as np

# ── PATH SETUP ────────────────────────────────────────────────────────────────
REPO_ROOT = "/home/aryan/projects_linux/WLASL_USING_MEDIAPIPE_ND_ML"
sys.path.append(os.path.join(REPO_ROOT, "UI"))
sys.path.append(os.path.join(REPO_ROOT, "ExtractLandmarks"))

from preprocess import process_landmarks
from motion_detection import compute_motion_per_frame

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
LANDMARK_FOLDER = "/run/media/aryan/PoopiDrive/projects_linux/microsoft asl citizen/landmarks/"
NUM_WORDS = 10
WARMUP_RUNS = 20
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"--- Pipeline Logic Benchmarking (Spatial Norm + Motion) ---")
    
    words = [d for d in os.listdir(LANDMARK_FOLDER) if os.path.isdir(os.path.join(LANDMARK_FOLDER, d))]
    if not words:
        print(f"No word directories found in {LANDMARK_FOLDER}")
        return
    
    words = words[:NUM_WORDS]
    
    norm_latencies = []
    motion_latencies = []
    
    total_runs = 0
    
    for word_idx, word in enumerate(words):
        word_dir = os.path.join(LANDMARK_FOLDER, word)
        npy_files = [f for f in os.listdir(word_dir) if f.endswith('.npy')]
        
        for npy_idx, npy in enumerate(npy_files[:2]):  # Max 2 files per word
            v_path = os.path.join(word_dir, npy)
            try:
                data = np.load(v_path)
            except Exception as e:
                print(f"Error loading {v_path}: {e}")
                continue
            
            # 1. Bench Spatial Normalisation (per frame)
            f_latencies = []
            for frame in data:
                start = time.perf_counter()
                process_landmarks(frame)
                end = time.perf_counter()
                f_latencies.append((end - start) * 1000)
            
            avg_f = np.mean(f_latencies)
            norm_latencies.extend(f_latencies)
            
            # 2. Bench Motion Detection (per sequence)
            start_m = time.perf_counter()
            compute_motion_per_frame(data)
            end_m = time.perf_counter()
            m_ms = (end_m - start_m) * 1000
            motion_latencies.append(m_ms)
            
            load_suffix = " (Warmup)" if total_runs == 0 else ""
            print(f"Word: {word:<12} | File: {npy[:15]:<15} | Avg Norm: {avg_f:6.4f}ms/f | Motion: {m_ms:6.2f}ms{load_suffix}")
            total_runs += 1

    print(f"\n--- Grouped Results ---")
    if norm_latencies:
        print(f"Spatial Normalisation (per frame):")
        print(f"  Avg (All):        {np.mean(norm_latencies):.4f} ms")
        print(f"  Avg (Subsequent): {np.mean(norm_latencies[1:]):.4f} ms")
    if motion_latencies:
        print(f"\nMotion Energy Calculation (per sequence):")
        print(f"  Avg (All):        {np.mean(motion_latencies):.2f} ms")
        print(f"  Avg (Subsequent): {np.mean(motion_latencies[1:]):.2f} ms")
    
if __name__ == "__main__":
    main()
