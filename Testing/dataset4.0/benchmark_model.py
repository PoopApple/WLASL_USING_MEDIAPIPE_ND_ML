import os
import sys
import time
import numpy as np
import tensorflow as tf

# ── PATH SETUP ────────────────────────────────────────────────────────────────
REPO_ROOT = "/home/aryan/projects_linux/WLASL_USING_MEDIAPIPE_ND_ML"
sys.path.append(os.path.join(REPO_ROOT, "UI"))

from get_prediction import load_prediction_model, run_inference

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
MODEL_PATH = "/home/aryan/projects_linux/WLASL_USING_MEDIAPIPE_ND_ML/UI/ASL_Models/asl_bigru_bigger_v1_aug_allw_06-05-26__03-47_alltimebest.keras"
NPZ_FOLDER = "/home/aryan/projects_linux/WLASL_USING_MEDIAPIPE_ND_ML/ExtractLandmarks/dataset4.0/landmarks_npz/"
NUM_SAMPLES = 20
WARMUP_RUNS = 10
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"--- Model Inference Benchmarking ---")
    print(f"Loading model: {os.path.basename(MODEL_PATH)}...")
    model = load_prediction_model(MODEL_PATH)
    
    # for f in os.walk(NPZ_FOLDER):
    #     print(f)
    

    npz_files=[]

    for path_,folders,files in os.walk(NPZ_FOLDER):
        for file in files:
            if file.endswith(".npz"):
                filepath = os.path.join(path_,file)
                npz_files.append(filepath)


    # npz_files = [f for f in os.walk(NPZ_FOLDER) if f[2].endswith('.npz')]
    if not npz_files:
        print(f"No NPZ files found in {NPZ_FOLDER}")
        return
    
    samples = npz_files[:NUM_SAMPLES]
    
    rt_latencies = []
    vt_latencies = []
    
    total_runs = 0
    
    for i, npz_file in enumerate(samples):
        path = os.path.join(NPZ_FOLDER, npz_file)
        data = np.load(path)
        # Expected keys: 'arr' or 'landmarks', 'mask'
        arr = data.get('data')
        mask = data.get('mask')
        
        if arr is None or mask is None:
            continue
            
        sequence = (arr, mask)
        
        # 1. Real-Time Inference (1 slice, no TTA)
        start_rt = time.perf_counter()
        run_inference(model, sequence, mode="real_time")
        end_rt = time.perf_counter()
        rt_ms = (end_rt - start_rt) * 1000
        rt_latencies.append(rt_ms)
            
        # 2. Video-Testing Inference (3 slices, with TTA)
        start_vt = time.perf_counter()
        run_inference(model, sequence, mode="video_testing")
        end_vt = time.perf_counter()
        vt_ms = (end_vt - start_vt) * 1000
        vt_latencies.append(vt_ms)
        
        load_suffix = " (Model Load/Warmup)" if i == 0 else ""
        print(f"Sample {i+1:02d} | {npz_file[:20]:<20} | RT: {rt_ms:6.2f}ms | VT: {vt_ms:6.2f}ms{load_suffix}")
            
        total_runs += 1

    print(f"\n--- Grouped Results ({len(rt_latencies)} samples) ---")
    if rt_latencies:
        print(f"Real-Time Mode (1 pass):")
        print(f"  First Run:        {rt_latencies[0]:.2f} ms")
        print(f"  Avg (All):        {np.mean(rt_latencies):.2f} ms")
        print(f"  Avg (Subsequent): {np.mean(rt_latencies[1:]):.2f} ms")
        print(f"  P95:              {np.percentile(rt_latencies, 95):.2f} ms")
    if vt_latencies:
        print(f"\nVideo-Testing Mode (6 pass TTA):")


if __name__ == '__main__':
    main()
