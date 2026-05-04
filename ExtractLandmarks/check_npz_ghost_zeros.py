"""
check_npz_ghost_zeros.py
========================
Verifies the absent-hand normalisation bug is NOT present.

The bug: when a hand landmark has visibility=0, its raw coords are (0,0,0).
After normalisation → (0 - center) / length = -center/length (non-zero ghost).

Fix: if visibility==0, keep the landmark at exactly (0,0,0,0).

This script checks raw .npy files, computes the exact expected ghost value
(-shoulder_center / shoulder_length) per frame, and checks whether any
absent landmark matches that ghost pattern.

Usage:
    uv run check_npz_ghost_zeros.py
"""

import os
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── CONFIG ─────────────────────────────────────────────────────────
# Raw .npy files (before normalisation) — needed to compute shoulder center
LANDMARKS_DIR = "/run/media/aryan/PoopiDrive/projects_linux/microsoft asl citizen/landmarks/"
N_WORKERS     = 8
MATCH_THRESH  = 0.01    # tolerance for floating point comparison
MAX_FILES     = 5000    # cap for speed (set to None for full scan)

# In the raw .npy, landmark 11 = left shoulder, 12 = right shoulder
LEFT_SHOULDER_IDX  = 11
RIGHT_SHOULDER_IDX = 12

# Hand landmark index range in raw .npy (0-indexed): 33-74
HAND_SLICE = slice(33, 75)
# ───────────────────────────────────────────────────────────────────


def check_file(npy_path):
    try:
        arr          = np.load(npy_path, mmap_mode="r")
        total_frames = int(arr[0, 0, 0])
        frames       = arr[1:total_frames + 1]   # (T, 75, 4)

        ghost_count  = 0
        total_absent = 0
        example      = None

        for frame in frames:
            left_sh  = frame[LEFT_SHOULDER_IDX,  :3]
            right_sh = frame[RIGHT_SHOULDER_IDX, :3]
            center   = (left_sh + right_sh) / 2.0
            length   = np.linalg.norm(left_sh - right_sh) + 1e-8

            # Expected ghost values if bug is present:
            # ghost_x = (0 - center_x) / length = -center_x / length
            ghost_xyz = -center / length    # shape (3,)

            hand_lms = frame[HAND_SLICE]    # (42, 4)
            for lm in hand_lms:
                vis = lm[3]
                if vis > 0.0:
                    continue    # only check absent landmarks
                total_absent += 1

                xyz   = lm[:3]
                # Ghost: xyz should ≈ ghost_xyz if bug present
                # Fixed: xyz should be (0, 0, 0)
                diff  = np.abs(xyz - ghost_xyz)
                if np.all(diff < MATCH_THRESH) and np.any(np.abs(ghost_xyz) > 0.01):
                    ghost_count += 1
                    if example is None:
                        example = {
                            "lm_xyz":    xyz.tolist(),
                            "ghost_xyz": ghost_xyz.tolist(),
                            "diff":      diff.tolist(),
                        }

        return {"total_absent": total_absent, "ghost_count": ghost_count, "example": example}
    except Exception as e:
        return {"error": str(e)}


def collect_npy_paths(landmarks_dir, max_files):
    paths = []
    for word in os.listdir(landmarks_dir):
        word_dir = os.path.join(landmarks_dir, word)
        if not os.path.isdir(word_dir):
            continue
        for fname in os.listdir(word_dir):
            if fname.endswith(".npy"):
                paths.append(os.path.join(word_dir, fname))
        if max_files and len(paths) >= max_files:
            break
    return paths[:max_files] if max_files else paths


def main():
    if not os.path.isdir(LANDMARKS_DIR):
        print(f"[ERROR] LANDMARKS_DIR not found: {LANDMARKS_DIR}")
        sys.exit(1)

    print(f"Scanning raw .npy files in {LANDMARKS_DIR} ...")
    paths = collect_npy_paths(LANDMARKS_DIR, MAX_FILES)
    print(f"  Checking {len(paths)} files with {N_WORKERS} threads ...\n")

    total_absent     = 0
    total_ghost      = 0
    files_with_ghost = 0
    errors           = 0
    examples         = []
    done             = 0

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(check_file, p): p for p in paths}
        for fut in as_completed(futures):
            done += 1
            if done % 1000 == 0:
                print(f"  ... {done}/{len(paths)} checked")
            result = fut.result()
            if "error" in result:
                errors += 1
                continue
            total_absent += result["total_absent"]
            total_ghost  += result["ghost_count"]
            if result["ghost_count"] > 0:
                files_with_ghost += 1
                if len(examples) < 5:
                    examples.append({
                        "file":    os.path.basename(futures[fut]),
                        "count":   result["ghost_count"],
                        "detail":  result["example"],
                    })

    print("=" * 60)
    print(f"Files scanned      : {len(paths)}  ({errors} errors)")
    print(f"Absent landmarks   : {total_absent:,}  (vis == 0)")
    print(f"Ghost matches found: {total_ghost:,}")
    print(f"  (match = absent lm xyz ≈ -shoulder_center/shoulder_length)")
    print(f"Files affected     : {files_with_ghost}")
    print()

    if total_ghost == 0:
        print("✅  BUG IS FIXED — no ghost shift values detected.")
    else:
        rate = 100.0 * total_ghost / total_absent if total_absent else 0
        print(f"❌  BUG PRESENT — {rate:.3f}% of absent landmarks have ghost values.")
        print(f"\nFirst {len(examples)} affected files:")
        for e in examples:
            d = e["detail"]
            print(f"\n  {e['file']}  ({e['count']} ghosts)")
            print(f"    lm xyz    : {[f'{v:.4f}' for v in d['lm_xyz']]}")
            print(f"    ghost_xyz : {[f'{v:.4f}' for v in d['ghost_xyz']]}")
            print(f"    diff      : {[f'{v:.4f}' for v in d['diff']]}")


if __name__ == "__main__":
    main()
