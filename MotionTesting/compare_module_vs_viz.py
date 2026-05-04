"""
compare_module_vs_viz.py
========================
Validates that motion_detection.py produces identical results to the
inline logic in motion_threshold_viz.py across 100+ videos.

Prints a summary and details of any discrepancies.

Usage:
    uv run compare_module_vs_viz.py
"""

import os
import sys
import random

import numpy as np

# ── Import the module ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from motion_detection import (
    get_final_extraction,
    compute_motion_per_frame,
    find_motion_segments,
    find_longest_segment,
    _hand_present,
    DEFAULTS,
)

# ── Same config as viz script ──────────────────────────────────────
START_THRESH = DEFAULTS["start_thresh"]
END_THRESH   = DEFAULTS["end_thresh"]
COOLDOWN     = DEFAULTS["cooldown"]
MIN_FRAMES   = DEFAULTS["min_frames"]
METHOD       = DEFAULTS["method"]
TOP_K        = DEFAULTS["top_k"]
TIP_WEIGHT   = DEFAULTS["tip_weight"]
HEAD_PAD     = DEFAULTS["head_pad"]
VIS_THRESH   = DEFAULTS["vis_thresh"]

LANDMARKS_DIR = "/run/media/aryan/PoopiDrive/projects_linux/microsoft asl citizen/landmarks/"
N_SAMPLES     = 10000
SEED          = 42


# ── Reference implementation (copied verbatim from viz script) ─────
# This is the "ground truth" inline logic so we can diff precisely.

def ref_get_final_extraction(frames_raw):
    """Inline copy of the viz script logic — used as reference."""
    energies    = compute_motion_per_frame(frames_raw,
                                           method=METHOD, top_k=TOP_K,
                                           tip_weight=TIP_WEIGHT, vis_thresh=VIS_THRESH)
    segments    = find_motion_segments(energies, START_THRESH, END_THRESH, COOLDOWN, MIN_FRAMES)
    longest_seg = find_longest_segment(segments)

    if longest_seg is None:
        return None

    ls, le = longest_seg

    first_hand_frame = ls
    for fi in range(ls, le):
        if (_hand_present(frames_raw[fi], slice(33, 54), VIS_THRESH) or
                _hand_present(frames_raw[fi], slice(54, 75), VIS_THRESH)):
            first_hand_frame = fi
            break

    final_start = max(0, first_hand_frame - HEAD_PAD)
    return (final_start, le)


def load_npy(npy_path):
    arr          = np.load(npy_path)
    total_frames = int(arr[0, 0, 0])
    frames_raw   = arr[1:total_frames + 1]
    return frames_raw


def collect_npy_paths(landmarks_dir, n, seed):
    all_paths = []
    for word in os.listdir(landmarks_dir):
        word_dir = os.path.join(landmarks_dir, word)
        if not os.path.isdir(word_dir):
            continue
        for fname in os.listdir(word_dir):
            if fname.endswith(".npy"):
                all_paths.append(os.path.join(word_dir, fname))
    random.seed(seed)
    return random.sample(all_paths, min(n, len(all_paths)))


# ── Main ───────────────────────────────────────────────────────────

def main():
    print(f"Collecting {N_SAMPLES} random .npy files ...")
    paths = collect_npy_paths(LANDMARKS_DIR, N_SAMPLES, SEED)
    print(f"  Found {len(paths)} files to compare.\n")

    matches      = 0
    mismatches   = []
    load_errors  = 0

    for path in paths:
        try:
            frames_raw = load_npy(path)
        except Exception as e:
            print(f"  [LOAD ERROR] {path}: {e}")
            load_errors += 1
            continue

        ref    = ref_get_final_extraction(frames_raw)
        module = get_final_extraction(
            frames_raw,
            start_thresh = START_THRESH,
            end_thresh   = END_THRESH,
            cooldown     = COOLDOWN,
            min_frames   = MIN_FRAMES,
            method       = METHOD,
            top_k        = TOP_K,
            tip_weight   = TIP_WEIGHT,
            head_pad     = HEAD_PAD,
            vis_thresh   = VIS_THRESH,
        )

        if ref == module:
            matches += 1
        else:
            mismatches.append({
                "path":   path,
                "ref":    ref,
                "module": module,
                "frames": len(frames_raw),
            })

    total = matches + len(mismatches)
    print("=" * 60)
    print(f"Results: {total} videos compared  ({load_errors} load errors skipped)")
    print(f"  ✅ Match     : {matches}")
    print(f"  ❌ Mismatch  : {len(mismatches)}")
    print("=" * 60)

    if mismatches:
        print("\n── MISMATCHES ──────────────────────────────────────────────")
        for m in mismatches:
            name = os.path.basename(m["path"])
            print(f"\n  {name}")
            print(f"    Frames : {m['frames']}")
            print(f"    Ref    : {m['ref']}")
            print(f"    Module : {m['module']}")
    else:
        print("\nAll outputs match exactly. Module is validated. ✅")


if __name__ == "__main__":
    main()
