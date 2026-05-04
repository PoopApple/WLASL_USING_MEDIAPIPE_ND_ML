"""
trimmed_stats.py
================
Computes statistics on the trimmed signing-segment lengths after applying
the motion detection module. This helps decide the optimal temporal
normalisation target (64 vs 128 frames).

Usage:
    uv run trimmed_stats.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless – saves plot to file instead of showing
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Import motion detection module ─────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "MotionTesting"))
from motion_detection import get_final_extraction

# ── CONFIG ─────────────────────────────────────────────────────────
LANDMARKS_DIR = "/run/media/aryan/PoopiDrive/projects_linux/microsoft asl citizen/landmarks/"
N_WORKERS     = 8     # parallel threads for fast scanning
PLOT_PATH     = "trimmed_segment_stats.png"
# ───────────────────────────────────────────────────────────────────


def process_file(npy_path):
    """Return (raw_frames, trimmed_frames) or None on error."""
    try:
        arr          = np.load(npy_path, mmap_mode="r")
        total_frames = int(arr[0, 0, 0])
        frames_raw   = arr[1:total_frames + 1]
        raw_len      = total_frames

        seg = get_final_extraction(frames_raw)
        if seg is None:
            return (raw_len, None)   # no segment found
        trimmed_len = seg[1] - seg[0]
        return (raw_len, trimmed_len)
    except Exception:
        return None


def collect_npy_paths(landmarks_dir):
    paths = []
    for word in os.listdir(landmarks_dir):
        word_dir = os.path.join(landmarks_dir, word)
        if not os.path.isdir(word_dir):
            continue
        for fname in os.listdir(word_dir):
            if fname.endswith(".npy"):
                paths.append(os.path.join(word_dir, fname))
    return paths


def print_stats(label, values):
    values = np.array(values)
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 99]
    print(f"\n── {label} ({'N=' + str(len(values))}) ──────────────────────")
    print(f"  Min     : {int(np.min(values))}")
    print(f"  Max     : {int(np.max(values))}")
    print(f"  Mean    : {np.mean(values):.1f}")
    print(f"  Median  : {np.median(values):.1f}")
    print(f"  Std Dev : {np.std(values):.1f}")
    print("  Percentiles:")
    for p, v in zip(percentiles, np.percentile(values, percentiles)):
        print(f"    {p:3d}% = {v:.1f} frames")


def plot_histogram(raw_lengths, trimmed_lengths, no_seg_count, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ASL Dataset — Frame Length Distribution After Motion Trimming", fontsize=13)

    axes[0].hist(raw_lengths,     bins=60, color="#5588cc", edgecolor="white", linewidth=0.4)
    axes[0].set_title("Raw .npy Frame Counts")
    axes[0].set_xlabel("Frames")
    axes[0].set_ylabel("Videos")
    for target in [64, 96, 128]:
        axes[0].axvline(target, color="red", linewidth=1.2, linestyle="--", label=str(target))
    axes[0].legend()

    axes[1].hist(trimmed_lengths, bins=60, color="#55bb77", edgecolor="white", linewidth=0.4)
    axes[1].set_title(f"Trimmed Segment Lengths  ({no_seg_count} videos had no segment)")
    axes[1].set_xlabel("Frames")
    axes[1].set_ylabel("Videos")
    for target in [32, 48, 64, 96]:
        axes[1].axvline(target, color="red", linewidth=1.2, linestyle="--", label=str(target))
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"\nHistogram saved → {path}")


def main():
    print(f"Scanning {LANDMARKS_DIR} ...")
    paths = collect_npy_paths(LANDMARKS_DIR)
    print(f"  Found {len(paths)} .npy files. Processing with {N_WORKERS} threads ...\n")

    raw_lengths     = []
    trimmed_lengths = []
    no_seg_count    = 0
    errors          = 0
    done            = 0

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(process_file, p): p for p in paths}
        for fut in as_completed(futures):
            done += 1
            if done % 5000 == 0:
                print(f"  ... {done}/{len(paths)} processed")
            result = fut.result()
            if result is None:
                errors += 1
                continue
            raw_len, trimmed_len = result
            raw_lengths.append(raw_len)
            if trimmed_len is None:
                no_seg_count += 1
            else:
                trimmed_lengths.append(trimmed_len)

    print(f"\n{'='*60}")
    print(f"Total files   : {len(paths)}")
    print(f"Load errors   : {errors}")
    print(f"No segment    : {no_seg_count}  ({100*no_seg_count/len(paths):.1f}% of dataset)")
    print(f"Valid segments: {len(trimmed_lengths)}")

    print_stats("Raw .npy frame counts",    raw_lengths)
    print_stats("Trimmed segment lengths",  trimmed_lengths)

    # Decide recommendation
    t = np.array(trimmed_lengths)
    for target in [32, 48, 64, 96, 128]:
        coverage = 100.0 * np.mean(t <= target)
        print(f"\n  {target:3d}-frame target covers {coverage:.1f}% of trimmed segments"
              f"  ({np.sum(t > target)} would be downsampled)")

    plot_histogram(raw_lengths, trimmed_lengths, no_seg_count, PLOT_PATH)


if __name__ == "__main__":
    main()
    """
    ============================================================
    Total files   : 83399
    Load errors   : 0
    No segment    : 1975  (2.4% of dataset)
    Valid segments: 81424

    ── Raw .npy frame counts (N=83399) ──────────────────────
    Min     : 3
    Max     : 680
    Mean    : 82.8
    Median  : 75.0
    Std Dev : 37.7
    Percentiles:
        10% = 47.0 frames
        20% = 55.0 frames
        30% = 62.0 frames
        40% = 68.0 frames
        50% = 75.0 frames
        60% = 82.0 frames
        70% = 90.0 frames
        80% = 102.0 frames
        85% = 113.0 frames
        90% = 130.0 frames
        95% = 158.0 frames
        99% = 211.0 frames

    ── Trimmed segment lengths (N=81424) ──────────────────────
    Min     : 11
    Max     : 241
    Mean    : 38.8
    Median  : 37.0
    Std Dev : 14.0
    Percentiles:
        10% = 23.0 frames
        20% = 27.0 frames
        30% = 31.0 frames
        40% = 34.0 frames
        50% = 37.0 frames
        60% = 40.0 frames
        70% = 44.0 frames
        80% = 49.0 frames
        85% = 52.0 frames
        90% = 56.0 frames
        95% = 64.0 frames
        99% = 83.0 frames

    32-frame target covers 35.3% of trimmed segments  (52643 would be downsampled)

    48-frame target covers 79.9% of trimmed segments  (16387 would be downsampled)

    64-frame target covers 95.4% of trimmed segments  (3738 would be downsampled)

    96-frame target covers 99.6% of trimmed segments  (330 would be downsampled)

    128-frame target covers 99.9% of trimmed segments  (52 would be downsampled)

    
    
    """