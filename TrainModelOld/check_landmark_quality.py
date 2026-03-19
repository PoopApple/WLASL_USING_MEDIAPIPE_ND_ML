"""
Check landmark extraction quality
If landmarks are poorly extracted, no model will work well
"""
import numpy as np
import os

print("=" * 70)
print("LANDMARK QUALITY CHECK")
print("=" * 70)

dataset_path = "../gte9_landmarks"

# Load first sample
for word in sorted(os.listdir(dataset_path)):
    word_folder = os.path.join(dataset_path, word)
    if os.path.isdir(word_folder):
        files = [f for f in os.listdir(word_folder) if f.endswith('.npy')]
        if files:
            sample = np.load(os.path.join(word_folder, files[0]))
            break

print(f"\n📊 SAMPLE LANDMARK ARRAY:")
print(f"   Shape: {sample.shape}")
print(f"   - Frames: {sample.shape[0]}")
print(f"   - Landmarks: {sample.shape[1]}")
print(f"   - Dimensions per landmark: {sample.shape[2]} (x, y, z, visibility)")

print(f"\n📊 VALUE RANGES:")
print(f"   Min: {sample.min():.6f}")
print(f"   Max: {sample.max():.6f}")
print(f"   Mean: {sample.mean():.6f}")
print(f"   Std: {sample.std():.6f}")

# Check visibility (should be 0-1, last column)
visibility = sample[:, :, 3]
print(f"\n👁️ VISIBILITY ANALYSIS (confidence scores):")
print(f"   Min: {visibility.min():.3f}")
print(f"   Max: {visibility.max():.3f}")
print(f"   Mean: {visibility.mean():.3f}")
print(f"   Zero values: {np.sum(visibility == 0.0)} / {visibility.size}")

# Check coordinates
coords = sample[:, :, :3]  # x, y, z
print(f"\n📍 COORDINATE ANALYSIS (x, y, z):")
print(f"   Should be normalized (0-1) for MediaPipe")
print(f"   Min: {coords.min():.3f}")
print(f"   Max: {coords.max():.3f}")

if coords.max() > 2.0 or coords.min() < -2.0:
    print(f"   ⚠️  WARNING: Coordinates seem unnormalized!")
    print(f"      Raw pixel coords detected instead of normalized 0-1 range")
    print(f"      This WILL hurt model performance!")
elif coords.min() >= 0 and coords.max() <= 1.5:
    print(f"   ✅ Coordinates appear normalized (good)")

# Check NaNs
print(f"\n🔍 DATA INTEGRITY:")
print(f"   NaN count: {np.isnan(sample).sum()}")
print(f"   Inf count: {np.isinf(sample).sum()}")
if np.isnan(sample).sum() == 0 and np.isinf(sample).sum() == 0:
    print(f"   ✅ No NaN or Inf values")

# Statistics
print(f"\n📈 FRAME STATISTICS:")
for frame_idx in [0, sample.shape[0]//2, sample.shape[0]-1]:
    frame = sample[frame_idx]
    print(f"   Frame {frame_idx}: mean={frame.mean():.3f}, std={frame.std():.3f}, visibility_mean={frame[:, 3].mean():.3f}")

print("\n" + "=" * 70)
print("RECOMMENDATION:")
print("=" * 70)

if coords.max() > 2.0:
    print("🚨 LANDMARK QUALITY ISSUE DETECTED!")
    print("   Fix: Check new_detect_modified_landmark_with_np_arr_only.py")
    print("   Landmarks should be normalized to 0-1 range")
elif visibility.mean() < 0.5:
    print("⚠️  LOW VISIBILITY SCORES!")
    print("   Consider lowering detection confidence thresholds in MediaPipe")
elif np.std(coords) < 0.05:
    print("⚠️  LOW VARIANCE IN COORDINATES!")
    print("   Landmarks might be collapsing to center or body part")
else:
    print("✅ Landmark quality appears reasonable")
    print("   Proceed with training")
