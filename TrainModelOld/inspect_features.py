"""
Inspect landmark features to detect issues
"""
import numpy as np
import os

print("=" * 60)
print("FEATURE INSPECTION")
print("=" * 60)

# Load sample
x = np.load("../gte9_landmarks/x.npy")
y = np.load("../gte9_landmarks/y.npy")

print(f"\n📊 Dataset Shape: {x.shape}")
print(f"   - Samples: {x.shape[0]}")
print(f"   - Frames/sample: {x.shape[1]}")
print(f"   - Features/frame: {x.shape[2]}")

# Check for anomalies
print(f"\n🔍 Feature Statistics:")
print(f"   - Min value: {x.min():.6f}")
print(f"   - Max value: {x.max():.6f}")
print(f"   - Mean: {x.mean():.6f}")
print(f"   - Std Dev: {x.std():.6f}")
print(f"   - NaN count: {np.isnan(x).sum()}")
print(f"   - Inf count: {np.isinf(x).sum()}")

# Check zeros (visibility = 0)
zero_count = np.sum(x == 0.0)
print(f"   - Zero values: {zero_count} ({100*zero_count/x.size:.2f}%)")

# Per-feature analysis
print(f"\n🔍 Per-Feature Stats (first sample, first frame):")
sample = x[0, 0, :]
print(f"   Shape: {sample.shape} (63 landmarks × 4 dims)")
print(f"   Dims explanation: [x, y, z, visibility] repeated for 63 landmarks")
print(f"   Sample min: {sample.min():.3f}, max: {sample.max():.3f}")

# Count classes
unique_classes = len(np.unique(y))
print(f"\n📊 Number of classes: {unique_classes}")

# Data ranges before normalization
coords = x[:, :, :3]  # x, y, z only (ignore visibility)
print(f"\n🔍 Coordinate Ranges (x, y, z):")
print(f"   - Min: {coords.min():.3f}")
print(f"   - Max: {coords.max():.3f}")
print(f"   - This should typically be 0-1 (normalized) or similar")
print(f"   - If range is huge, normalization in landmark extraction is needed!")

print("\n" + "=" * 60)
