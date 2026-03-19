"""
Analysis for small sign language dataset
204 classes × 9-10 samples each = ~2000 total samples
"""
import numpy as np
import os
from collections import Counter

print("=" * 70)
print("SIGN LANGUAGE DATASET ANALYSIS (SMALL DATASET CONFIGURATION)")
print("=" * 70)

# Count actual data
dataset_path = "../gte9_landmarks"
class_counts = {}
total_samples = 0

for word in sorted(os.listdir(dataset_path)):
    word_folder = os.path.join(dataset_path, word)
    if os.path.isdir(word_folder):
        count = len([f for f in os.listdir(word_folder) if f.endswith('.npy')])
        class_counts[word] = count
        total_samples += count

print(f"\n📊 DATASET SIZE:")
print(f"   Total Classes: {len(class_counts)}")
print(f"   Total Samples: {total_samples}")
print(f"   Average Samples/Class: {total_samples / len(class_counts):.1f}")

counts = list(class_counts.values())
print(f"\n📊 SAMPLES PER CLASS DISTRIBUTION:")
print(f"   Min: {min(counts)}")
print(f"   Max: {max(counts)}")
print(f"   Median: {np.median(counts):.1f}")
print(f"   Std Dev: {np.std(counts):.1f}")

# Calculate train/val split
TRAIN_SPLIT = 0.15
n_val = int(total_samples * TRAIN_SPLIT)
n_train = total_samples - n_val

print(f"\n📊 TRAIN/VAL SPLIT (85/15):")
print(f"   Training samples: {n_train}")
print(f"   Validation samples: {n_val}")
print(f"   After 3x augmentation: {n_train * 3} training samples")

# Model capacity analysis
SEQUENCE_LENGTH = 70
FEATURE_DIM = 63 * 4
total_features = SEQUENCE_LENGTH * FEATURE_DIM

gru_64 = 64 * 2  # bidirectional
gru_32 = 32 * 2  # bidirectional
dense_64 = 64

total_params = (
    (FEATURE_DIM * gru_64 * 3) +  # GRU: x, h, W matrices
    (gru_64 * gru_64 * 3) +
    (gru_64 * gru_32 * 3) +
    (gru_32 * gru_32 * 3) +
    (gru_32 * dense_64) +
    (dense_64 * len(class_counts))
)

print(f"\n🧠 MODEL CAPACITY:")
print(f"   Total parameters: ~{total_params/1000:.0f}K")
print(f"   Ratio (samples/params): {n_train * 3 / total_params:.2f}x")
print(f"   Target: >2x (to prevent overfitting on small data)")

print(f"\n⚠️  KEY CHALLENGES:")
print(f"   ❌ Very small dataset (2000 samples)")
print(f"   ❌ High dimensionality (252 features)")
print(f"   ❌ Many classes (204) with few samples each")
print(f"   ❌ Risk of overfitting: train on ~1700, validate on ~300")

print(f"\n✅ MITIGATION STRATEGIES:")
print(f"   ✓ Small model (Bi-GRU 64→32 instead of 128→64)")
print(f"   ✓ Heavy dropout (0.2-0.4)")
print(f"   ✓ L2 regularization (weight_decay=5e-3)")
print(f"   ✓ Data augmentation (3x via noise injection)")
print(f"   ✓ Very low learning rate (1e-4)")
print(f"   ✓ High early stopping patience (30 epochs)")

print(f"\n🎯 EXPECTED PERFORMANCE:")
print(f"   - Realistic accuracy: 30-60% on validation")
print(f"   - If >70%: Check for train/val data leak or overfitting")
print(f"   - Better: Focus on improving landmark extraction quality")

print("\n" + "=" * 70)
print("RUN: python train_test1_using_ltsm.py")
print("=" * 70)
