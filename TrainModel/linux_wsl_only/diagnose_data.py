"""
Diagnose dataset quality and class imbalance
"""
import os
import numpy as np
from collections import Counter

dataset_path = "../gte9_landmarks"

class_counts = {}
total_samples = 0
nan_found = False

print("=" * 60)
print("DATASET DIAGNOSTICS")
print("=" * 60)

for word in sorted(os.listdir(dataset_path)):
    word_folder = os.path.join(dataset_path, word)
    
    if os.path.isdir(word_folder):
        count = len([f for f in os.listdir(word_folder) if f.endswith('.npy')])
        class_counts[word] = count
        total_samples += count
        
        # Check first file for NaNs
        files = [f for f in os.listdir(word_folder) if f.endswith('.npy')]
        if files:
            sample = np.load(os.path.join(word_folder, files[0]))
            if np.any(np.isnan(sample)):
                print(f"⚠️  NaNs found in {word}/{files[0]}")
                nan_found = True

print(f"\n📊 Total classes: {len(class_counts)}")
print(f"📊 Total samples: {total_samples}")
print(f"📊 Average samples/class: {total_samples / len(class_counts):.1f}")

# Count imbalance
counts = list(class_counts.values())
print(f"📊 Min samples/class: {min(counts)}")
print(f"📊 Max samples/class: {max(counts)}")
print(f"📊 Std Dev: {np.std(counts):.1f}")

# Find underrepresented classes
print(f"\n⚠️  Classes with <15 samples:")
underrep = [(k, v) for k, v in class_counts.items() if v < 15]
if underrep:
    for word, count in sorted(underrep, key=lambda x: x[1])[:20]:
        print(f"   {word}: {count}")
else:
    print("   None (all classes have ≥15 samples)")

if not nan_found:
    print("\n✅ No NaN values found in sample checks")

# Distribution analysis
median_count = np.median(counts)
print(f"\n📊 Median samples/class: {median_count:.1f}")
print(f"📊 Classes above median: {sum(1 for c in counts if c >= median_count)}")
print(f"📊 Classes below median: {sum(1 for c in counts if c < median_count)}")

# Recommendation
if max(counts) / min(counts) > 2:
    print("\n🚨 HIGH CLASS IMBALANCE DETECTED!")
    print("   Recommendation: Use class_weight='balanced' or resample data")
