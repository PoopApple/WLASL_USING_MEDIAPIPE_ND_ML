"""
Dataset Validation Script for GTE9 Landmarks
Checks data quality and identifies problematic samples

Usage: python3.12 validate_dataset.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json

# Configuration
LANDMARK_PATH = "../gte9_landmarks"
RESULTS_PATH = "../testing/dataset_validation"
os.makedirs(RESULTS_PATH, exist_ok=True)

SEQUENCE_LENGTH = 70
NUM_LANDMARKS = 63
NUM_FEATURES = 4

# Validation thresholds
MIN_NON_ZERO_FRAMES = 10  # Minimum frames with actual data
MIN_VISIBILITY = 0.1  # Minimum average visibility for pose landmarks
MAX_COORDINATE = 10.0  # Maximum reasonable coordinate value (normalized)
MIN_COORDINATE = -10.0  # Minimum reasonable coordinate value
MIN_MOVEMENT = 0.05  # Minimum movement variance (to detect frozen videos)
MAX_ZERO_RATIO = 0.4  # Maximum ratio of zero frames allowed

print("\n" + "="*80)
print("DATASET VALIDATION FOR GTE9 LANDMARKS")
print("="*80)

# ==================== VALIDATION FUNCTIONS ====================

def check_shape(data, word, filename):
    """Check if data has correct shape"""
    expected_shape = (SEQUENCE_LENGTH, NUM_LANDMARKS, NUM_FEATURES)
    if data.shape != expected_shape:
        return False, f"Wrong shape: {data.shape} (expected {expected_shape})"
    return True, "OK"


def check_for_nans_infs(data, word, filename):
    """Check for NaN or Inf values"""
    if np.isnan(data).any():
        nan_count = np.isnan(data).sum()
        return False, f"Contains {nan_count} NaN values"
    if np.isinf(data).any():
        inf_count = np.isinf(data).sum()
        return False, f"Contains {inf_count} Inf values"
    return True, "OK"


def check_coordinate_range(data, word, filename):
    """Check if coordinates are within reasonable range"""
    coords = data[:, :, :3]  # x, y, z coordinates
    
    max_val = np.max(coords)
    min_val = np.min(coords)
    
    if max_val > MAX_COORDINATE:
        return False, f"Max coordinate {max_val:.2f} exceeds threshold {MAX_COORDINATE}"
    if min_val < MIN_COORDINATE:
        return False, f"Min coordinate {min_val:.2f} below threshold {MIN_COORDINATE}"
    
    return True, "OK"


def check_zero_frames(data, word, filename):
    """Check for excessive zero frames (missing detections)"""
    # Count frames where all landmarks are zero
    frame_sums = np.sum(np.abs(data), axis=(1, 2))
    zero_frames = np.sum(frame_sums == 0)
    zero_ratio = zero_frames / SEQUENCE_LENGTH
    
    if zero_ratio > MAX_ZERO_RATIO:
        return False, f"Too many zero frames: {zero_frames}/{SEQUENCE_LENGTH} ({zero_ratio:.1%})"
    
    return True, f"Zero frames: {zero_frames}/{SEQUENCE_LENGTH} ({zero_ratio:.1%})"


def check_visibility_scores(data, word, filename):
    """Check visibility/confidence scores for pose landmarks"""
    # Visibility is the 4th feature (index 3) for pose landmarks (indices 0-20)
    pose_visibility = data[:, :21, 3]
    
    # Filter out zero frames
    non_zero_frames = pose_visibility[np.any(pose_visibility > 0, axis=1)]
    
    if len(non_zero_frames) == 0:
        return False, "No pose landmarks detected in any frame"
    
    avg_visibility = np.mean(non_zero_frames[non_zero_frames > 0])
    
    if avg_visibility < MIN_VISIBILITY:
        return False, f"Low average visibility: {avg_visibility:.3f} (min: {MIN_VISIBILITY})"
    
    return True, f"Avg visibility: {avg_visibility:.3f}"


def check_movement_variance(data, word, filename):
    """Check if video has actual movement (not frozen)"""
    # Calculate variance across time for x, y coordinates
    coords = data[:, :, :2]  # x, y only
    
    # Filter out zero frames
    non_zero_mask = np.any(coords != 0, axis=(1, 2))
    if np.sum(non_zero_mask) < 3:
        return False, "Not enough non-zero frames to check movement"
    
    non_zero_coords = coords[non_zero_mask]
    variance = np.var(non_zero_coords)
    
    if variance < MIN_MOVEMENT:
        return False, f"Very low movement variance: {variance:.6f} (likely frozen video)"
    
    return True, f"Movement variance: {variance:.4f}"


def check_hand_detection(data, word, filename):
    """Check if hands are detected in reasonable number of frames"""
    left_hand = data[:, 21:42, :]  # Left hand landmarks
    right_hand = data[:, 42:63, :]  # Right hand landmarks
    
    left_detected = np.sum(np.any(left_hand != 0, axis=(1, 2)))
    right_detected = np.sum(np.any(right_hand != 0, axis=(1, 2)))
    
    total_hand_detections = left_detected + right_detected
    
    if total_hand_detections < MIN_NON_ZERO_FRAMES:
        return False, f"Too few hand detections: L={left_detected}, R={right_detected}"
    
    return True, f"Hand detections: L={left_detected}, R={right_detected}"


def check_pose_detection(data, word, filename):
    """Check if pose is detected in reasonable number of frames"""
    pose = data[:, :21, :]  # Pose landmarks
    
    pose_detected = np.sum(np.any(pose != 0, axis=(1, 2)))
    
    if pose_detected < MIN_NON_ZERO_FRAMES:
        return False, f"Too few pose detections: {pose_detected}/{SEQUENCE_LENGTH}"
    
    return True, f"Pose detections: {pose_detected}/{SEQUENCE_LENGTH}"


def check_symmetry(data, word, filename):
    """Check for unusual asymmetry (might indicate bad cropping or detection)"""
    # Compare left vs right side landmarks
    left_shoulder = data[:, 7, :]  # Left shoulder
    right_shoulder = data[:, 8, :]  # Right shoulder
    
    # Filter non-zero frames
    valid_frames = np.logical_and(
        np.any(left_shoulder != 0, axis=1),
        np.any(right_shoulder != 0, axis=1)
    )
    
    if np.sum(valid_frames) < 5:
        return True, "Not enough frames to check symmetry"
    
    left_pos = left_shoulder[valid_frames, :2]
    right_pos = right_shoulder[valid_frames, :2]
    
    # Calculate average distance between shoulders
    shoulder_width = np.mean(np.abs(left_pos - right_pos))
    
    if shoulder_width < 0.05:
        return False, f"Unusually narrow shoulder width: {shoulder_width:.4f}"
    if shoulder_width > 2.0:
        return False, f"Unusually wide shoulder width: {shoulder_width:.4f}"
    
    return True, f"Shoulder width: {shoulder_width:.4f}"


# ==================== MAIN VALIDATION ====================

def validate_sample(data, word, filename):
    """Run all validation checks on a sample"""
    checks = [
        ("Shape", check_shape),
        ("NaN/Inf", check_for_nans_infs),
        ("Coordinate Range", check_coordinate_range),
        ("Zero Frames", check_zero_frames),
        ("Visibility", check_visibility_scores),
        ("Movement", check_movement_variance),
        ("Hand Detection", check_hand_detection),
        ("Pose Detection", check_pose_detection),
        ("Symmetry", check_symmetry),
    ]
    
    results = {}
    is_valid = True
    
    for check_name, check_func in checks:
        try:
            passed, message = check_func(data, word, filename)
            results[check_name] = {
                "passed": passed,
                "message": message
            }
            if not passed:
                is_valid = False
        except Exception as e:
            results[check_name] = {
                "passed": False,
                "message": f"Error: {str(e)}"
            }
            is_valid = False
    
    return is_valid, results


def validate_dataset():
    """Validate entire dataset"""
    print("\n📂 Loading dataset from:", LANDMARK_PATH)
    
    stats = {
        "total_samples": 0,
        "valid_samples": 0,
        "invalid_samples": 0,
        "words_processed": 0,
        "invalid_by_word": defaultdict(list),
        "invalid_by_reason": defaultdict(int),
        "validation_details": []
    }
    
    invalid_files = []
    valid_files = []
    
    # Process each word folder
    for word in sorted(os.listdir(LANDMARK_PATH)):
        word_folder = os.path.join(LANDMARK_PATH, word)
        
        if not os.path.isdir(word_folder):
            continue
        
        stats["words_processed"] += 1
        print(f"\n📁 Validating word: {word}")
        
        word_valid_count = 0
        word_invalid_count = 0
        
        for filename in sorted(os.listdir(word_folder)):
            if not filename.endswith('.npy'):
                continue
            
            filepath = os.path.join(word_folder, filename)
            stats["total_samples"] += 1
            
            try:
                # Load data
                data = np.load(filepath)
                
                # Validate
                is_valid, results = validate_sample(data, word, filename)
                
                if is_valid:
                    stats["valid_samples"] += 1
                    word_valid_count += 1
                    valid_files.append(filepath)
                else:
                    stats["invalid_samples"] += 1
                    word_invalid_count += 1
                    invalid_files.append(filepath)
                    stats["invalid_by_word"][word].append(filename)
                    
                    # Count reasons for invalidity
                    for check_name, result in results.items():
                        if not result["passed"]:
                            stats["invalid_by_reason"][check_name] += 1
                    
                    # Store details
                    stats["validation_details"].append({
                        "word": word,
                        "filename": filename,
                        "filepath": filepath,
                        "results": results
                    })
                    
                    print(f"  ❌ {filename}")
                    for check_name, result in results.items():
                        if not result["passed"]:
                            print(f"     └─ {check_name}: {result['message']}")
            
            except Exception as e:
                print(f"  ⚠️ Error loading {filename}: {e}")
                stats["invalid_samples"] += 1
                word_invalid_count += 1
                invalid_files.append(filepath)
                stats["invalid_by_word"][word].append(filename)
                stats["invalid_by_reason"]["Load Error"] += 1
        
        print(f"  ✅ Valid: {word_valid_count} | ❌ Invalid: {word_invalid_count}")
    
    return stats, invalid_files, valid_files


def generate_report(stats, invalid_files, valid_files):
    """Generate validation report"""
    print("\n" + "="*80)
    print("VALIDATION REPORT")
    print("="*80)
    
    print(f"\n📊 Summary:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Valid samples: {stats['valid_samples']} ({stats['valid_samples']/stats['total_samples']*100:.1f}%)")
    print(f"  Invalid samples: {stats['invalid_samples']} ({stats['invalid_samples']/stats['total_samples']*100:.1f}%)")
    print(f"  Words processed: {stats['words_processed']}")
    
    if stats['invalid_samples'] > 0:
        print(f"\n❌ Invalid samples by reason:")
        for reason, count in sorted(stats['invalid_by_reason'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count} samples")
        
        print(f"\n❌ Invalid samples by word (top 10):")
        invalid_by_word_sorted = sorted(stats['invalid_by_word'].items(), 
                                       key=lambda x: len(x[1]), reverse=True)
        for word, files in invalid_by_word_sorted[:10]:
            print(f"  {word}: {len(files)} invalid samples")
    
    # Save detailed report
    report_file = f"{RESULTS_PATH}/validation_report.txt"
    with open(report_file, 'w') as f:
        f.write("DATASET VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total samples: {stats['total_samples']}\n")
        f.write(f"Valid samples: {stats['valid_samples']} ({stats['valid_samples']/stats['total_samples']*100:.1f}%)\n")
        f.write(f"Invalid samples: {stats['invalid_samples']} ({stats['invalid_samples']/stats['total_samples']*100:.1f}%)\n\n")
        
        if stats['invalid_samples'] > 0:
            f.write("\nINVALID SAMPLES BY REASON:\n")
            f.write("-"*80 + "\n")
            for reason, count in sorted(stats['invalid_by_reason'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"{reason}: {count} samples\n")
            
            f.write("\n\nINVALID SAMPLES BY WORD:\n")
            f.write("-"*80 + "\n")
            for word, files in sorted(stats['invalid_by_word'].items()):
                f.write(f"\n{word} ({len(files)} invalid):\n")
                for filename in files:
                    f.write(f"  - {filename}\n")
            
            f.write("\n\nDETAILED VALIDATION RESULTS:\n")
            f.write("-"*80 + "\n")
            for detail in stats['validation_details']:
                f.write(f"\n{detail['word']}/{detail['filename']}:\n")
                for check_name, result in detail['results'].items():
                    status = "✓" if result['passed'] else "✗"
                    f.write(f"  {status} {check_name}: {result['message']}\n")
    
    print(f"\n✅ Detailed report saved to: {report_file}")
    
    # Save invalid files list
    invalid_list_file = f"{RESULTS_PATH}/invalid_files.txt"
    with open(invalid_list_file, 'w') as f:
        for filepath in invalid_files:
            f.write(filepath + "\n")
    print(f"✅ Invalid files list saved to: {invalid_list_file}")
    
    # Save valid files list
    valid_list_file = f"{RESULTS_PATH}/valid_files.txt"
    with open(valid_list_file, 'w') as f:
        for filepath in valid_files:
            f.write(filepath + "\n")
    print(f"✅ Valid files list saved to: {valid_list_file}")
    
    # Save JSON report
    json_report = {
        "total_samples": stats['total_samples'],
        "valid_samples": stats['valid_samples'],
        "invalid_samples": stats['invalid_samples'],
        "validity_percentage": stats['valid_samples']/stats['total_samples']*100,
        "invalid_by_reason": dict(stats['invalid_by_reason']),
        "invalid_by_word": {k: v for k, v in stats['invalid_by_word'].items()},
        "invalid_files": invalid_files,
        "valid_files": valid_files
    }
    
    json_file = f"{RESULTS_PATH}/validation_report.json"
    with open(json_file, 'w') as f:
        json.dump(json_report, f, indent=2)
    print(f"✅ JSON report saved to: {json_file}")


def create_filtered_dataset_info():
    """Create instructions for using filtered dataset"""
    print("\n" + "="*80)
    print("CREATING FILTERED DATASET")
    print("="*80)
    
    instructions = """
To use only VALID samples in training:

1. Modify your data loading function to filter based on valid_files.txt

2. Add this code to test_recommended_models.py:

```python
def load_data_filtered():
    # Load valid files list
    with open('../testing/dataset_validation/valid_files.txt', 'r') as f:
        valid_files = [line.strip() for line in f.readlines()]
    
    X, y = [], []
    
    for filepath in valid_files:
        try:
            data = np.load(filepath)
            # Extract word from path
            word = filepath.split('/')[-2]
            X.append(data)
            y.append(word)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
    
    return np.array(X), np.array(y)
```

3. Or simply delete/move invalid files to a backup folder

"""
    
    print(instructions)
    
    instructions_file = f"{RESULTS_PATH}/usage_instructions.txt"
    with open(instructions_file, 'w') as f:
        f.write(instructions)
    print(f"✅ Usage instructions saved to: {instructions_file}")


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("\n🔍 Starting dataset validation...")
    print(f"Checking {LANDMARK_PATH}...")
    
    stats, invalid_files, valid_files = validate_dataset()
    generate_report(stats, invalid_files, valid_files)
    create_filtered_dataset_info()
    
    print("\n" + "="*80)
    print("✅ VALIDATION COMPLETE!")
    print("="*80)
    
    if stats['invalid_samples'] > 0:
        print(f"\n⚠️  Found {stats['invalid_samples']} invalid samples out of {stats['total_samples']}")
        print(f"📊 Validity rate: {stats['valid_samples']/stats['total_samples']*100:.1f}%")
        print(f"\n💡 Recommendation:")
        if stats['invalid_samples'] / stats['total_samples'] > 0.1:
            print("   - Consider removing invalid samples before training")
            print("   - Check validation report for common issues")
            print("   - May need to regenerate landmarks for some videos")
        else:
            print("   - Small number of invalid samples, likely safe to proceed")
            print("   - Can optionally filter them out for cleaner training")
    else:
        print("\n✅ All samples passed validation!")
    
    print(f"\n📁 Results saved to: {RESULTS_PATH}/")
