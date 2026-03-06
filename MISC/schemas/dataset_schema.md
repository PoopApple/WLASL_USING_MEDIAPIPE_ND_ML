# ASL Dataset Landmark Structure Documentation

## Overview
This dataset contains preprocessed video data for American Sign Language (ASL) recognition, converted into landmark coordinates using MediaPipe Pose and Hand detection.

## Dataset Shape

### Final Output Shape
```python
(70, 63, 4)
```
- **70 frames**: All videos normalized to 70 frames using linear interpolation
- **63 landmarks**: Total body landmarks (pose + hands)
- **4 features per landmark**: [x, y, z, visibility/confidence]

### Storage Details
- **Data type**: `float32` (optimized for TensorFlow)
- **File format**: `.npy` (NumPy binary)
- **Size per video**: ~69 KB (70 × 63 × 4 × 4 bytes)

---

## Landmark Breakdown (63 Total)

### 1. Pose Landmarks (0-20) - 21 landmarks
Selected from MediaPipe's 33 pose landmarks, focusing on upper body and face:

| Index Range | Body Part | Landmark IDs | Description |
|-------------|-----------|--------------|-------------|
| 0 | Face | 0 | Nose |
| 1 | Face | 2 | Left eye (inner) |
| 2 | Face | 5 | Right eye (inner) |
| 3 | Face | 7 | Left ear |
| 4 | Face | 8 | Right ear |
| 5 | Face | 9 | Mouth (left) |
| 6 | Face | 10 | Mouth (right) |
| 7 | Shoulders | 11 | Left shoulder |
| 8 | Shoulders | 12 | Right shoulder |
| 9 | Arms | 13 | Left elbow |
| 10 | Arms | 15 | Left wrist |
| 11 | Arms | 14 | Right elbow |
| 12 | Arms | 16 | Right wrist |
| 13 | Torso | 23 | Left hip |
| 14 | Torso | 24 | Right hip |
| 15-20 | Wrist detail | 17-22 | Additional wrist/hand connection points |

**Note**: Hands attached to pose were included because hand landmarks may not always be visible due to overlapping.

### 2. Left Hand Landmarks (21-41) - 21 landmarks
MediaPipe hand detection provides 21 landmarks per hand:

| Index Range | Finger | Landmarks |
|-------------|--------|-----------|
| 21 | Wrist | Hand base |
| 22-25 | Thumb | CMC, MCP, IP, Tip |
| 26-29 | Index | MCP, PIP, DIP, Tip |
| 30-33 | Middle | MCP, PIP, DIP, Tip |
| 34-37 | Ring | MCP, PIP, DIP, Tip |
| 38-41 | Pinky | MCP, PIP, DIP, Tip |

### 3. Right Hand Landmarks (42-62) - 21 landmarks
Same structure as left hand:

| Index Range | Finger | Landmarks |
|-------------|--------|-----------|
| 42 | Wrist | Hand base |
| 43-46 | Thumb | CMC, MCP, IP, Tip |
| 47-50 | Index | MCP, PIP, DIP, Tip |
| 51-54 | Middle | MCP, PIP, DIP, Tip |
| 55-58 | Ring | MCP, PIP, DIP, Tip |
| 59-62 | Pinky | MCP, PIP, DIP, Tip |

---

## Feature Dimensions (4 per landmark)

Each landmark has 4 values: `[x, y, z, visibility]`

### Coordinate Normalization

#### Pose Landmarks (indices 0-20):
- **x-coordinate**: Normalized by shoulder width
  ```python
  x_normalized = (x - center_of_shoulder_x) / shoulder_length
  ```
- **y-coordinate**: Normalized by torso height
  ```python
  y_normalized = (y - center_of_shoulder_y) / torso_height
  ```
- **z-coordinate**: Centered at shoulder
  ```python
  z_normalized = z - center_of_shoulder_z
  ```
- **visibility**: MediaPipe confidence (0-1)

#### Hand Landmarks (indices 21-62):
- **x-coordinate**: 
  ```python
  x_normalized = (x - center_of_shoulder_x) / shoulder_length
  ```
- **y-coordinate**: 
  ```python
  y_normalized = (y - center_of_shoulder_y) / torso_height
  ```
- **z-coordinate**: 
  ```python
  z_normalized = z - center_of_shoulder_z
  ```
- **visibility**: Always 1.0 (hands are either detected or zeros)

### Reference Points Calculation

**Center of Shoulder**:
```python
center_of_shoulder = [
    (landmark[11].x + landmark[12].x) / 2,  # x
    (landmark[11].y + landmark[12].y) / 2,  # y
    (landmark[11].z + landmark[12].z) / 2   # z
]
```

**Center of Hips**:
```python
center_of_hips = [
    (landmark[23].x + landmark[24].x) / 2,  # x
    (landmark[23].y + landmark[24].y) / 2,  # y
    (landmark[23].z + landmark[24].z) / 2   # z
]
```

**Shoulder Length** (width):
```python
shoulder_length = sqrt((landmark[11].x - landmark[12].x)² + 
                       (landmark[11].y - landmark[12].y)²)
```

**Torso Height**:
```python
torso_height = sqrt((center_shoulder[0] - center_hips[0])² + 
                    (center_shoulder[1] - center_hips[1])²)
```

---

## Missing Data Handling

### Zero Padding
When landmarks are not detected:
```python
missing_landmark = [0.0, 0.0, 0.0, 0.0]
```

### Cases:
1. **No pose detected**: All pose landmarks (0-20) = zeros
2. **No left hand detected**: Landmarks (21-41) = zeros
3. **No right hand detected**: Landmarks (42-62) = zeros

### Sticky Hands Feature
In [NEW_LM_VISION_TASKS.py](NEW_LM_VISION_TASKS.py), a "sticky hands" mechanism fills missing hand frames with the last known hand position to reduce zero-padding.

---

## Temporal Normalization

All videos are normalized to exactly **70 frames** using linear interpolation:

```python
linear_indices = np.linspace(0, total_frames - 1, 70, dtype=int)
normalized_frames = original_frames[linear_indices]
```

This ensures consistent input size regardless of original video length.

---

## Data Augmentation

### Video Flipping
Some videos are horizontally flipped to augment the dataset:
```python
image = cv2.flip(image, 1)
```
This creates mirror versions of signs for better model generalization.

---

## Quality Filtering

### Bad Torso Detection
Videos with abnormal torso measurements are flagged:
```python
if torso_height < 0.31:
    # Video logged to bad_torso.txt
```

These may indicate poor pose detection or unusual camera angles.

---

## File Organization

```
dataset_root/
├── word1/
│   ├── video1.npy  # shape: (70, 63, 4)
│   ├── video2.npy
│   └── ...
├── word2/
│   └── ...
└── ...
```

Each `.npy` file contains a single normalized video as a NumPy array.

---

## GTE9 Dataset Variant

### Overview
The `gte9_dataset` (Greater Than or Equal to 9) is a filtered subset containing only words with **9 or more video instances** per word. This provides a more balanced dataset for training with sufficient samples per class.

### Dataset Details
- **Source**: Filtered from main WLASL dataset
- **Filter criteria**: Words with ≥9 video examples
- **Total words**: 205 sign words
- **Processing**: Identical to main dataset (uses same landmark extraction pipeline)

### Shape & Structure
**Same as main dataset:**
```python
(70, 63, 4)
```
- Same 70-frame normalization
- Same 63 landmarks (21 pose + 21 left hand + 21 right hand)
- Same 4 features per landmark [x, y, z, visibility]

### File Organization
```
gte9_dataset/          # Raw videos (filtered subset)
└── word1/
    ├── video1.mp4
    ├── video2.mp4
    └── ...

gte9_landmarks/        # Processed landmarks (.npy files)
└── word1/
    ├── video1.npy    # shape: (70, 63, 4)
    ├── video2.npy
    └── ...
```

### Usage Recommendation
The GTE9 dataset is **recommended for initial model training** because:
1. Better class balance (each word has ≥9 examples)
2. Reduces class imbalance issues
3. More reliable validation splits possible
4. Faster iteration during development (smaller dataset)

### Word List
The complete list of 205 words in GTE9 dataset can be found in `stats/gte9_list.txt`.

---

## Usage Example

```python
import numpy as np

# Load a sample
data = np.load('path/to/video.npy')
print(data.shape)  # (70, 63, 4)

# Access specific landmarks
frame_0 = data[0]              # First frame
nose = data[0, 0]              # Nose landmark in first frame
left_hand = data[:, 21:42]     # All left hand landmarks across all frames
x_coords = data[:, :, 0]       # All x-coordinates
```

---

## Model Input Recommendations

### For LSTM/GRU:
```python
# Reshape to (samples, timesteps, features)
X = data.reshape(num_samples, 70, 63*4)  # (N, 70, 252)
```

### For 3D CNN:
```python
# Keep as (samples, timesteps, landmarks, features)
X = data  # (N, 70, 63, 4)
# Or add channel dimension
X = data[..., np.newaxis]  # (N, 70, 63, 4, 1)
```

### For Transformer:
```python
# Treat as sequence: (samples, timesteps, embedding_dim)
X = data.reshape(num_samples, 70, 252)
```

---

## Important Notes

1. **Coordinate System**: MediaPipe uses normalized coordinates (0-1 range), but this dataset applies additional normalization relative to body proportions
2. **Z-depth**: Less reliable than x/y coordinates
3. **Visibility**: Pose landmarks have actual visibility scores; hand landmarks use 1.0 or 0.0
4. **Missing Frames**: Handle zeros appropriately in your model (masking, interpolation, or ignore)
5. **Data Augmentation**: Consider the flipped versions as separate samples or handle them in your training pipeline

---

## References

- MediaPipe Pose: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
- MediaPipe Hands: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md
- Processing Scripts: 
  - [new_detect_modified_landmark_with_np_arr_only.py](new_detect_modified_landmark_with_np_arr_only.py)
  - [NEW_LM_VISION_TASKS.py](NEW_LM_VISION_TASKS.py)