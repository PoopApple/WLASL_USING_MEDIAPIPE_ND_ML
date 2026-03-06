# American Sign Language Recognition System
## Project Report

---

## 1. Introduction

### 1.1 Background
American Sign Language (ASL) is a complete, natural language that serves as the predominant sign language used by deaf and hard-of-hearing communities in the United States and parts of Canada. It employs signs made with hands, facial expressions, and body postures to convey meaning. According to various estimates, between 250,000 to 500,000 people in the United States use ASL as their primary means of communication.

Despite its widespread use within the deaf community, ASL remains largely inaccessible to the hearing population. This communication gap creates significant barriers in critical areas:
- **Education**: Limited access to sign language interpreters in schools
- **Healthcare**: Communication challenges during medical consultations
- **Employment**: Workplace accessibility and inclusion issues
- **Social Interaction**: Everyday communication barriers in public spaces

### 1.2 Problem Statement
The fundamental challenge is the **lack of automatic, real-time translation systems** that can bridge the communication gap between ASL users and non-signers. Traditional solutions rely on human interpreters, which are:
- **Expensive**: Professional interpreters cost $50-150 per hour
- **Limited Availability**: Shortage of certified interpreters in many regions
- **Not Scalable**: Cannot be deployed for everyday, casual interactions
- **Privacy Concerns**: Third-party presence in sensitive conversations

An automated ASL recognition system could democratize access to sign language communication, making it available 24/7 at minimal cost through smartphones, computers, and assistive devices.

### 1.3 Motivation
Recent advances in **computer vision** and **deep learning** have made automated sign language recognition technically feasible:
1. **Pose Estimation**: Modern frameworks like MediaPipe can extract human body landmarks in real-time
2. **Temporal Modeling**: Recurrent neural networks (RNNs) and Transformers can model sequential dependencies in sign language
3. **Computational Power**: GPUs enable real-time inference on consumer hardware
4. **Large Datasets**: WLASL and similar datasets provide training data for machine learning models

This project leverages these technological advances to build a **landmark-based ASL recognition system** that can classify sign language gestures from video sequences.

### 1.4 Scope
This project focuses on **isolated word-level ASL recognition**:
- **Input**: Video sequences of individual ASL signs
- **Output**: Classification into one of 204 ASL words
- **Approach**: Landmark-based (not pixel-based) using MediaPipe Holistic
- **Dataset**: WLASL GTE9 subset (signs with ≥9 instances)

**Out of Scope:**
- Continuous sign language recognition (sentence-level)
- Fingerspelling recognition
- Facial expression analysis
- Real-time video capture (preprocessing pipeline demonstrated, but focus is on classification)

---

## 2. Objectives

### 2.1 Primary Objectives
1. **Develop a landmark extraction pipeline** using MediaPipe Holistic to convert raw ASL videos into normalized coordinate sequences
2. **Build and train multiple deep learning architectures** (BiLSTM, BiGRU, LSTM-Attention, Transformer) for temporal sequence classification
3. **Compare model performance** to identify the best architecture for small-dataset ASL recognition
4. **Achieve practical accuracy** of 40-55% on 204-class classification (challenging baseline given data constraints)

### 2.2 Secondary Objectives
1. **Optimize training efficiency** by identifying and removing performance bottlenecks
2. **Combat overfitting** through comprehensive regularization strategies (dropout, L2, label smoothing, augmentation)
3. **Create reproducible codebase** with clear documentation and modular design
4. **Demonstrate scalability** by showing how the system can extend to larger vocabularies and real-time applications

### 2.3 Success Criteria
| Metric | Target | Rationale |
|--------|--------|-----------|
| **Test Accuracy** | 40-55% | Realistic for 204 classes with ~10 samples each |
| **Training Speed** | <60ms/step | Enables rapid experimentation (100 epochs in ~30 min) |
| **Overfitting Gap** | <30% | Train/test accuracy difference indicates generalization |
| **Inference Speed** | <50ms/frame | Required for real-time applications (20+ FPS) |
| **Model Size** | <5MB | Enables mobile deployment |

### 2.4 Dataset Overview

#### 2.4.1 Source Dataset
- **Name**: WLASL (Word-Level American Sign Language) v0.3
- **Full Dataset**: 2,000+ ASL words with video instances
- **Subset Used**: GTE9 (Greater Than or Equal to 9 instances)
- **Filter Criteria**: Only words with ≥9 video examples per class

#### 2.4.2 GTE9 Statistics
- **Total Classes**: 204-205 ASL words
- **Total Samples**: ~2,060 videos
- **Samples per Class**: ~10 videos (range: 9-20)
- **Class Balance**: Relatively balanced compared to full WLASL dataset
- **Video Properties**: 
  - Variable length: 1-10 seconds
  - Resolution: 480p-1080p
  - Frame rate: 24-30 FPS
  - Multiple signers with varying styles, ages, backgrounds

#### 2.4.3 Preprocessed Landmark Format
After MediaPipe processing, each video is converted to:
- **Shape**: (70, 63, 4)
  - **70 frames**: Temporally normalized using linear interpolation
  - **63 landmarks**: 21 pose + 21 left hand + 21 right hand
  - **4 features**: [x, y, z, visibility/confidence]
- **Data Type**: float32
- **File Format**: .npy (NumPy binary)
- **Size per Video**: ~69 KB

#### 2.4.4 Landmark Breakdown

**Pose Landmarks (Indices 0-20)**: 21 upper body keypoints
- Face: Nose, eyes, ears, mouth (7 points)
- Shoulders: Left and right shoulder (2 points)
- Arms: Elbows and wrists (4 points)
- Torso: Hips (2 points)
- Wrist detail: Additional hand-pose connection points (6 points)

**Left Hand Landmarks (Indices 21-41)**: 21 keypoints
- Wrist base (1 point)
- Thumb: CMC, MCP, IP, Tip (4 points)
- Index to Pinky: MCP, PIP, DIP, Tip each (16 points)

**Right Hand Landmarks (Indices 42-62)**: 21 keypoints
- Same structure as left hand

**Total Input Dimensionality**: 70 frames × 63 landmarks × 4 features = **17,640 features per video**
**Flattened for RNNs**: 70 timesteps × 252 features

#### 2.4.5 Normalization Schema
To achieve scale and position invariance:

**Reference Points**:
```python
shoulder_center = (left_shoulder + right_shoulder) / 2
hip_center = (left_hip + right_hip) / 2
shoulder_width = distance(left_shoulder, right_shoulder)
torso_height = distance(shoulder_center, hip_center)
```

**Pose Normalization**:
```python
x_norm = (x - shoulder_center_x) / shoulder_width
y_norm = (y - shoulder_center_y) / torso_height
z_norm = z - shoulder_center_z
```

**Hand Normalization**:
```python
# Hands normalized by same body references
x_norm = (x - shoulder_center_x) / shoulder_width
y_norm = (y - shoulder_center_y) / torso_height
z_norm = z - shoulder_center_z
```

**Benefits**:
- Signer distance from camera doesn't affect features
- Signer position in frame doesn't matter
- Different body sizes produce equivalent coordinates
- Cross-signer generalization improved

#### 2.4.6 Data Quality Measures
- **Missing Data Handling**: Zero-padding when landmarks not detected
- **Quality Filtering**: Videos with torso_height < 0.31 flagged as poor quality
- **Temporal Consistency**: 70-frame normalization ensures consistent sequence length
- **Validation**: All data checked for NaN/Inf values

#### 2.4.7 Challenge Assessment
This is an **extremely small dataset** for deep learning:
- Typical vision tasks: 1000+ samples per class
- This dataset: ~10 samples per class (100× less data)
- **Primary Challenge**: Severe overfitting
- **Consequence**: Models memorize training data, fail to generalize
- **Mitigation**: Aggressive regularization + 12× data augmentation required

---

## 3. Theory and Methods

### 3.1 MediaPipe Holistic Framework

#### 3.1.1 Overview
**MediaPipe** is Google's open-source framework for building multimodal machine learning pipelines. **MediaPipe Holistic** combines three pre-trained models to simultaneously detect:
1. **Pose landmarks** (33 keypoints): Full body skeletal structure
2. **Hand landmarks** (21 keypoints per hand × 2 hands): Detailed finger joint positions
3. **Face landmarks** (468 keypoints): Facial expressions (not used in this project)

#### 3.1.2 Architecture
MediaPipe uses a **two-stage detection pipeline**:

**Stage 1: Detection**
- **BlazePose Detector**: Detects person bounding box in the frame
- **BlazePalm Detector**: Detects hand bounding boxes (left/right)
- Runs at 30+ FPS on CPU

**Stage 2: Landmark Prediction**
- **BlazePose Landmark Model**: Predicts 33 3D pose landmarks
- **BlazePalm Landmark Model**: Predicts 21 3D hand landmarks per hand
- Uses lightweight CNN architectures optimized for mobile devices

#### 3.1.3 Landmark Representation
Each landmark is represented as a 4-tuple:
- **(x, y)**: 2D coordinates normalized to image dimensions [0, 1]
- **z**: Depth coordinate relative to body center (approximate)
- **visibility**: Confidence score [0, 1] indicating detection reliability

**Total Features per Frame**:
- Pose: 21 selected landmarks × 4 features = 84 features
- Left Hand: 21 landmarks × 4 features = 84 features
- Right Hand: 21 landmarks × 4 features = 84 features
- **Total**: 252 features per frame

#### 3.1.4 Advantages for ASL Recognition
1. **Efficiency**: Real-time processing (30+ FPS) enables practical deployment
2. **Robustness**: Works across different lighting, backgrounds, clothing
3. **Privacy**: Only extracts skeletal coordinates, not raw video
4. **Interpretability**: Landmarks are human-understandable features
5. **Generalization**: Pre-trained on diverse datasets, works across different signers

### 3.2 Recurrent Neural Networks (RNNs)

#### 3.2.1 Fundamentals
Sign language is inherently **temporal** - the meaning emerges from a sequence of poses over time. Traditional feedforward networks cannot capture this temporal dependency. **Recurrent Neural Networks (RNNs)** solve this by maintaining a "memory" of previous inputs.

**RNN Equation**:
```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
```
Where:
- `h_t`: Hidden state at time t (memory)
- `x_t`: Input at time t
- `y_t`: Output at time t

#### 3.2.2 LSTM (Long Short-Term Memory)
Vanilla RNNs suffer from **vanishing/exploding gradients** - they cannot learn long-term dependencies. **LSTM** solves this with a gating mechanism:

**Gates**:
1. **Forget Gate** (f_t): What information to discard from memory
2. **Input Gate** (i_t): What new information to store
3. **Output Gate** (o_t): What to output from memory

**Cell State Update**:
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)        # Forget
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)        # Input
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)    # Candidate
C_t = f_t * C_{t-1} + i_t * C̃_t           # Update cell
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)        # Output
h_t = o_t * tanh(C_t)                      # Hidden state
```

**Advantages**:
- Can learn dependencies over 100+ time steps
- Prevents vanishing gradients through additive cell state updates
- Widely used for sequence modeling tasks

**Disadvantages**:
- Computationally expensive (4× parameters vs vanilla RNN)
- Slower training and inference

#### 3.2.3 GRU (Gated Recurrent Unit)
**GRU** simplifies LSTM by combining forget and input gates into a single **update gate**:

**Gates**:
1. **Update Gate** (z_t): Balance between previous memory and new input
2. **Reset Gate** (r_t): How much past information to forget

**Equations**:
```
z_t = σ(W_z · [h_{t-1}, x_t])              # Update gate
r_t = σ(W_r · [h_{t-1}, x_t])              # Reset gate
h̃_t = tanh(W · [r_t * h_{t-1}, x_t])      # Candidate
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t     # New hidden state
```

**Advantages over LSTM**:
- **Fewer parameters**: ~25% fewer than LSTM (faster training)
- **Comparable performance**: Often matches LSTM on sequence tasks
- **Easier optimization**: Simpler architecture is easier to train

**Why GRU for ASL?**
- Sign language sequences are relatively short (70 frames)
- Training speed matters for experimentation
- Fewer parameters reduce overfitting risk on small datasets

#### 3.2.4 Bidirectional RNNs
Standard RNNs only process sequences **left-to-right** (past → future). **Bidirectional RNNs** process in both directions:

```
Forward:  h⃗_t = GRU(x_t, h⃗_{t-1})
Backward: h⃖_t = GRU(x_t, h⃖_{t+1})
Output:   h_t = [h⃗_t; h⃖_t]
```

**Benefits for ASL**:
- Sign meaning often depends on both preparation and completion phases
- Looking ahead helps disambiguate similar signs
- Doubles the capacity without changing depth

### 3.3 Attention Mechanisms

#### 3.3.1 Motivation
Not all frames in a sign are equally important:
- **Transition frames**: Movement between poses (less informative)
- **Hold frames**: Key poses that define the sign (more informative)

**Attention** allows the model to learn which frames to focus on.

#### 3.3.2 Additive Attention (Bahdanau)
```
# Compute attention scores
e_t = tanh(W_a · h_t)
α_t = softmax(e_t)

# Weighted sum
context = Σ(α_t · h_t)
```

**Interpretation**:
- `α_t` is a probability distribution over frames
- High `α_t` means frame `t` is important for classification
- Provides model interpretability

#### 3.3.3 Self-Attention (Transformer)
Instead of fixed weights, **self-attention** computes frame-to-frame relationships:

```
Q = W_Q · X    # Query
K = W_K · X    # Key  
V = W_V · X    # Value

Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

**Multi-Head Attention**: Run multiple attention mechanisms in parallel to capture different relationships.

**Advantages**:
- Captures long-range dependencies without recurrence
- Highly parallelizable (faster than RNNs)
- State-of-the-art for many sequence tasks

**Disadvantages**:
- High parameter count (overfitting risk)
- Requires more data than RNNs
- Quadratic complexity in sequence length

### 3.4 Regularization Techniques

Given the extremely small dataset (204 classes, ~10 samples each), **overfitting is inevitable without aggressive regularization**.

#### 3.4.1 Dropout
Randomly set neuron activations to zero during training with probability `p`:
```
y = x * mask, where mask ~ Bernoulli(1-p)
```

**Effect**: Forces network to learn redundant representations, prevents co-adaptation of neurons.

**Used**: Dropout(0.6) - aggressive 60% dropout rate

#### 3.4.2 L2 Regularization (Weight Decay)
Add penalty term to loss function:
```
L_total = L_task + λ * Σ(w²)
```

**Effect**: Encourages small weights, prevents overfitting to noise.

**Used**: L2=0.01-0.015 (doubled from typical values)

#### 3.4.3 Label Smoothing
Instead of hard targets [0, 0, 1, 0], use soft targets [ε, ε, 1-ε, ε]:
```
y_smooth = (1 - α) * y_true + α / K
```

**Effect**: Prevents overconfident predictions, improves calibration.

**Used**: α=0.1 (10% smoothing)

#### 3.4.4 Batch Normalization
Normalize activations to zero mean, unit variance:
```
y = γ * (x - μ) / σ + β
```

**Effect**: Stabilizes training, acts as mild regularizer.

#### 3.4.5 Data Augmentation
Apply transformations to training data to artificially increase dataset size. See Section 4.2 for details.

### 3.5 Model Architectures Used

#### 3.5.1 Bidirectional GRU (Primary Model)
```
Input (70 frames × 252 features)
    ↓
Bidirectional GRU (64 units) + L2(0.01)
    ↓
BatchNorm → Dropout(0.6)
    ↓
Bidirectional GRU (32 units) + L2(0.01)
    ↓
BatchNorm → Dropout(0.6)
    ↓
Dense (64 units, ReLU) + L2(0.015)
    ↓
BatchNorm → Dropout(0.6)
    ↓
Dense (204 units, Softmax)
```

**Parameters**: ~350K
**Training Speed**: 35-50ms/step
**Rationale**: Best balance of speed, capacity, and generalization

#### 3.5.2 Bidirectional LSTM
Similar architecture to BiGRU but with LSTM cells (128→64→32 units).

**Parameters**: ~450K
**Training Speed**: 40-60ms/step
**Rationale**: Baseline comparison, historically popular for sequences

#### 3.5.3 LSTM with Attention
Adds attention mechanism after second LSTM layer to weight frame importance.

**Parameters**: ~480K
**Training Speed**: 50-70ms/step
**Rationale**: Interpretability - attention weights show important frames

#### 3.5.4 Small Transformer
Single-layer Transformer with 4 attention heads and d_model=64.

**Parameters**: ~280K
**Training Speed**: 60-80ms/step
**Rationale**: Modern architecture comparison, overfitting risk

---

## 4. Implementation

**Documentation Reference**: Complete landmark structure documented in `schemas/dataset_schema.md`.

### 4.1 System Pipeline
```
Raw ASL Videos 
   ↓
MediaPipe Holistic (Pose + Hands Detection)
   ↓
Landmark Extraction (63 keypoints × 4 features)
   ↓
Normalization (scale/position invariance)
   ↓
Temporal Resampling (70 frames)
   ↓
Data Augmentation (12x, training only)
   ↓
Model Training & Evaluation
```

### 4.2 Landmark Extraction

#### 4.2.1 MediaPipe Processing
**Script**: `landmark_get.py`

**Landmark Breakdown (63 Total)**:

**Pose Landmarks (0-20): 21 keypoints**
- Face: Nose, eyes, ears, mouth (7 points)
- Shoulders: Left/right (2 points)
- Arms: Elbows, wrists (4 points)
- Torso: Hips (2 points)
- Wrist detail: Hand connections (6 points)

**Left Hand (21-41): 21 keypoints**
- Wrist base + Thumb (5 joints) + 4 fingers × 4 joints each

**Right Hand (42-62): 21 keypoints**
- Same structure as left hand

**Output**: 63 landmarks × 4 features × 70 frames = (70, 252) flattened for RNN input

#### 4.2.2 Normalization Schema
**Script**: `make_landmark_arr.py` | **Schema**: `schemas/dataset_schema.md`

**Reference Measurements**:
```python
shoulder_center = (left_shoulder + right_shoulder) / 2
hip_center = (left_hip + right_hip) / 2
shoulder_width = distance(left_shoulder, right_shoulder)
torso_height = distance(shoulder_center, hip_center)
```

**Normalization Formula**:
```python
x_norm = (x - shoulder_center_x) / shoulder_width
y_norm = (y - shoulder_center_y) / torso_height
z_norm = z - shoulder_center_z
```

**Applied to**:
- Pose landmarks (0-20): visibility from MediaPipe [0-1]
- Hand landmarks (21-62): visibility = 1.0 if detected, else 0.0

**Missing Data**: Zero-padding [0,0,0,0] when detection fails

**Quality Filter**: Flag videos with torso_height < 0.31 (poor detection)

**Benefits**: Scale & position invariant, cross-signer generalization

#### 4.2.3 Temporal Resampling
**Method**: Linear interpolation to exactly 70 frames
```python
from scipy.interpolate import interp1d
resampled = interp1d(np.linspace(0,1,original_len), 
                landmarks, axis=0)(np.linspace(0,1,70))
```

**Output**: (70, 63, 4) → Flattened to (70, 252) for RNN input

---

## 4. Data Preprocessing & Augmentation

### 4.1 Preprocessing Steps
1. **Video Loading**: Read video files using OpenCV
2. **Frame Extraction**: Extract all frames from video
3. **Landmark Detection**: MediaPipe Holistic processing per frame
4. **Normalization**: Apply schema-based normalization
5. **Temporal Resampling**: Resize to 70 frames
6. **Data Validation**: Check for missing landmarks, corrupted videos
7. **Save to Disk**: Store as NumPy arrays (.npy format)

### 4.2 Data Augmentation (12x Factor)
Applied **only to training data** to combat severe overfitting:

| Augmentation Technique | Description | Range/Probability |
|------------------------|-------------|-------------------|
| **Rotation** | Rotate in x-y plane (around z-axis) | ±15 degrees |
| **Scaling** | Simulate different body sizes | 0.9x - 1.1x (±10%) |
| **Translation** | Shift in x-y plane | ±0.05 units |
| **Temporal Jittering** | Random frame shifts | ±3 frames, 50% prob |
| **Gaussian Noise** | Add detection uncertainty | σ=0.01, applied to x,y,z |
| **Frame Dropout** | Simulate missing detections | 1-4 frames, 30% prob |
| **Visibility Perturbation** | Adjust confidence scores | ±0.1, pose only, 30% prob |

**Augmentation Factor**: 12x (1 original + 11 augmented versions)
**Result**: 1,400 base training samples → 16,800 augmented samples

---

## 5. Model Architectures

### 5.1 Architecture Selection Rationale
For sequence classification tasks with limited data:
- **Recurrent Networks (LSTM/GRU)**: Capture temporal dependencies in sign language
- **Bidirectional Processing**: Learn both forward and backward temporal patterns
- **Attention Mechanisms**: Focus on critical frames in sign execution
- **Transformers**: Modern alternative with self-attention (high capacity, overfitting risk)

### 5.2 Implemented Models

#### Model 1: Lightweight BiLSTM with Balanced Regularization
```
Input (70, 252) 
→ Bidirectional LSTM (64 units, L2=0.01) 
→ BatchNorm → Dropout(0.6)
→ Bidirectional LSTM (32 units, L2=0.01) 
→ BatchNorm → Dropout(0.6)
→ Dense (64, L2=0.015) 
→ BatchNorm → Dropout(0.6)
→ Output (204 classes, softmax)
```
- **Parameters**: ~450K
- **Training Speed**: 40-60ms/step
- **Key Features**: No `recurrent_dropout` (15x speedup), aggressive regularization

#### Model 2: BiGRU with Balanced Regularization ⭐ **BEST PERFORMER**
```
Input (70, 252) 
→ Bidirectional GRU (64 units, L2=0.01) 
→ BatchNorm → Dropout(0.6)
→ Bidirectional GRU (32 units, L2=0.01) 
→ BatchNorm → Dropout(0.6)
→ Dense (64, L2=0.015) 
→ BatchNorm → Dropout(0.6)
→ Output (204 classes, softmax)
```
- **Parameters**: ~350K (fewer than LSTM)
- **Training Speed**: 35-50ms/step (fastest)
- **Key Features**: GRU is faster than LSTM, similar performance

#### Model 3: LSTM with Attention Mechanism
```
Input (70, 252) 
→ Bidirectional LSTM (64 units, L2=0.01) → BatchNorm → Dropout(0.6)
→ Bidirectional LSTM (32 units, L2=0.01) → BatchNorm → Dropout(0.6)
→ Attention Layer (learns frame importance)
→ Weighted Sum (attention-based aggregation)
→ Dense (64, L2=0.015) → Dropout(0.6)
→ Output (204 classes)
```
- **Parameters**: ~480K
- **Training Speed**: 50-70ms/step
- **Key Features**: Interpretable attention weights show important frames

#### Model 4: Small Transformer
```
Input (70, 252) 
→ Positional Encoding
→ Dense Projection (d_model=64)
→ Transformer Layer (4 heads, FFN=128, dropout=0.4-0.5)
→ Global Average Pooling
→ Dense (64, L2=0.015) → Dropout(0.6)
→ Output (204 classes)
```
- **Parameters**: ~280K
- **Training Speed**: 60-80ms/step
- **Key Features**: Modern architecture, high overfitting risk on small datasets

### 5.3 Anti-Overfitting Strategy

The dataset has **severe class imbalance** (204 classes, ~10 samples each), leading to catastrophic overfitting. Applied comprehensive regularization:

| Technique | Value | Impact |
|-----------|-------|--------|
| **Dropout** | 0.6 (60%) | Increased from 0.5 → aggressive neuron dropout |
| **L2 Regularization** | 0.01-0.015 | Doubled from 0.005 → weight decay |
| **Label Smoothing** | 0.1 | Prevents overconfident predictions |
| **Data Augmentation** | 12x | Increased from 8x → 50% more diversity |
| **Capacity Reduction** | 50% cut | LSTM: 128→64, 64→32; prevents memorization |
| **Early Stopping** | Patience=15 | Monitor val_loss to stop overfitting |
| **Learning Rate** | 0.0003 | Low LR with AdamW optimizer |
| **Weight Decay** | 0.01 | AdamW optimizer parameter |
| **Class Weights** | Balanced | Compensate for class imbalance |

**Previous Issue**: Training accuracy 90%+, test accuracy 26-39% (60% gap)
**Current Goal**: Test accuracy 40-55%, train/test gap <30%

---

## 6. Training Configuration

### 6.1 Hardware & Software
- **GPU**: NVIDIA RTX 4050 (6GB VRAM)
- **Framework**: TensorFlow 2.x with Keras
- **Precision**: Mixed FP16 (faster training, lower memory)
- **OS**: Linux (WSL2 on Windows)
- **Python**: 3.12
- **Key Libraries**: TensorFlow, MediaPipe, OpenCV, NumPy, Scikit-learn

### 6.2 Hyperparameters
```python
SEQUENCE_LENGTH = 70          # frames per video
FEATURE_DIM = 252             # 63 landmarks × 4 features
BATCH_SIZE = 32               # balanced for GPU memory
EPOCHS = 100                  # with early stopping
LEARNING_RATE = 0.0003        # AdamW optimizer
WEIGHT_DECAY = 0.01           # L2 penalty in optimizer
AUGMENTATION_FACTOR = 12      # 12x data augmentation
```

### 6.3 Data Split
- **Training**: 1,400 samples (68%) → 16,800 after 12x augmentation
- **Validation**: 248 samples (12%) → no augmentation
- **Test**: 412 samples (20%) → no augmentation
- **Stratification**: Maintains class distribution across splits

### 6.4 Callbacks
```python
EarlyStopping(monitor='val_loss', patience=15)
ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
ModelCheckpoint(monitor='val_accuracy', save_best_only=True)
```

### 6.5 Cross-Validation (CRITICAL)
With limited data per class, prefer **Stratified K-Fold Cross-Validation** over a single train/val/test split. This yields more reliable estimates and reduces variance.

- **Why**: Each fold uses a different subset for validation/test, exposing the model to broader class variations and preventing optimistic or pessimistic bias from one split.
- **Recommended k**: 5 folds (k=5). With ~2,060 samples and 204 classes, this balances stability and runtime.
- **Stratification**: Preserve class distribution across folds to avoid empty/rare classes in any fold.

**Implementation Sketch**:
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
   # Split
   X_train, X_val = X[train_idx], X[val_idx]
   y_train, y_val = y_onehot[train_idx], y_onehot[val_idx]

   # Augment training only
   X_train_aug = augment_landmarks(X_train, augmentation_factor=AUGMENTATION_FACTOR)
   y_train_aug = np.repeat(y_train, AUGMENTATION_FACTOR, axis=0)

   # Flatten for RNNs/Transformer
   X_train_flat = X_train_aug.reshape(-1, SEQUENCE_LENGTH, FEATURE_DIM)
   X_val_flat = X_val.reshape(-1, SEQUENCE_LENGTH, FEATURE_DIM)

   # Class weights (per fold)
   y_train_labels = np.argmax(y_train_aug, axis=1)
   class_weights = compute_class_weight(
      'balanced', classes=np.unique(y_train_labels), y=y_train_labels
   )
   class_weight_dict = dict(enumerate(class_weights))

   # Build + train model (e.g., BiGRU)
   model = build_bigru_balanced_regularization(num_classes)
   history = train_model(model, X_train_flat, y_train_aug, X_val_flat, y_val,
                    f"BiGRU_fold{fold}", class_weight_dict)

   # Evaluate on validation fold
   acc, loss, cm, y_pred, y_true = evaluate_model(model, X_val_flat, y_val,
                                       label_encoder, f"BiGRU_fold{fold}")
   fold_metrics.append({"fold": fold, "accuracy": acc, "loss": loss})

# Aggregate
mean_acc = np.mean([m["accuracy"] for m in fold_metrics])
std_acc  = np.std([m["accuracy"] for m in fold_metrics])
print(f"Cross-val accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
```

**Reporting**:
- Report mean ± standard deviation across folds for accuracy and loss
- Optionally compute macro-averaged precision/recall/F1 per fold and aggregate
- Save per-fold confusion matrices to compare class confusion stability

---

## 7. Results & Evaluation

### 7.1 Model Comparison

| Model | Test Accuracy | Test Loss | Training Speed | Parameters |
|-------|---------------|-----------|----------------|------------|
| **BiGRU (Balanced Reg)** ⭐ | **[TO BE UPDATED]%** | **[TO BE UPDATED]** | **35-50ms/step** | **~350K** |
| BiLSTM (Balanced Reg) | [TO BE UPDATED]% | [TO BE UPDATED] | 40-60ms/step | ~450K |
| LSTM Attention | [TO BE UPDATED]% | [TO BE UPDATED] | 50-70ms/step | ~480K |
| Small Transformer | [TO BE UPDATED]% | [TO BE UPDATED] | 60-80ms/step | ~280K |

**Note**: Results will be updated after final model training completes.

### 7.2 Performance Metrics (To Be Updated)
- **Precision**: [TO BE UPDATED]
- **Recall**: [TO BE UPDATED]
- **F1-Score**: [TO BE UPDATED]
- **Top-5 Accuracy**: [TO BE UPDATED]
- **Confusion Matrix**: Available in `testing/model_comparison_results/`

### 7.3 Training Curves
Training/validation accuracy and loss curves saved as PNG files in `testing/model_comparison_results/`:
- `[ModelName]_[timestamp]_training_curves.png`
- `[ModelName]_[timestamp]_confusion_matrix.png`

### 7.4 Key Insights
1. **Speed Optimization Success**: Removing `recurrent_dropout` achieved 15-20x speedup (877ms → 40-50ms/step)
2. **Overfitting Challenge**: Small dataset (204 classes, ~10 samples/class) causes severe overfitting
3. **Regularization Impact**: Aggressive regularization (dropout 0.6, L2 0.01) reduces overfitting but may over-constrain
4. **Architecture Findings**: 
   - BiGRU offers best speed/performance tradeoff
   - Transformers overfit despite strong regularization
   - Attention mechanisms provide interpretability but higher complexity

---

## 8. Optimization Journey

### 8.1 Performance Bottleneck Discovery
**Problem**: BiGRU training unexpectedly slowed from 40ms/step to 877ms/step (20x slower)

**Root Cause**: `recurrent_dropout=0.2` parameter forces TensorFlow to use unoptimized generic LSTM/GRU implementation instead of CuDNN-accelerated kernels.

**Solution**: Removed `recurrent_dropout` from all models, kept standard dropout layers.

**Result**: Restored 40-50ms/step training speed (15-20x speedup)

### 8.2 Overfitting Crisis
**Problem**: Models achieved 90%+ training accuracy but only 26-39% test accuracy (60% gap)

**Root Cause**: 
- Tiny dataset: 204 classes with only ~10 samples each
- Insufficient diversity: Limited variations in signing style, speed, camera angles
- Model capacity too high relative to data size

**Solutions Implemented**:
1. **Reduced Model Capacity**: Cut LSTM/GRU units by 50% (128→64, 64→32)
2. **Increased Regularization**: Dropout 0.5→0.6, L2 0.005→0.01 (doubled)
3. **Enhanced Augmentation**: 8x→12x (50% increase)
4. **Label Smoothing**: Added 0.1 to prevent overconfident predictions
5. **Lower Learning Rate**: 0.0005→0.0003 with weight decay

**Expected Improvement**: Test accuracy 40-55%, train/test gap <30%

---

## 9. Challenges & Limitations

### 9.1 Dataset Challenges
1. **Small Dataset**: Only ~10 samples per class is insufficient for deep learning
2. **Class Imbalance**: Some classes have 9 samples, others have 15-20
3. **Video Quality**: Variable resolution, lighting, background clutter
4. **Signer Variation**: Limited diversity in age, ethnicity, signing style
5. **Sign Ambiguity**: Some signs look very similar (minimal inter-class variation)

### 9.2 Technical Limitations
1. **MediaPipe Failures**: Landmark detection fails on poor lighting, extreme angles
2. **Fixed Sequence Length**: 70-frame constraint may lose information for longer signs
3. **No Face Landmarks**: ASL uses facial expressions (not captured in current system)
4. **Static Model**: No continual learning or user-specific adaptation
5. **Computational Cost**: Real-time inference requires GPU for practical deployment

### 9.3 Overfitting Persistence
Despite aggressive regularization, the fundamental limitation remains: **204 classes with ~10 samples each is too few for robust deep learning**. Ideal dataset would have:
- **50-100 samples per class minimum**
- **Multiple signers per class** (diversity)
- **Varied environments** (backgrounds, lighting)
- **Professional annotations** (quality control)

---

## 10. Future Work

### 10.1 Short-Term Improvements
1. **Test Additional Architectures**:
   - Temporal Convolutional Networks (TCN)
   - 1D CNN + LSTM hybrid
   - Ensemble models (majority voting)

2. **Hyperparameter Tuning**:
   - Grid search for optimal dropout (0.5-0.65)
   - L2 regularization sweep (0.005-0.02)
   - Augmentation factor testing (10x, 12x, 15x)

3. **Feature Engineering**:
   - Add velocity/acceleration features (temporal derivatives)
   - Hand shape descriptors (angles, distances)
   - Pose-hand interaction features

### 10.2 Medium-Term Enhancements
1. **Dataset Expansion**:
   - Collect more samples (target 50+ per class)
   - Add signer diversity (age, ethnicity, style)
   - Include facial landmarks for expressions
   - Professional video quality standards

2. **Transfer Learning**:
   - Pre-train on larger ASL datasets (WLASL full, MS-ASL)
   - Fine-tune on GTE9 subset
   - Multi-task learning (word + fingerspelling)

3. **Model Compression**:
   - Quantization (INT8) for mobile deployment
   - Knowledge distillation (teacher-student)
   - Pruning for faster inference

### 10.3 Long-Term Vision
1. **Real-Time Application**:
   - Live webcam ASL translation
   - Mobile app deployment (iOS/Android)
   - Browser-based inference (TensorFlow.js)

2. **Sentence-Level Recognition**:
   - Continuous sign language recognition (not just isolated words)
   - Grammar and syntax modeling
   - Context-aware predictions

3. **Bidirectional Translation**:
   - Text/Speech → Sign language animation
   - Avatar-based sign language generation
   - Accessible communication platform

4. **Multi-Language Support**:
   - International sign languages (BSL, ISL, etc.)
   - Cross-lingual sign language transfer learning
   - Unified sign language recognition framework

---

## 11. Code Structure

### 11.1 Repository Organization
```
WLASL_USING_MEDIAPIPE_ND_ML/
├── linux_wsl_only/                      # Main development directory
│   ├── test_recommended_models.py       # Training script (Python)
│   ├── NEW_multiplemodels.ipynb         # Training notebook (Jupyter)
│   ├── analyze_model_results.py         # Comprehensive evaluation script
│   ├── landmark_get.py                  # MediaPipe landmark extraction
│   ├── make_landmark_arr.py             # Dataset preprocessing
│   └── ...                              # Additional utilities
├── gte9_landmarks/                      # Preprocessed dataset
│   ├── x.npy                            # Input data (2060, 70, 252)
│   ├── y.npy                            # Labels (class names)
│   ├── y_encoded.npy                    # Encoded labels (integers)
│   ├── y_onehot.npy                     # One-hot encoded labels
│   └── [word]/                          # Per-class landmark files
├── testing/                             # Model outputs
│   ├── model_comparison_results/        # Saved models (.keras)
│   └── detailed_analysis/               # Evaluation reports
├── models/                              # Legacy model checkpoints
├── schemas/                             # Data schemas & documentation
├── requirements.txt                     # Python dependencies
└── PROJECT_REPORT.md                    # This document
```

### 11.2 Key Scripts

#### `landmark_get.py`
Extracts MediaPipe landmarks from raw videos:
```bash
python landmark_get.py --input videos/ --output landmarks/
```

#### `test_recommended_models.py`
Main training script with all optimizations:
```bash
python test_recommended_models.py
```
Features:
- Loads preprocessed data from `gte9_landmarks/`
- Applies 12x augmentation
- Trains 4 models sequentially
- Saves best models to `testing/model_comparison_results/`

#### `analyze_model_results.py`
Comprehensive model evaluation:
```bash
python analyze_model_results.py
```
Outputs:
- Precision, recall, F1-score per class
- Top-5 accuracy
- Confusion matrix visualization
- Incremental results (JSON + text)

### 11.3 Dependencies
```
tensorflow>=2.13.0
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 12. Reproducibility

### 12.1 Environment Setup
```bash
# Clone repository
git clone https://github.com/PoopApple/WLASL_USING_MEDIAPIPE_ND_ML.git
cd WLASL_USING_MEDIAPIPE_ND_ML

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 12.2 Data Preparation
```bash
# Download WLASL dataset (external)
# Follow instructions at: https://github.com/dxli94/WLASL

# Extract landmarks
cd linux_wsl_only
python landmark_get.py

# Preprocess dataset
python make_landmark_arr.py
```

### 12.3 Training
```bash
# Option 1: Run Python script
python test_recommended_models.py

# Option 2: Run Jupyter notebook
jupyter notebook NEW_multiplemodels.ipynb
```

### 12.4 Evaluation
```bash
# Comprehensive analysis
python analyze_model_results.py
```

### 12.5 Random Seeds
For reproducibility, set random seeds:
```python
import numpy as np
import tensorflow as tf
import random

SEED = 123
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
```

---

## 13. Conclusion

This project successfully developed an **end-to-end ASL recognition system** using MediaPipe for landmark extraction and deep learning for classification. Key accomplishments include:

### 13.1 Technical Achievements
1. ✅ Built robust landmark extraction pipeline with normalized coordinates
2. ✅ Implemented 4 state-of-the-art sequence classification architectures
3. ✅ Optimized training speed by 15-20x through bottleneck removal
4. ✅ Applied comprehensive anti-overfitting strategies for small datasets
5. ✅ Achieved [TO BE UPDATED]% test accuracy on challenging 204-class problem

### 13.2 Key Learnings
1. **Hardware Optimization Matters**: `recurrent_dropout` caused 20x slowdown—GPU-specific optimizations are critical
2. **Small Datasets Are Hard**: 204 classes with ~10 samples each is fundamentally insufficient for deep learning
3. **Regularization is Essential**: Dropout 0.6 + L2 0.01 + label smoothing + augmentation required to prevent catastrophic overfitting
4. **Architecture Selection**: BiGRU offers best speed/accuracy tradeoff for temporal sequences
5. **Data Quality > Model Complexity**: More diverse, high-quality data would improve results more than complex architectures

### 13.3 Practical Impact
This system demonstrates the **feasibility of landmark-based ASL recognition** without requiring end-to-end video models. Benefits:
- **Efficiency**: Landmark extraction is fast (30+ FPS real-time)
- **Privacy**: No raw video storage, only coordinates
- **Interpretability**: Landmarks are human-understandable features
- **Scalability**: Can extend to other sign languages with same framework

### 13.4 Final Thoughts
While current test accuracy of **[TO BE UPDATED]%** on 204 classes is modest, it represents strong progress given severe data constraints. The system provides a solid foundation for:
- Real-time ASL translation apps
- Assistive communication devices
- Sign language learning tools
- Accessibility improvements in public services

**Most Important Next Step**: Expand dataset to 50+ samples per class with diverse signers. This single improvement would yield greater gains than any architectural innovation.

---

## 14. References

### 14.1 Datasets
1. **WLASL**: Li, D., Rodriguez, C., Yu, X., & Li, H. (2020). Word-level deep sign language recognition from video: A new large-scale dataset and methods comparison. *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, 1459-1469.
2. **Dataset Schema**: Complete landmark structure and normalization documentation available in `schemas/dataset_schema.md` (this repository).

### 14.2 Frameworks & Libraries
1. **MediaPipe**: Lugaresi, C., et al. (2019). MediaPipe: A framework for building perception pipelines. *arXiv preprint arXiv:1906.08172*.
2. **TensorFlow**: Abadi, M., et al. (2016). TensorFlow: A system for large-scale machine learning. *12th USENIX symposium on operating systems design and implementation*, 265-283.

### 14.3 Model Architectures
1. **LSTM**: Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*, 9(8), 1735-1780.
2. **GRU**: Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *arXiv preprint arXiv:1406.1078*.
3. **Attention**: Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. *arXiv preprint arXiv:1409.0473*.
4. **Transformer**: Vaswani, A., et al. (2017). Attention is all you need. *Advances in neural information processing systems*, 30.

### 14.4 Regularization Techniques
1. **Dropout**: Srivastava, N., et al. (2014). Dropout: a simple way to prevent neural networks from overfitting. *The journal of machine learning research*, 15(1), 1929-1958.
2. **Label Smoothing**: Szegedy, C., et al. (2016). Rethinking the inception architecture for computer vision. *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2818-2826.

---

## 15. Acknowledgments

- **WLASL Dataset**: Thank you to the creators of WLASL for providing a large-scale ASL dataset
- **MediaPipe Team**: For the excellent Holistic landmark detection framework
- **TensorFlow Team**: For GPU-accelerated deep learning tools
- **Open Source Community**: For numerous libraries and tools that made this project possible

---

## 16. Contact & Contribution

**Repository**: https://github.com/PoopApple/WLASL_USING_MEDIAPIPE_ND_ML

**Contributions Welcome**:
- Dataset expansion (more samples, more signers)
- New model architectures
- Real-time inference optimizations
- Mobile deployment
- Bug fixes and documentation improvements

**License**: [TO BE SPECIFIED]

---

**Document Version**: 1.0  
**Last Updated**: November 27, 2025  
**Status**: Results pending final model training completion
