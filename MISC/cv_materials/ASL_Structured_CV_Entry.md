# American Sign Language Recognition (WLASL GTE9)

## One-Line Domain Tag
Computer Vision • Machine Learning • Systems • Linux • Data Engineering

## Tech Stack
- **Deep Learning Framework**: TensorFlow 2.20.0, Keras 3.12.0
  - Layers: `LSTM`, `GRU`, `Bidirectional`, `Dense`, `Dropout`, `BatchNormalization`
  - Optimizers: `AdamW` with weight decay
  - Losses: `categorical_crossentropy`, label smoothing
  - Callbacks: `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`, `TensorBoard`
- **Computer Vision**: 
  - MediaPipe 0.10.21 (Tasks API: `PoseLandmarker`, `HandLandmarker` in `RunningMode.VIDEO`)
  - OpenCV 4.11.0 (`cv2` for video I/O, frame manipulation, flipping)
- **Scientific Computing**: 
  - NumPy 2.3.4 (array operations, linear interpolation via `np.linspace`, normalization)
  - SciPy 1.16.2 (linear algebra with `np.linalg.norm`)
- **Machine Learning Utilities**: 
  - scikit-learn 1.7.2 (`train_test_split`, `LabelEncoder`, `confusion_matrix`, `classification_report`, `precision_recall_fscore_support`, `top_k_accuracy_score`)
- **Visualization**: 
  - Matplotlib 3.10.7 (training curves, 3D landmark plots via `mpl_toolkits.mplot3d.Axes3D`)
  - Seaborn 0.13.2 (confusion matrix heatmaps)
- **GPU Acceleration**: 
  - NVIDIA CUDA 12.9, cuDNN 9.14.0, CUDA Toolkit (cublas, cufft, cusolver, cusparse, nccl)
  - Mixed precision training (`tf.keras.mixed_precision` with FP16 policy)
  - CuDNN-optimized RNN kernels (BiGRU/BiLSTM)
- **Parallel Processing**: Python `multiprocessing` (batch landmark extraction with configurable worker pools)
- **Development & Monitoring**: 
  - Jupyter (IPython 9.7.0, ipykernel 7.1.0)
  - TensorBoard 2.20.0 (real-time training monitoring)
- **Data Serialization**: Protocol Buffers (MediaPipe `landmark_pb2`), JSON, CSV
- **Environment**: Python 3.12, Linux (Ubuntu/WSL2)

## Context and Motivation
- Real-world problem: Recognize isolated ASL words from video to reduce reliance on human interpreters and improve accessibility in everyday settings.
- Engineering significance: Deliver real-time, privacy-preserving recognition using skeletal landmarks while handling severe data scarcity (~10 samples/class), requiring robust temporal modeling and strong regularization.

## Core Idea
- Concept: Convert raw ASL videos into normalized pose + hand landmark sequences (70 × 63 × 4) using MediaPipe Tasks, then classify using compact, bidirectional RNNs (BiGRU/BiLSTM) optimized for small datasets.
- Why this approach: Landmark features are efficient, interpretable, and robust to lighting/backgrounds. Simpler RNNs with strong regularization generalize better than pixel-heavy CNNs or high-capacity Transformers under limited data.

## System Design or Architecture
- Structure:
  - Ingestion → MediaPipe Tasks landmark extraction (pose + hands) → Shoulder/Torso normalization → Fixed-length sequence (70 frames) → `.npy` storage → Train/Val/Test split (80/15/5) → 8× augmentation → RNN training with early stopping → Evaluation (accuracy, Top‑k, confusion).
- Components:
  - Landmark extraction: `NEW_LM_VISION_TASKS.py` using `PoseLandmarker` and `HandLandmarker` (Video mode) with assets under `vision_models/`.
  - Dataset artifacts: `gte9_landmarks/` containing `x.npy`, `y.npy`, `y_encoded.npy`, `y_onehot.npy` and per‑class `.npy` files.
  - Training/Evaluation: scripts in `linux_wsl_only/` (e.g., `NEW_train_pipeline.py`, `train_test1_using_ltsm.py`, `analyze_model_results.py`).
  - Outputs: curves, confusion matrices, summaries in `testing/model_comparison_results/`.
- Key decisions & tradeoffs:
  - Landmark normalization (scale/position invariance) vs pixel fidelity.
  - Compact RNNs + heavy regularization vs complex models (overfit risk).
  - Fixed 70‑frame sequences for simplicity vs variable‑length complexity.
  - 8× augmentation to expand limited training data vs computational overhead.

## Key Features with Intent
- Feature 1: MediaPipe Tasks‑based landmark extraction
  - What: Pose + two-hand landmarks via `PoseLandmarker` and `HandLandmarker` in Video mode.
  - Why: Efficient, privacy‑preserving skeletal features.
  - Solves: Removes dependency on raw pixels; robust to background/lighting noise.
- Feature 2: Body‑relative normalization + sticky hands
  - What: Normalize x/y by shoulder width and torso height; backfill missing hand frames with last known values.
  - Why: Cross‑signer invariance and reduced zeros.
  - Solves: Stabilizes features; mitigates intermittent detection.
- Feature 3: 8× data augmentation pipeline
  - What: Gaussian noise (σ=0.01), temporal jitter, landmark scaling to generate synthetic training samples; expands ~1700 → ~13,600 sequences.
  - Why: Overcome severe data scarcity (~10 samples/class) and improve generalization.
  - Solves: Reduces overfitting by exposing model to variations without collecting more videos.
- Feature 4: Compact, regularized RNN classifiers
  - What: BiGRU/BiLSTM with dropout (0.4–0.6), L2 (0.01–0.015), label smoothing (0.1), AdamW optimizer.
  - Why: Control overfitting with ~10 samples/class; strong penalties prevent memorization.
  - Solves: Achieves practical accuracy on small datasets without complex architectures.
- Feature 5: Reproducible evaluation pipeline
  - What: Stratified train/val/test split, Top‑k metrics, confusion matrices, incremental result exports, early stopping, model checkpointing.
  - Why: Comparable runs, prevent overfitting, transparent diagnostics.
  - Solves: Traceability and analysis of model behavior; saves best models automatically.

## Your Technical Contributions
- Implemented landmark pipeline in `NEW_LM_VISION_TASKS.py` (Tasks API `RunningMode.VIDEO`), landmark selection, and shoulder/torso normalization; added sticky‑hands backfilling.
- Designed 8× data augmentation strategy (Gaussian noise, temporal jitter, landmark scaling) that increased effective training data from ~1700 to ~13,600 samples.
- Built training and evaluation tooling in `linux_wsl_only/`, including `analyze_model_results.py` for top‑k exports and confusion matrices; implemented stratified split with class balancing.
- Engineered regularization stack (dropout 0.4–0.6, L2 0.01–0.015, label smoothing 0.1, batch normalization) tuned for small data; optimized training speed by removing `recurrent_dropout` to enable CuDNN kernels (15–20× speedup).
- Implemented ML pipeline components: early stopping (patience=15), model checkpointing (save best only), learning rate scheduling (ReduceLROnPlateau), and mixed precision training (FP16).
- Organized dataset artifacts (`gte9_landmarks/`) with proper train/val/test splits, label encodings (integer + one-hot), and shapes suitable for RNNs.

## Engineering Challenges
- Overfitting under data scarcity
  - Difficulty: Models memorize sequences; validation stagnates.
  - Solution: Reduced model capacity, stronger regularization, 8× augmentation, label smoothing; guided by `SMALL_DATASET_GUIDE.md`.
- Training performance degradation with `recurrent_dropout`
  - Difficulty: CuDNN kernels disabled → 15–20× slowdown.
  - Solution: Removed recurrent dropout; retained standard dropout; restored ~35–50 ms/step.

## Performance, Scalability, Reliability Considerations
- Behavior with scale: More samples/classes improve generalization; current design favors small datasets with compact models.
- Optimizations: CuDNN‑accelerated RNNs, batched training/inference, lean feature dimensionality (252 features/frame).
- For larger scale: Dataset expansion, transfer learning (WLASL/MS‑ASL), quantization/distillation for mobile, real-time streaming.

## Validation and Results
- Method: 
  - Stratified train/val/test split (80/15/5) to ensure class balance across splits.
  - Metrics: Top‑1/Top‑5 accuracy, categorical cross-entropy loss, precision/recall/F1 (macro & weighted), confusion matrices.
  - Validation: Per‑sample top‑5 predictions export, class-wise performance analysis, training curves monitoring.
  - Hardware: GPU-accelerated training (~35–50ms/step with CuDNN kernels).
- Best run (171 classes, 8× aug, 9,288 train samples, 342 test samples): 
  - Model: BiGRU Balanced Regularization (64→32 units, dropout 0.6, L2 0.01)
  - Top‑1: 49.71%, Top‑5: 78.65%, Train Acc: 93.26%, Loss: 2.998
  - Training: ~100 epochs, early stopping triggered, AdamW optimizer (lr=3e-4)
  - Source: `testing/model_comparison_results/summary_20251127_080903.txt`.
- Secondary: Lightweight BiLSTM → Top‑1 32.46%, Top‑5 65.50%, Loss 3.4534.
- Comparison baseline: Random guessing ~0.58% (1/171), demonstrating significant improvement.

## Learning and Impact
- Lessons: Landmark normalization + compact temporal models outperform pixel‑heavy approaches under data scarcity; regularization and augmentation are essential.
- Transferable skills: Resilient pipeline design, diagnosing overfitting, GPU kernel performance tuning, reproducible evaluation frameworks.
