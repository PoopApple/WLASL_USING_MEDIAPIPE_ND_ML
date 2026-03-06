# ASL Recognition — Comprehensive Overview

## Executive Summary
Landmark-based ASL word recognition built on MediaPipe Tasks (pose + hands) and compact bidirectional RNNs. Videos are converted to normalized sequences (70 frames × 63 landmarks × 4 features) and classified across 171+ classes. Best BiGRU achieved Top‑1 49.71% and Top‑5 78.65% on GTE9 subset, with a reproducible pipeline for preprocessing, training, and evaluation.

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

## Key Folders & Files
- Reports: `PROJECT_REPORT.md`, `REFERENCES.md`
- Schemas: `schemas/dataset_schema.md` (shape, normalization, indices), `schemas/recommended_models.md` (architectures & guidance)
- MediaPipe Tasks assets: `vision_models/pose_landmarker_heavy.task`, `vision_models/hand_landmarker.task`
- Landmark extraction: `NEW_LM_VISION_TASKS.py` (Tasks API, normalization, sticky‑hands), `new_detect_modified_landmark_with_np_arr_only.py`
- Training & evaluation: `linux_wsl_only/NEW_train_pipeline.py`, `linux_wsl_only/train_test1_using_ltsm.py`, `linux_wsl_only/analyze_model_results.py`, `linux_wsl_only/SMALL_DATASET_GUIDE.md`
- Preprocessed dataset: `gte9_landmarks/x.npy`, `y.npy`, `y_encoded.npy`, `y_onehot.npy` and per‑class `.npy` files
- Metrics & artifacts: `testing/model_comparison_results/summary_20251127_080903.txt`, `testing/model_comparison_results/incremental_results.json`, training curves & confusion matrices in the same folder
- Dataset stats: `stats/gte9_list.txt`, `stats/stats_freq.json`, `stats/stats_of_vids.ipynb`

## Dataset & Schema
- Subset: WLASL GTE9 (≥9 instances/class); runs here use 171 classes.
- Landmark format: `(70, 63, 4)` → 70 frames × (21 pose + 21 left hand + 21 right hand) × [x,y,z,visibility].
- Normalization: x by shoulder width; y by torso height; z centered at shoulders; visibility from MediaPipe (hands default to 1.0 or zero when missing).
- Missing data: zero‑padding and sticky‑hands (backfill last known hand) to reduce gaps.

## MediaPipe Tasks Used
- Pose: `PoseLandmarker` (Video mode), asset `pose_landmarker_heavy.task`.
  - Docs: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
- Hands: `HandLandmarker` (Video mode, `num_hands=2`), asset `hand_landmarker.task`.
  - Docs: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md
- Implementation references: landmark selection and normalization in `NEW_LM_VISION_TASKS.py` (`RunningMode.VIDEO`).

## Pipeline Overview
1. Extract landmarks per frame (pose + hands) via MediaPipe Tasks.
2. Normalize coordinates to shoulder/torso references for scale/position invariance.
3. Fix sequence length to 70 frames via linear interpolation.
4. Store sequences as `.npy` under `gte9_landmarks/`; aggregate `x.npy`, labels (`y.npy`, `y_encoded.npy`, `y_onehot.npy`).
5. Apply stratified train/val/test split (80/15/5) with class balancing.
6. Data augmentation: 8× expansion using Gaussian noise (σ=0.01), temporal jitter, landmark scaling (~1700 → ~13,600 training samples).
7. Train compact RNN models (BiGRU / lightweight BiLSTM) with dropout (0.4–0.6), L2 (0.01–0.015), label smoothing (0.1), batch normalization.
8. ML training pipeline: AdamW optimizer, early stopping (patience=15), model checkpointing (best only), learning rate scheduling (ReduceLROnPlateau), mixed precision (FP16).
9. Evaluate using Top‑k metrics, confusion matrices, and per‑sample top‑5 exports.

## Models & Training
- Recommended models (see `schemas/recommended_models.md`): BiGRU/BiLSTM, optional attention; Transformers/3D CNNs are prone to overfitting on small data.
- Data augmentation: 8× expansion via Gaussian noise (σ=0.01), temporal jitter, landmark scaling; increases training samples from ~1700 → ~13,600.
- Regularization: Dropout (0.4–0.6), L2 (0.01–0.015), label smoothing (0.1), batch normalization.
- Training setup: AdamW optimizer (lr=3e-4, weight decay), stratified split (80/15/5), categorical cross-entropy loss.
- Pipeline components: Early stopping (patience=15, monitor val_accuracy), model checkpointing (save_best_only), learning rate scheduling (ReduceLROnPlateau), mixed precision training (FP16).
- Performance optimization: Avoid `recurrent_dropout` to retain CuDNN kernels (major speedup); typical ~35–50 ms/step.

## Results
- Best run (171 classes, 8× aug, 9,288 train samples, 342 test samples): BiGRU Balanced Regularization
  - Top‑1: 49.71%
  - Top‑5: 78.65%
  - Train Acc: 93.26%
  - Loss: 2.998
  - Training: ~100 epochs with early stopping, AdamW optimizer (lr=3e-4)
  - Source: `testing/model_comparison_results/summary_20251127_080903.txt`
- Secondary: Lightweight BiLSTM — Top‑1 32.46%, Top‑5 65.50%, Loss 3.4534.
- Baseline comparison: Random guessing ~0.58% (1/171 classes).
- Artifacts: Training curves, confusion matrices, and top‑5 predictions in `testing/model_comparison_results/`.

## Run & Evaluate (Example)
```bash
# Landmark extraction (batch)
python3 NEW_LM_VISION_TASKS.py

# Train (example script)
python3 linux_wsl_only/train_test1_using_ltsm.py

# Comprehensive evaluation
python3 linux_wsl_only/analyze_model_results.py
```

## Challenges & Mitigations
- Overfitting with ~10 samples/class → reduced capacity, strong regularization (dropout 0.4–0.6, L2 0.01–0.015), 8× augmentation, label smoothing (0.1).
- Slow training from `recurrent_dropout` → removed to enable CuDNN kernels (15–20× speedup); retained standard dropout.
- Landmark gaps/misses → sticky‑hands backfilling; quality checks via `linux_wsl_only/check_landmark_quality.py`.
- Class imbalance → stratified splits, class weight balancing during training.
- Validation plateau → early stopping (patience=15), learning rate reduction on plateau.

## Future Scope
- Data: Expand per‑class samples, signer diversity, add facial landmarks; pretrain on full WLASL/MS‑ASL, fine‑tune on GTE9.
- Features & models: Velocity/angles, hand‑shape descriptors; TCNs, 1D‑CNN+RNN hybrids, ensembles.
- Deployment: Quantization/distillation for mobile; real‑time webcam; browser (TF.js).
- Beyond words: Continuous (sentence‑level) recognition; grammar/context modeling; bidirectional text↔sign generation.

## References
- Dataset: WLASL (Li et al., 2020) — arXiv:1910.11006.
- MediaPipe Holistic: Pose/Hands docs linked above; BlazePose & MediaPipe Hands papers.
- Training & optimization: Adam/AdamW, Dropout, BatchNorm, Label Smoothing; mixed precision.
