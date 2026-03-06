# CV BULLET POINTS & ACHIEVEMENTS
## American Sign Language Recognition Project

---

## EXECUTIVE SUMMARY FOR CV

Built a **complete computer vision and deep learning system** that automatically recognizes and classifies American Sign Language (ASL) words from video. Achieved **34.5% accuracy on challenging 204-class classification task** using MediaPipe landmark extraction and recurrent neural networks, demonstrating mastery of **ML optimization, deep learning architectures, and real-time video processing**.

---

## TECHNICAL BULLET POINTS

### Computer Vision & MediaPipe Integration
- ✅ Integrated **Google's MediaPipe Holistic** framework for real-time pose and hand landmark detection (30+ FPS)
- ✅ Extracted **63 anatomical landmarks** (pose + left/right hands) with 4D features per frame (x, y, z, confidence)
- ✅ Designed body-centric **normalization schema** for scale and position invariance, enabling cross-signer generalization
- ✅ Implemented **temporal resampling** via linear interpolation for variable-length videos to fixed 70-frame sequences
- ✅ Processed 2,060+ ASL videos in parallel using 12 worker multiprocessing with RAM management (3GB across workers)

### Deep Learning & Model Development
- ✅ Designed and trained **4 state-of-the-art architectures**:
  - Lightweight BiLSTM (210K params) — **best performer at 34.5% accuracy**
  - Bidirectional GRU (350K params) — optimized for temporal sequences
  - LSTM with Additive Attention (480K params) — interpretable attention weights
  - Small Transformer (280K params) — modern self-attention architecture
- ✅ Implemented **bidirectional processing** to capture both forward and backward temporal dependencies in sign language
- ✅ Achieved **70× improvement over random baseline** (34.5% vs 0.49%)

### Data Augmentation & Regularization
- ✅ Engineered **12× data augmentation pipeline** combining 7 transformation techniques:
  - Geometric (rotation ±15°, scaling 0.9-1.1×, translation ±0.05)
  - Temporal (frame jittering ±3 frames)
  - Noise injection (Gaussian σ=0.01, frame dropout, visibility perturbation)
- ✅ Expanded training dataset from 1,400 to 16,800 samples while maintaining semantic consistency
- ✅ Applied **comprehensive regularization** (dropout 0.35-0.6, L2 0.001-0.002, label smoothing 0.1) to combat overfitting on small dataset
- ✅ Reduced **train/test overfitting gap from 68% to 33%** through balanced regularization strategy

### Performance Optimization
- ✅ **Achieved 15-20× training speedup** (877ms → 40-50ms per step) by identifying and removing `recurrent_dropout` GPU bottleneck
- ✅ Optimized model size to **~1.5MB** for mobile deployment while maintaining accuracy
- ✅ Enabled **real-time inference** (<50ms per video) through GPU-accelerated RNN kernels and mixed precision training
- ✅ Profiled and analyzed bottlenecks systematically, applying root cause analysis to GPU optimization issues

### Data Engineering & Preprocessing
- ✅ Built complete **video-to-features pipeline** handling 2,060+ videos with robust error handling
- ✅ Implemented **stratified train/val/test splitting** (68/12/20) preserving class distribution across 204 classes
- ✅ Designed schema for **17,640-feature input vectors** (70 frames × 252 features) with efficient NumPy storage (.npy format)
- ✅ Applied **quality filtering** to identify and handle detection failures (zero-padding for missing landmarks)

---

## ARCHITECTURE & IMPLEMENTATION DETAILS

### Best Performing Model (Lightweight BiLSTM)
```
2 Stacked LSTM Layers (128→64 units)
↓ Bidirectional processing
↓ BatchNormalization & Dropout(0.35-0.6)
↓ 2 Dense layers with L2 regularization (0.001-0.002)
↓ 204-class Softmax output
```
- **Test Accuracy**: 34.50%
- **Top-5 Accuracy**: 70.18%
- **Training Speed**: 35-45ms/step
- **Parameters**: 210K

### Regularization Innovations
- Combined **multiple regularization techniques** in balanced way:
  - Aggressive dropout (0.35-0.6) for small dataset
  - Moderate L2 (0.001-0.002) avoiding underfitting
  - Label smoothing for calibrated predictions
  - Batch normalization for gradient stability
- Achieved **optimal balance** between model capacity and generalization

---

## METRICS & QUANTIFIABLE RESULTS

### Accuracy Metrics
- **Test Accuracy**: 34.50% (best model)
- **Top-5 Accuracy**: 70.18% (correct answer in top 5)
- **Training Accuracy**: 67.50%
- **Baseline (random guessing)**: 0.49%
- **Improvement**: **70× better than random**

### Performance Metrics
- **Training Speed**: 35-50ms per step (optimized)
- **Epoch Duration**: 20-30 minutes
- **Total Training Time**: 50-80 hours (100 epochs)
- **Inference Speed**: <50ms per video
- **Model Size**: ~1.5MB (mobile-ready)

### Dataset Scale
- **Total Videos**: 2,060
- **Classes**: 204 ASL words
- **Samples/Class**: ~10 (extremely limited)
- **Feature Dimension**: 17,640 (70 frames × 252 features)
- **Total Landmarks Extracted**: 9.1 million 3D points
- **Augmented Training Samples**: 16,800 (12× expansion)

### System Metrics
- **Multiprocessing Workers**: 12 (parallel video processing)
- **MediaPipe FPS**: 30+ (real-time detection)
- **Training/Test Gap**: 33% (improved from 68%)
- **Storage**: 142 MB (compressed landmarks)

---

## TECHNOLOGIES & TOOLS MASTERED

### Deep Learning Frameworks
- **TensorFlow 2.13+** — Model building, training, optimization
- **Keras API** — High-level neural network design
- **CUDA/cuDNN** — GPU acceleration for RNNs and operations
- **Mixed Precision Training** — FP16 for faster training

### Computer Vision
- **MediaPipe 0.10+** — Pose/hand landmark detection
- **OpenCV 4.8+** — Video I/O, frame processing
- **NumPy** — Efficient array operations (9.1M landmarks)
- **SciPy** — Interpolation, linear algebra

### Data Science & ML
- **Scikit-learn** — Preprocessing, class weights, metrics
- **Matplotlib** — Confusion matrices, training curves
- **Python 3.12** — Core implementation language

### Hardware & Infrastructure
- **NVIDIA RTX 4050** — GPU acceleration (6GB VRAM)
- **Linux WSL2** — Development environment
- **Multi-core CPU** — Parallel processing (12 workers)

---

## CHALLENGES SOLVED

### Challenge 1: Catastrophic Overfitting
- **Problem**: 91% training accuracy, 23% test accuracy (68% gap)
- **Root Cause**: Insufficient regularization for 204 classes with ~10 samples each
- **Solution**: Applied comprehensive regularization (dropout 0.6, L2 0.01) + 12× augmentation
- **Result**: Reduced overfitting gap to 33%, improved test accuracy to 34.5%
- **Learning**: Data constraints require aggressive regularization; model capacity is secondary

### Challenge 2: 20× Training Slowdown
- **Problem**: BiGRU training slowed from 40ms to 877ms per step (unexplainable)
- **Root Cause**: `recurrent_dropout=0.2` forced TensorFlow to disable GPU optimization
- **Investigation**: Systematic profiling identified GPU kernel selection issue
- **Solution**: Removed `recurrent_dropout`, kept standard dropout layers
- **Result**: Restored 40-50ms/step speed (15-20× speedup)
- **Learning**: Understanding framework internals is crucial for optimization

### Challenge 3: Variable-Length Sequences
- **Problem**: Videos range from 24-300+ frames; can't batch different lengths
- **Solution**: Linear interpolation resampling to exactly 70 frames
- **Benefit**: Fixed tensor shapes, efficient batching, preserved temporal information
- **Trade-off**: 70 frames ≈ 2-3 seconds adequate for most ASL words

### Challenge 4: MediaPipe Resource Leaks
- **Problem**: Each video processing increased RAM by ~150MB; no cleanup
- **Solution**: Properly close MediaPipe objects and garbage collect after each video
- **Result**: Enabled processing 2,060+ videos without RAM overflow

### Challenge 5: Data Quality Issues
- **Problem**: Poor lighting, extreme angles caused detection failures
- **Solution**: Zero-padding for missing landmarks, quality filtering (torso_height validation)
- **Result**: Robust system handling real-world video variations

---

## ADVANCED TECHNIQUES IMPLEMENTED

### Normalization Techniques
- **Body-Centric Normalization**: Normalized all landmarks relative to shoulder width and torso height
  - Enables scale invariance (signer height doesn't matter)
  - Enables position invariance (signer location in frame doesn't matter)
  - Improves cross-signer generalization
- **Visibility-Based Weighting**: Used confidence scores from MediaPipe to validate detections

### Sequence Modeling
- **Bidirectional Processing**: Forward + backward RNNs for complete temporal context
- **Temporal Dependencies**: Captured long-range dependencies (70 timesteps)
- **Sequence-to-Label**: Mapped variable-length behaviors to fixed 204-class output

### Attention & Interpretability
- **Additive Attention**: Implemented frame-level attention (Bahdanau mechanism)
- **Model Interpretability**: Attention weights visualized important frames in signing
- **Explainability**: Could identify which frames contribute to classification decisions

### Optimization Techniques
- **Mixed Precision Training**: FP16 for faster training, FP32 for accuracy
- **Learning Rate Scheduling**: Reduced LR when validation loss plateaued
- **Early Stopping**: Monitored validation loss, stopped when overfitting began (patience=15)
- **Class Weighting**: Balanced classes despite imbalanced representation

---

## SYSTEM DESIGN DECISIONS

### Why Landmarks Instead of Raw Video?
- ✅ **Efficiency**: 252-dim feature vector vs 100K+ pixel values
- ✅ **Privacy**: Only coordinates, not raw video
- ✅ **Robustness**: Works across lighting, background, clothing variations
- ✅ **Interpretability**: Human-understandable features

### Why BiLSTM Over LSTM?
- ✅ **Bidirectional**: Sign meaning depends on both preparation and completion
- ✅ **Simpler than Transformer**: RNNs work better with small datasets
- ✅ **Fast Training**: 35-50ms/step enables rapid experimentation

### Why 12× Augmentation?
- ✅ **Dataset Too Small**: ~10 samples/class insufficient for deep learning
- ✅ **Diversity**: Augmentation exposes model to variations (rotation, scale, noise)
- ✅ **Training/Test Generalization**: Reduces overfitting gap from 68% to 33%

### Why Aggressive Regularization?
- ✅ **High Dimensionality**: 17,640 features, only 1,400 training samples
- ✅ **Curse of Dimensionality**: Dropout 0.6, L2 0.01 necessary to prevent memorization
- ✅ **Trade-off**: Slightly underfits training (67%) but generalizes better (34.5% test)

---

## REPRODUCIBILITY & CODE QUALITY

### Reproducible Research
- **Fixed Random Seeds**: Set SEED=123 for NumPy, TensorFlow, Python
- **Version Control**: Complete repository with `.git` history
- **Environment Management**: `requirements.txt` with exact package versions
- **Documentation**: 1000+ line comprehensive PROJECT_REPORT.md

### Code Organization
- **Modular Design**: Separate modules for extraction, training, evaluation
- **Clear Naming**: Functions and variables clearly indicate purpose
- **Error Handling**: Robust handling of video failures, missing landmarks
- **Comments & Docstrings**: Explained complex algorithms and hyperparameter choices

### Testing & Validation
- **Data Validation**: Checked for NaN/Inf values, missing landmarks
- **Model Evaluation**: Computed accuracy, precision, recall, F1-score
- **Confusion Matrices**: Visualized per-class performance
- **Cross-Validation**: Stratified splitting preserved class distribution

---

## WHAT YOU'LL SAY IN INTERVIEWS

**On Technical Depth**:
*"This project required mastery of multiple deep learning architectures. I tested LSTM, GRU, Attention, and Transformer models, ultimately identifying BiLSTM as the best performer for small datasets. The key was systematic regularization—dropout 0.6 and L2 0.01—combined with 12× data augmentation to overcome severe overfitting."*

**On Problem Solving**:
*"Training mysteriously slowed from 40ms to 877ms per step. Through systematic profiling, I discovered `recurrent_dropout` disabled GPU optimization. Removing this single parameter restored 15-20× speedup—a great lesson in understanding framework internals."*

**On Data Engineering**:
*"The dataset only had ~10 samples per 204 classes—extremely limited. I engineered a robust pipeline extracting 63 anatomical landmarks using MediaPipe, then applied body-centric normalization for cross-signer generalization. This enabled models to learn sign semantics rather than signer-specific variations."*

**On Optimization**:
*"With only 1,400 training samples and 17,640-dimensional features, I applied aggressive regularization (dropout 0.6, L2 0.01, label smoothing) combined with 12× data augmentation. This reduced overfitting gap from 68% to 33% and improved test accuracy to 34.5%—70× better than random baseline."*

**On Learning**:
*"The fundamental insight: with small datasets, data quality and regularization matter far more than model complexity. Expanding the dataset to 50+ samples per class would improve accuracy more than any architectural innovation."*

---

## PORTFOLIO PROJECT HIGHLIGHTS

### What Impresses Technical Recruiters
1. **Real-World Constraints**: Solved challenging 204-class classification with only ~10 samples per class
2. **Multiple Architectures**: Tested and compared 4 deep learning approaches with systematic evaluation
3. **Performance Optimization**: Achieved 15-20× speedup through bottleneck identification
4. **Complete Pipeline**: End-to-end system from raw video to predictions
5. **Documentation**: Comprehensive PROJECT_REPORT.md shows communication skills
6. **Problem Solving**: Systematic debugging approach (recurrent_dropout discovery)

### Demonstrates These Competencies
- ✅ Deep Learning (LSTM, GRU, Attention, Transformer)
- ✅ Computer Vision (MediaPipe, video processing, landmark tracking)
- ✅ Data Engineering (14.2M features, augmentation, normalization)
- ✅ Software Engineering (optimization, parallel processing, documentation)
- ✅ ML Fundamentals (overfitting, regularization, train/test splits)
- ✅ Research & Analysis (systematic evaluation, visualization)

---

## NUMBERS FOR YOUR CV

### Quantifiable Achievements
- 📊 **34.5%** test accuracy on 204-class classification (70× better than random)
- 📊 **70.18%** top-5 accuracy (answer in top 5 predictions)
- 📊 **15-20×** training speedup (877ms → 40-50ms per step)
- 📊 **33%** reduced overfitting gap (from 68% to 33%)
- 📊 **12×** data augmentation factor
- 📊 **2,060** videos processed
- 📊 **9.1M** landmarks extracted
- 📊 **4** deep learning architectures tested
- 📊 **63** anatomical landmarks tracked
- 📊 **204** sign classes recognized

---

## WHAT MAKES THIS IMPRESSIVE

### For ML Engineer Roles
- Handled **challenging small-dataset problem** systematically
- Implemented **multiple SOTA architectures** with proper evaluation
- Achieved **significant performance optimization** through investigation
- Demonstrated **deep understanding** of regularization and overfitting

### For CV/Computer Vision Roles
- Integrated **real-time landmark detection** (MediaPipe)
- Applied **normalization techniques** for robustness
- Processed **variable-length video sequences** elegantly
- Extracted **63 anatomical features** with quality assurance

### For Software Engineering Roles
- Built **complete production pipeline** (video → features → model → predictions)
- Optimized **parallel processing** (12 workers, RAM management)
- Achieved **15-20× speedup** through systematic debugging
- Maintained **reproducible, well-documented code**

---

## SUGGESTED CV FORMAT

### Project Title
**American Sign Language Recognition System using MediaPipe and Deep Learning**

### Description (2-3 sentences)
Built an end-to-end computer vision system that automatically recognizes and classifies American Sign Language words from video using MediaPipe landmark extraction and recurrent neural networks. Achieved 34.5% accuracy on challenging 204-class classification task using only 2,060 training videos (~10 per class). Optimized system performance by 15-20× through GPU profiling and engineering comprehensive regularization strategy to combat overfitting.

### Key Achievements
- Engineered complete pipeline extracting 63 anatomical landmarks from 2,060+ ASL videos using Google's MediaPipe framework (30+ FPS real-time detection)
- Designed body-centric normalization schema enabling cross-signer generalization; applied linear interpolation for temporal resampling to fixed 70-frame sequences
- Developed and compared 4 deep learning architectures (BiLSTM, BiGRU, LSTM+Attention, Transformer); BiLSTM achieved **34.5% accuracy** (70× improvement over 0.49% random baseline)
- Engineered 12× data augmentation pipeline combining geometric transformations, temporal jittering, and noise injection; expanded training dataset from 1,400 to 16,800 samples
- Diagnosed and fixed 20× training slowdown (877ms → 40-50ms per step) by identifying `recurrent_dropout` GPU optimization bottleneck
- Applied comprehensive regularization (dropout 0.6, L2 0.01, label smoothing, batch normalization) reducing overfitting gap from 68% to 33%

### Technologies
MediaPipe, TensorFlow, Keras, LSTM/GRU, Python, CUDA/GPU, NumPy, OpenCV, Scikit-learn, Matplotlib

### Metrics
- Test Accuracy: 34.50% | Top-5 Accuracy: 70.18%
- Training Speed: 35-50ms/step | Model Size: ~1.5MB
- Processed 2,060 videos | 9.1M landmarks extracted | 17,640-dim features

---

**Last Updated**: January 4, 2026
**Status**: Ready for CV & Interview Use
