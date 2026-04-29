# Real-Time ASL Recognition System

A lightweight, word-level American Sign Language (ASL) recognition system optimized for edge devices. This project utilizes MediaPipe for spatial landmark extraction and a Bidirectional Gated Recurrent Unit (BiGRU) for temporal sequence classification across a 106-word vocabulary.

## Key Features
* **Real-Time Inference:** Operates at ~240ms per frame for extraction and ~10-40ms for model inference, making it suitable for live video feeds.
* **Spatial Reduction:** Extracts skeletal landmarks via MediaPipe rather than processing raw RGB video pixels, drastically reducing computational overhead and background noise.
* **Temporal Normalization:** Handles variable-length signs using zero-padding paired with a masking layer, ensuring the recurrent network only learns from actual human movement.
* **Highly Efficient:** The production BiGRU model achieves high accuracy with only ~1.33 million parameters (a 25% memory reduction compared to standard BiLSTM architectures).

## Tech Stack
* **Framework:** TensorFlow / Keras (v2.21)
* **Computer Vision:** MediaPipe, OpenCV
* **Data Processing:** NumPy, Scikit-learn
* **Visualization:** Matplotlib, Seaborn

## Dataset & Preprocessing
* **Source:** Word-Level American Sign Language (WLASL) dataset and Microsoft ASl Citizen dataset.
* **Vocabulary:** 106 distinct ASL signs.
* **Pipeline:** 1. Videos are processed to extract 3D spatial coordinates (X, Y, Z, Visibility).
  2. Data is transformed into a `(128, 64, 4)` tensor representation.
  3. Sequences are padded to a fixed maximum length, and a binary mask is generated for training.
* **Validation:** Stratified 80/20 Train/Test split (Validation support: 1,094 samples).

## 🧠 Model Architecture (BiGRU)
The final production model utilizes a Bidirectional GRU to capture forward and backward temporal dependencies in the sign sequences.

1. **Input & Masking:** `InputLayer` -> `Reshape` -> `Input Masking`
2. **Temporal Processing:**
   * `Bidirectional(GRU)` (512 units)
   * `Bidirectional(GRU)` (256 units)
3. **Regularization:** `Dropout` layer to prevent overfitting.
4. **Classification:** `Dense(ReLU)` -> `Dense(Softmax)` mapping to 106 classes.

## 📈 Evaluation & Benchmarks
An automated evaluation script (`report.py`) generates comprehensive side-by-side comparisons of model architectures, providing:
* Precision, Recall, and F1-Scores per class.
* System efficiency metrics (Parameter count, Inference Latency).
* High-resolution Confusion Matrices and grouped F1-score bar charts.

## ⚙️ Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PoopApple/ASL_recognition.git
   cd ASL_recognition
