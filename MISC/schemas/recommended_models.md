# Recommended Models for GTE9 ASL Dataset

This document provides model architectures optimized for the GTE9 dataset with shape `(70, 63, 4)`.

---

## Dataset Context
- **Input shape**: `(70, 63, 4)` - 70 frames × 63 landmarks × 4 features
- **Classes**: 205 sign words
- **Total samples**: ~2,150 videos (205 classes × ~10.5 videos per class)
- **Task**: Multi-class classification
- **Features**: Temporal sequence of spatial landmarks

### ⚠️ Small Dataset Warning
With only ~10 videos per class, **overfitting is the primary challenge**. This document includes specific strategies for handling limited data.

### Realistic Expectations for GTE9 Dataset

**What to expect:**
- ✅ **55-70% accuracy** is achievable with proper augmentation and regularization
- ✅ Simple models (LSTM, GRU) will outperform complex ones
- ✅ 5-10x data augmentation is **essential**
- ⚠️ Training accuracy will be much higher than validation accuracy (normal!)
- ⚠️ Some classes may have 0% accuracy (not enough variation in 10 samples)

**What NOT to expect:**
- ❌ 90%+ accuracy (impossible with 10 samples/class)
- ❌ Complex models working better (they'll overfit)
- ❌ Training without augmentation being successful

**Action Plan:**
1. Start with **Lightweight Bi-LSTM** (provided below)
2. Implement **5x augmentation** (provided below)
3. Use **K-Fold Cross-Validation** (provided below)
4. Monitor train/val gap closely
5. If val accuracy < 50%, increase augmentation factor to 10x

---

## 1. LSTM-based Models

### Basic Bidirectional LSTM
**Best for**: Baseline model, quick experimentation

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_bilstm_model(num_classes=205):
    model = models.Sequential([
        layers.Input(shape=(70, 252)),  # 63*4 = 252 features
        
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.3),
        
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Usage
X_train = X_train.reshape(-1, 70, 252)  # Flatten landmarks
model = build_bilstm_model(num_classes=205)
```

**Expected Performance**: 60-75% accuracy (with proper regularization)  
**Training Time**: ~2-5 min/epoch (GPU)  
**Parameters**: ~500K  
**Overfitting Risk**: ⚠️ Medium - needs dropout and early stopping

---

### Stacked LSTM with Attention
**Best for**: Better temporal modeling, improved accuracy

```python
def build_lstm_attention_model(num_classes=205):
    inputs = layers.Input(shape=(70, 252))
    
    # Stacked LSTM layers
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(inputs)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    
    # Attention mechanism
    attention = layers.Dense(1, activation='tanh')(x)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(256)(attention)  # 256 = 2*128 (bidirectional)
    attention = layers.Permute([2, 1])(attention)
    
    # Apply attention
    x = layers.Multiply()([x, attention])
    x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(x)
    
    # Classification head
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

**Expected Performance**: 70-80% accuracy (if carefully regularized)  
**Training Time**: ~5-8 min/epoch (GPU)  
**Parameters**: ~2M  
**Overfitting Risk**: ⚠️⚠️ High - needs aggressive regularization

---

## 2. GRU-based Models

### Bidirectional GRU
**Best for**: Faster training than LSTM, similar performance

```python
def build_gru_model(num_classes=205):
    model = models.Sequential([
        layers.Input(shape=(70, 252)),
        
        layers.Bidirectional(layers.GRU(256, return_sequences=True)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Bidirectional(layers.GRU(128, return_sequences=True)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Bidirectional(layers.GRU(64)),
        layers.Dropout(0.3),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

**Expected Performance**: 68-78% accuracy  
**Training Time**: ~3-6 min/epoch (GPU)  
**Parameters**: ~1.5M  
**Overfitting Risk**: ⚠️ Medium - good balance for small datasets

---

## 3. 3D CNN Models

### Spatial-Temporal 3D CNN
**Best for**: Capturing spatial relationships between landmarks

```python
def build_3dcnn_model(num_classes=205):
    inputs = layers.Input(shape=(70, 63, 4, 1))  # Add channel dimension
    
    # First 3D Conv block
    x = layers.Conv3D(32, kernel_size=(3, 3, 2), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 1))(x)
    x = layers.Dropout(0.3)(x)
    
    # Second 3D Conv block
    x = layers.Conv3D(64, kernel_size=(3, 3, 2), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 1))(x)
    x = layers.Dropout(0.3)(x)
    
    # Third 3D Conv block
    x = layers.Conv3D(128, kernel_size=(3, 3, 2), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 1))(x)
    x = layers.Dropout(0.4)(x)
    
    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Usage
X_train = X_train[..., np.newaxis]  # Add channel dimension
```

**Expected Performance**: 65-75% accuracy (likely to overfit)  
**Training Time**: ~8-12 min/epoch (GPU)  
**Parameters**: ~3M  
**Overfitting Risk**: ⚠️⚠️⚠️ Very High - **NOT RECOMMENDED** for small dataset

---

## 4. Transformer-based Models

### Temporal Transformer
**Best for**: State-of-the-art performance, capturing long-range dependencies

```python
def build_transformer_model(num_classes=205, d_model=256, num_heads=8, num_layers=4):
    inputs = layers.Input(shape=(70, 252))
    
    # Positional encoding
    positions = tf.range(start=0, limit=70, delta=1)
    position_embedding = layers.Embedding(input_dim=70, output_dim=252)(positions)
    x = inputs + position_embedding
    
    # Project to d_model dimensions
    x = layers.Dense(d_model)(x)
    
    # Transformer encoder layers
    for _ in range(num_layers):
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=0.1
        )(x, x)
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        ffn = models.Sequential([
            layers.Dense(d_model * 4, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(d_model)
        ])
        ffn_output = ffn(x)
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification head
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

**Expected Performance**: 75-85% accuracy (needs extensive augmentation)  
**Training Time**: ~10-15 min/epoch (GPU)  
**Parameters**: ~5M  
**Overfitting Risk**: ⚠️⚠️⚠️ Very High - requires strong regularization & augmentation

---

## 5. Hybrid Models

### CNN-LSTM (ConvLSTM2D)
**Best for**: Combining spatial and temporal features

```python
def build_convlstm_model(num_classes=205):
    # Reshape to treat landmarks as spatial grid
    # (70, 63, 4) -> treat as (70, 9, 7, 4) or similar
    inputs = layers.Input(shape=(70, 9, 7, 4))
    
    # ConvLSTM layers
    x = layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=True,
        activation='relu'
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.ConvLSTM2D(
        filters=32,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=False,
        activation='relu'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Preprocessing for ConvLSTM
def reshape_for_convlstm(X):
    # Reshape (N, 70, 63, 4) -> (N, 70, 9, 7, 4)
    N = X.shape[0]
    return X.reshape(N, 70, 9, 7, 4)
```

**Expected Performance**: 72-80% accuracy (risk of overfitting)  
**Training Time**: ~12-18 min/epoch (GPU)  
**Parameters**: ~2.5M  
**Overfitting Risk**: ⚠️⚠️ High - complex architecture for small data

---

### 3D CNN + LSTM
**Best for**: Extract spatial features first, then model temporal dynamics

```python
def build_3dcnn_lstm_model(num_classes=205):
    inputs = layers.Input(shape=(70, 63, 4, 1))
    
    # 3D CNN feature extraction
    x = layers.Conv3D(32, kernel_size=(3, 3, 2), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling3D(pool_size=(2, 2, 1))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv3D(64, kernel_size=(3, 3, 2), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 1))(x)
    x = layers.Dropout(0.3)(x)
    
    # Reshape for LSTM
    # After pooling: (35, ~15, 4, 64) -> flatten spatial dims
    shape = x.shape
    x = layers.Reshape((shape[1], -1))(x)  # (35, features)
    
    # LSTM for temporal modeling
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.3)(x)
    
    # Classification
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

**Expected Performance**: 74-82% accuracy (likely overfits)  
**Training Time**: ~15-20 min/epoch (GPU)  
**Parameters**: ~4M  
**Overfitting Risk**: ⚠️⚠️⚠️ Very High - **NOT RECOMMENDED** for small dataset

---

## 6. Graph Neural Networks (Advanced)

### Spatial-Temporal Graph Convolutional Network (ST-GCN)
**Best for**: Modeling skeleton connections explicitly

```python
import tensorflow as tf
from tensorflow.keras import layers, models

class GraphConvolution(layers.Layer):
    def __init__(self, units, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel",
            shape=[input_shape[-1], self.units],
            initializer="glorot_uniform",
            trainable=True,
        )
        
    def call(self, inputs, adjacency):
        # inputs: (batch, nodes, features)
        # adjacency: (nodes, nodes)
        x = tf.matmul(inputs, self.kernel)
        x = tf.matmul(adjacency, x)
        return x

def create_adjacency_matrix():
    """Create adjacency matrix for body landmarks"""
    # Define connections between landmarks
    # 63 landmarks: pose (21) + left hand (21) + right hand (21)
    
    connections = [
        # Pose connections
        (0, 1), (0, 2),  # Nose to eyes
        (1, 3), (2, 4),  # Eyes to ears
        (0, 5), (0, 6),  # Nose to mouth
        (7, 8),          # Shoulders
        (7, 9), (9, 10), # Left arm
        (8, 11), (11, 12), # Right arm
        (7, 13), (8, 14),  # Torso
        (13, 14),        # Hips
        # Add hand connections (consecutive landmarks)
        *[(21+i, 21+i+1) for i in range(20)],  # Left hand
        *[(42+i, 42+i+1) for i in range(20)],  # Right hand
    ]
    
    adj = np.zeros((63, 63))
    for i, j in connections:
        adj[i, j] = 1
        adj[j, i] = 1  # Undirected
    
    # Add self-connections
    adj += np.eye(63)
    
    # Normalize
    degree = np.sum(adj, axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
    D_inv_sqrt = np.diag(degree_inv_sqrt)
    adj_normalized = D_inv_sqrt @ adj @ D_inv_sqrt
    
    return tf.constant(adj_normalized, dtype=tf.float32)

def build_stgcn_model(num_classes=205):
    inputs = layers.Input(shape=(70, 63, 4))
    
    adj_matrix = create_adjacency_matrix()
    
    # Spatial-temporal graph convolutions
    x = inputs
    
    for gcn_units in [64, 128, 256]:
        # Spatial graph convolution on each frame
        spatial_features = []
        for t in range(70):
            frame = x[:, t, :, :]  # (batch, 63, features)
            gcn_out = GraphConvolution(gcn_units)(frame, adj_matrix)
            gcn_out = layers.ReLU()(gcn_out)
            spatial_features.append(gcn_out)
        
        x = tf.stack(spatial_features, axis=1)  # (batch, 70, 63, gcn_units)
        x = layers.Dropout(0.3)(x)
        
        # Temporal convolution across frames
        x = layers.Reshape((70, -1))(x)  # (batch, 70, 63*gcn_units)
        x = layers.Conv1D(gcn_units, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.Reshape((70, 63, gcn_units))(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x[:, :, :, :])  # Pool over time and landmarks
    
    # Classification
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

**Expected Performance**: 78-88% accuracy (with massive augmentation)  
**Training Time**: ~20-30 min/epoch (GPU)  
**Parameters**: ~6M  
**Overfitting Risk**: ⚠️⚠️⚠️ Very High - only for advanced users with heavy augmentation

---

## 7. Ensemble Methods

### Simple Voting Ensemble
**Best for**: Maximum accuracy, production deployment

```python
def build_ensemble(num_classes=205):
    # Train multiple models
    lstm_model = build_bilstm_model(num_classes)
    transformer_model = build_transformer_model(num_classes)
    gru_model = build_gru_model(num_classes)
    
    return [lstm_model, transformer_model, gru_model]

def ensemble_predict(models, X_lstm, X_transformer):
    predictions = []
    
    # LSTM prediction
    predictions.append(models[0].predict(X_lstm))
    
    # Transformer prediction
    predictions.append(models[1].predict(X_transformer))
    
    # GRU prediction
    predictions.append(models[2].predict(X_lstm))
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return ensemble_pred
```

**Expected Performance**: 80-90% accuracy (if base models don't overfit)  
**Training Time**: Sum of individual models  
**Overfitting Risk**: ⚠️⚠️ High - only useful if individual models generalize  
**Note**: Requires training multiple models separately

---

## 🚨 Critical Strategies for Small Datasets (~10 samples/class)

### 1. Data Augmentation (ESSENTIAL)
With only 10 videos per class, augmentation is **mandatory**:

```python
import numpy as np

def augment_landmarks(X, augmentation_factor=5):
    """
    Generate multiple augmented versions of each video
    Target: 10 videos/class → 50+ videos/class
    """
    augmented_data = []
    
    for video in X:
        # Original
        augmented_data.append(video)
        
        for _ in range(augmentation_factor - 1):
            aug_video = video.copy()
            
            # 1. Random rotation (around z-axis, ±15 degrees)
            angle = np.random.uniform(-15, 15) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            aug_video[:, :, 0] = cos_a * video[:, :, 0] - sin_a * video[:, :, 1]
            aug_video[:, :, 1] = sin_a * video[:, :, 0] + cos_a * video[:, :, 1]
            
            # 2. Random scaling (±10%)
            scale = np.random.uniform(0.9, 1.1)
            aug_video[:, :, :3] *= scale
            
            # 3. Random translation
            shift_x = np.random.uniform(-0.05, 0.05)
            shift_y = np.random.uniform(-0.05, 0.05)
            aug_video[:, :, 0] += shift_x
            aug_video[:, :, 1] += shift_y
            
            # 4. Temporal jittering (small time shifts)
            if np.random.random() > 0.5:
                shift = np.random.randint(-3, 4)
                aug_video = np.roll(aug_video, shift, axis=0)
            
            # 5. Gaussian noise (small)
            noise = np.random.normal(0, 0.01, aug_video.shape)
            aug_video[:, :, :3] += noise[:, :, :3]
            
            # 6. Random frame dropout (simulate detection failures)
            if np.random.random() > 0.7:
                num_frames_to_drop = np.random.randint(1, 5)
                frames_to_drop = np.random.choice(70, num_frames_to_drop, replace=False)
                aug_video[frames_to_drop] = 0.0
            
            augmented_data.append(aug_video)
    
    return np.array(augmented_data)

# Usage
X_train_augmented = augment_landmarks(X_train, augmentation_factor=5)
# Now you have 5x more training data
```

### 2. Reduced Model Complexity
**Use simpler models** to avoid overfitting:

```python
# RECOMMENDED: Lightweight LSTM
def build_small_lstm(num_classes=205):
    model = models.Sequential([
        layers.Input(shape=(70, 252)),
        
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.5),  # Higher dropout!
        
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.5),
        
        layers.Dense(64, activation='relu', 
                    kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dropout(0.5),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

### 3. Cross-Validation (CRITICAL)
With limited data, use **K-Fold Cross-Validation** instead of single train/test split:

```python
from sklearn.model_selection import StratifiedKFold

def cross_validate_model(X, y, n_splits=5):
    """
    5-fold CV gives you 80% train (8 videos) / 20% val (2 videos) per class
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, np.argmax(y, axis=1))):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Augment training data
        X_train_aug = augment_landmarks(X_train, augmentation_factor=5)
        y_train_aug = np.repeat(y_train, 5, axis=0)
        
        # Build fresh model
        model = build_small_lstm(num_classes=205)
        
        # Train
        history = model.fit(
            X_train_aug, y_train_aug,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ],
            verbose=1
        )
        
        # Evaluate
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        fold_scores.append(val_acc)
        print(f"Fold {fold + 1} Validation Accuracy: {val_acc:.4f}")
    
    print(f"\n=== Average CV Accuracy: {np.mean(fold_scores):.4f} ±{np.std(fold_scores):.4f} ===")
    return fold_scores
```

### 4. Aggressive Regularization
```python
# Increase dropout rates
layers.Dropout(0.5)  # Instead of 0.3

# Add L2 regularization
layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.01))

# Reduce model size
# Use 64-128 units instead of 256-512

# Batch normalization
layers.BatchNormalization()
```

### 5. Transfer Learning (Advanced)
Pre-train on larger dataset (full WLASL), then fine-tune:

```python
# 1. Pre-train on full WLASL dataset (all classes)
base_model = build_small_lstm(num_classes=2000)  # All WLASL words
base_model.fit(X_full_wlasl, y_full_wlasl, epochs=20)

# 2. Remove last layer
base_model.pop()  # Remove softmax layer

# 3. Freeze early layers
for layer in base_model.layers[:-2]:
    layer.trainable = False

# 4. Add new classification head for GTE9
x = base_model.output
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(205, activation='softmax')(x)

# 5. Fine-tune on GTE9
model = models.Model(inputs=base_model.input, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_gte9_train, y_gte9_train, epochs=30)
```

### 6. Leave-One-Out Cross-Validation
For extremely small datasets, consider LOOCV:

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
for train_idx, test_idx in loo.split(X):
    # Train on N-1 samples, test on 1
    pass
```

---

## Training Recommendations

### Data Preprocessing
```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load data
X = np.load('gte9_landmarks/x.npy')  # Shape: (N, 70, 63, 4)
y = np.load('gte9_landmarks/y.npy')  # Shape: (N,)

# Handle missing data (zeros)
# Option 1: Mask zeros
# Option 2: Interpolate
# Option 3: Use as-is (let model learn)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded, num_classes=205)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, stratify=y_encoded, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, stratify=np.argmax(y_train, axis=1), random_state=42
)
```

### Callbacks
```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7
    ),
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True
    )
]
```

### Data Augmentation
```python
def augment_landmarks(X):
    """Apply random transformations"""
    # Random rotation (small angles)
    # Random scaling (slight zoom)
    # Random time warping
    # Random temporal dropout (mask some frames)
    return X_augmented
```

---

## Model Selection Guide

### For Small Datasets (~10 samples/class) - GTE9

| Priority | Model Type | Recommended Model | Why |
|----------|-----------|-------------------|-----|
| **🏆 BEST START** | RNN | **Lightweight Bi-LSTM** | Simple, less overfitting, proven for small data |
| **Runner Up** | RNN | **Bi-GRU** | Faster, fewer parameters than LSTM |
| **With Augmentation** | RNN | LSTM + Attention (reduced size) | If you have 5x augmentation |
| **Advanced** | Transformer | Small Transformer (2 layers) | Only with heavy augmentation & regularization |
| **❌ AVOID** | CNN | 3D CNN variants | Too many parameters, will overfit |
| **❌ AVOID** | GNN | ST-GCN | Too complex for limited data |
| **❌ AVOID** | Hybrid | CNN-LSTM hybrids | High overfitting risk |

### For Larger Datasets (when you get more data)

| Priority | Model Type | Recommended Model | Why |
|----------|-----------|-------------------|-----|
| **Best Balance** | Hybrid | 3D CNN + LSTM | Good accuracy, reasonable training time |
| **Highest Accuracy** | Transformer or GNN | Transformer or ST-GCN | State-of-the-art performance |
| **Best for Skeleton Data** | GNN | ST-GCN | Explicitly models body structure |
| **Production** | Ensemble | LSTM + Transformer + GRU | Maximum reliability |

---

## Hyperparameter Tuning Tips

### Learning Rate Schedule
```python
def cosine_decay_schedule(epoch, lr):
    max_epochs = 100
    return 0.001 * 0.5 * (1 + np.cos(np.pi * epoch / max_epochs))

from tensorflow.keras.callbacks import LearningRateScheduler
lr_scheduler = LearningRateScheduler(cosine_decay_schedule)
```

### Batch Size Recommendations
- **Small models (LSTM, GRU)**: 32-64
- **Medium models (Transformer)**: 16-32
- **Large models (3D CNN, GNN)**: 8-16

### Regularization
- **Dropout**: 0.3-0.5 for dense layers, 0.2-0.4 for recurrent layers
- **L2 regularization**: 0.0001-0.001
- **Batch normalization**: After conv/dense layers

---

## Expected Results Summary

### With Small Dataset (GTE9: ~10 videos/class)

| Model | Accuracy (No Aug) | Accuracy (With Aug) | Overfitting Risk | Recommendation |
|-------|-------------------|---------------------|------------------|----------------|
| Lightweight Bi-LSTM | 40-55% | **55-70%** | 🟡 Medium | ✅ **START HERE** |
| Bi-GRU | 40-55% | **55-68%** | 🟡 Medium | ✅ Good alternative |
| LSTM + Attention | 35-50% | 50-65% | 🔴 High | ⚠️ Use with caution |
| 3D CNN | 25-40% | 35-50% | 🔴 Very High | ❌ Not recommended |
| Transformer | 30-45% | 45-60% | 🔴 Very High | ⚠️ Only if experienced |
| 3D CNN + LSTM | 20-35% | 30-45% | 🔴 Very High | ❌ Will overfit badly |
| ST-GCN | 25-40% | 40-55% | 🔴 Very High | ❌ Too complex |
| Ensemble | N/A | 50-65% | 🔴 High | ⚠️ Only if base models work |

**Key Insights for Small Data:**
- Without augmentation: Expect 40-55% accuracy (random chance = 0.5%)
- With 5x augmentation: Can reach 55-70% with simple models
- Complex models will memorize training data and fail on validation
- **Focus on data augmentation and regularization, not model complexity**

### With Larger Dataset (500+ samples/class)

| Model | Accuracy | Speed | Memory | Difficulty |
|-------|----------|-------|--------|-----------|
| Bi-LSTM | 65-75% | Fast | Low | Easy |
| Bi-GRU | 68-78% | Fast | Low | Easy |
| LSTM + Attention | 70-80% | Medium | Medium | Medium |
| 3D CNN | 65-75% | Slow | High | Medium |
| Transformer | 75-85% | Medium | Medium | Hard |
| 3D CNN + LSTM | 74-82% | Slow | High | Hard |
| ST-GCN | 78-88% | Slow | High | Hard |
| Ensemble | 80-90% | Very Slow | High | Hard |

---

## Next Steps

1. **Start with Bi-LSTM** to establish baseline
2. **Try GRU** for faster iteration
3. **Add attention** to improve performance
4. **Experiment with Transformer** for best results
5. **Consider ST-GCN** if you want to leverage skeleton structure
6. **Build ensemble** for production deployment

Good luck with your ASL recognition model! 🤟
