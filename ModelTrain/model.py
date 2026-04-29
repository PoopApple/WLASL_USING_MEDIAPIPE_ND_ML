"""
ASL recognition model zoo.

All models take the same input:
    input_data:  (batch, 128, 64, 4)
    input_mask:  (batch, 128) bool

Models are ordered from simplest to most complex:

    1. original     — Your proven BiGRU (80% baseline)
    2. bigru_v2     — BiGRU + LayerNorm + wider head
    3. conv_bigru   — Conv1D local patterns → BiGRU global context
    4. tcn          — Temporal Convolutional Network (dilated convolutions)
    5. conv1d       — Pure Conv1D (fastest, lightweight)
"""

import os
import datetime
import tensorflow as tf

MAX_FRAMES = 128


# ══════════════════════════════════════════════════════════════════════
# 1. ORIGINAL — your proven 80% BiGRU
# ══════════════════════════════════════════════════════════════════════

def build_original(num_classes: int) -> tf.keras.Model:
    """Exact architecture from your old train.py that got 80%."""
    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")

    x = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(input_data)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(256, return_sequences=True)
    )(x, mask=input_mask)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(128)
    )(x, mask=input_mask)

    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(
        inputs=[input_data, input_mask], outputs=outputs, name="original"
    )


# ══════════════════════════════════════════════════════════════════════
# 2. BiGRU V2 — conservative upgrade (LayerNorm + wider head)
# ══════════════════════════════════════════════════════════════════════

def build_bigru_v2(num_classes: int) -> tf.keras.Model:
    """
    Same backbone, adds:
      - LayerNorm after each BiGRU (stabilises gradients)
      - Wider classification head (256 → 128 → output)
      - Staged dropout
    """
    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")

    x = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(input_data)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(256, return_sequences=True)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(128)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(
        inputs=[input_data, input_mask], outputs=outputs, name="bigru_v2"
    )


def build_bigru_v3(num_classes: int) -> tf.keras.Model:
    """
    Upgraded BiGRU to push past 80% accuracy.
    Improvements:
    1. Velocity Features: Calculates the frame-to-frame difference (motion) directly 
       in the graph and concatenates it with raw positions. The GRU no longer has 
       to 'guess' the velocity.
    2. Spatial Dropout: Drops entire coordinate channels across all 128 frames. 
       Forces the model to look at the whole body, not just the dominant hand.
    """
    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")

    # Calculate Velocity (frame[t] - frame[t-1])
    def calc_velocity(x):
        diff = x[:, 1:, :, :] - x[:, :-1, :, :]
        zeros = tf.zeros_like(x[:, 0:1, :, :])
        return tf.concat([zeros, diff], axis=1)
        
    velocity = tf.keras.layers.Lambda(calc_velocity)(input_data)
    
    # Concatenate position (4) and velocity (4) -> 8 channels per landmark
    combined = tf.keras.layers.Concatenate(axis=-1)([input_data, velocity])
    x = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 8))(combined)

    # SpatialDropout drops entire feature dimensions across all time steps
    x = tf.keras.layers.SpatialDropout1D(0.2)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(256, return_sequences=True)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(128)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(
        inputs=[input_data, input_mask], outputs=outputs, name="bigru_v3"
    )

def build_bigru_bigger_v1(num_classes: int) -> tf.keras.Model:
    """
    Scaled-up version of bigru_v2 for 1000+ words.
    Uses 512 and 256 GRU units instead of 256 and 128.
    Heavy dropout added to prevent overfitting on the long tail of rare words.
    """
    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")

    x = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(input_data)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(512, return_sequences=True)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)  # Extra dropout for capacity control

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(256)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(
        inputs=[input_data, input_mask], outputs=outputs, name="bigru_bigger_v1"
    )


def build_bigru_bigger_v2(num_classes: int) -> tf.keras.Model:
    """
    The ultimate scaled-up model: bigru_bigger_v2.
    Extremely high capacity designed for 2000 words.
    Uses 3 stacked BiGRUs (512 -> 512 -> 256) and larger dense layers.
    Includes SpatialDropout1D to prevent overfitting.
    """
    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")

    x = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(input_data)

    # Spatial Dropout to force holistic learning
    x = tf.keras.layers.SpatialDropout1D(0.2)(x)

    # Layer 1 (Massive Width)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(1024, return_sequences=True)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Layer 2
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(512)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(
        inputs=[input_data, input_mask], outputs=outputs, name="bigru_bigger_v2"
    )


# ══════════════════════════════════════════════════════════════════════
# 3. CONV + BiGRU — local temporal features → global sequence context
# ══════════════════════════════════════════════════════════════════════

def build_conv_bigru(num_classes: int) -> tf.keras.Model:
    """
    WHY: Conv1D captures local motion patterns (hand moving over 3-5 frames)
    that GRU alone might miss. The GRU then reads these local features
    and builds a global understanding of the full sign.

    Architecture:
        Conv1D(128, k=3) → Conv1D(128, k=3) → BiGRU(128) → Dense head

    The conv layers act as a learned feature extractor per-timestep,
    reducing the 256-dim raw features to 128-dim motion-aware features.
    """
    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")

    x = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(input_data)

    # ── Local temporal feature extraction ──
    # kernel_size=3 captures patterns over 3 consecutive frames
    # "same" padding keeps sequence length at 128
    x = tf.keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # ── Global temporal context ──
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(128)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)

    # ── Classification head ──
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(
        inputs=[input_data, input_mask], outputs=outputs, name="conv_bigru"
    )


# ══════════════════════════════════════════════════════════════════════
# 4. TCN — Temporal Convolutional Network (dilated convolutions)
# ══════════════════════════════════════════════════════════════════════

def _tcn_block(x, filters: int, kernel_size: int, dilation_rate: int):
    """Single TCN residual block with dilated convolution."""
    residual = x

    # Dilated causal conv → BatchNorm → ReLU → Dropout
    out = tf.keras.layers.Conv1D(
        filters, kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu",
    )(x)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Dropout(0.2)(out)

    out = tf.keras.layers.Conv1D(
        filters, kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu",
    )(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Dropout(0.2)(out)

    # Residual connection (project if channel mismatch)
    if residual.shape[-1] != filters:
        residual = tf.keras.layers.Conv1D(filters, 1, padding="same")(residual)

    return tf.keras.layers.Add()([residual, out])


def build_tcn(num_classes: int) -> tf.keras.Model:
    """
    WHY: TCN uses dilated convolutions to get exponentially growing
    receptive fields. With dilations [1, 2, 4, 8], a kernel_size=3 conv
    sees 128 frames — the entire sequence — but with far fewer parameters
    than an RNN and much faster to train.

    Very popular for skeleton-based action recognition.

    Architecture:
        4 TCN blocks (dilation 1,2,4,8) → GlobalAvgPool → Dense head
    """
    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")

    x = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(input_data)

    # ── TCN blocks with exponentially growing receptive field ──
    # dilation 1: sees 3 frames
    # dilation 2: sees 7 frames
    # dilation 4: sees 15 frames
    # dilation 8: sees 31 frames
    # Combined: sees 128+ frames (full sequence coverage)
    x = _tcn_block(x, filters=128, kernel_size=3, dilation_rate=1)
    x = _tcn_block(x, filters=128, kernel_size=3, dilation_rate=2)
    x = _tcn_block(x, filters=128, kernel_size=3, dilation_rate=4)
    x = _tcn_block(x, filters=128, kernel_size=3, dilation_rate=8)

    # ── Masked global average pooling ──
    # Zero out padded frames before averaging
    float_mask = tf.keras.layers.Lambda(
        lambda m: tf.cast(m, tf.float32)
    )(input_mask)  # (batch, 128)
    mask_expanded = tf.keras.layers.Lambda(
        lambda m: tf.expand_dims(m, -1)
    )(float_mask)  # (batch, 128, 1)

    x = tf.keras.layers.Multiply()([x, mask_expanded])

    # Sum then divide by number of valid frames
    x_sum = tf.keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)
    mask_count = tf.keras.layers.Lambda(
        lambda m: tf.maximum(tf.reduce_sum(m, axis=1, keepdims=True), 1.0)
    )(float_mask)
    x = tf.keras.layers.Lambda(lambda args: args[0] / args[1])([x_sum, mask_count])

    # ── Classification head ──
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(
        inputs=[input_data, input_mask], outputs=outputs, name="tcn"
    )


# ══════════════════════════════════════════════════════════════════════
# 5. PURE CONV1D — lightweight and fast
# ══════════════════════════════════════════════════════════════════════

def build_conv1d(num_classes: int) -> tf.keras.Model:
    """
    WHY: Simplest possible deep model. Stacked Conv1D layers with
    max pooling to progressively shrink the temporal dimension.
    Fastest to train, good sanity check baseline.

    Architecture:
        Conv1D(128) → MaxPool → Conv1D(128) → MaxPool → Conv1D(64)
        → Masked GlobalAvgPool → Dense head
    """
    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")

    x = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(input_data)

    x = tf.keras.layers.Conv1D(128, kernel_size=5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)  # 128 → 64

    x = tf.keras.layers.Conv1D(128, kernel_size=5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)  # 64 → 32

    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # ── Masked global average pooling ──
    # Downsample the mask to match pooled sequence length (128 → 64 → 32)
    float_mask = tf.keras.layers.Lambda(
        lambda m: tf.cast(m, tf.float32)
    )(input_mask)  # (batch, 128)
    # Pool mask same as data: take every 4th value (2 maxpools of size 2)
    downsampled_mask = tf.keras.layers.Lambda(
        lambda m: m[:, ::4]
    )(float_mask)  # (batch, 32)
    mask_expanded = tf.keras.layers.Lambda(
        lambda m: tf.expand_dims(m, -1)
    )(downsampled_mask)  # (batch, 32, 1)

    x = tf.keras.layers.Multiply()([x, mask_expanded])
    x_sum = tf.keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)
    mask_count = tf.keras.layers.Lambda(
        lambda m: tf.maximum(tf.reduce_sum(m, axis=1, keepdims=True), 1.0)
    )(downsampled_mask)
    x = tf.keras.layers.Lambda(lambda args: args[0] / args[1])([x_sum, mask_count])

    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(
        inputs=[input_data, input_mask], outputs=outputs, name="conv1d"
    )


# ══════════════════════════════════════════════════════════════════════
# 6. DUAL-PATH — Conv1D motion features + raw data → BiGRU
# ══════════════════════════════════════════════════════════════════════

def build_dualpath(num_classes: int) -> tf.keras.Model:
    """
    Two-stream architecture:

    Stream 1 (Motion):  Conv1D stack extracts LOCAL motion patterns
                        (velocity, acceleration, micro-gestures over 3-5 frames)

    Stream 2 (Raw):     Raw landmark positions pass through unchanged
                        (WHERE things are — absolute positions)

    Both streams are concatenated at each timestep, giving the BiGRU
    a rich representation: it sees both "where the hands are" AND
    "how the hands are moving" at every frame.

    Architecture:
        raw (256-d) ──────────────────────────┐
            │                                  │
            ├→ Conv1D(128, k=3) → BN           │
            ├→ Conv1D(128, k=5) → BN           │ (raw positions)
            ├→ Conv1D(64, k=3)  → BN           │
            │    = motion features (64-d)       │
            │                                  │
            └──── Concatenate ─────────────────┘
                        │
                   (256 + 64 = 320-d per frame)
                        │
                   BiGRU(128) → LN
                        │
                   Dense head → output
    """
    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")

    raw = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(input_data)  # (128, 256)

    # ── Stream 1: Conv1D motion feature extractor ──
    # Progressively extracts motion patterns at different scales
    m = tf.keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(raw)
    m = tf.keras.layers.BatchNormalization()(m)
    m = tf.keras.layers.Conv1D(128, kernel_size=5, padding="same", activation="relu")(m)
    m = tf.keras.layers.BatchNormalization()(m)
    m = tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(m)
    m = tf.keras.layers.BatchNormalization()(m)
    m = tf.keras.layers.Dropout(0.2)(m)  # (128, 64) motion features

    # ── Merge: concatenate raw positions + motion features ──
    merged = tf.keras.layers.Concatenate()([raw, m])  # (128, 256 + 64 = 320)

    # ── BiGRU reads the enriched sequence ──
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(128)
    )(merged, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)

    # ── Classification head ──
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(
        inputs=[input_data, input_mask], outputs=outputs, name="dualpath"
    )


# ══════════════════════════════════════════════════════════════════════
# 7. DUAL-PATH V2 — Conv1D motion + raw → deep BiGRU (merged from
#    dualpath input stream + bigru_v2 backbone)
# ══════════════════════════════════════════════════════════════════════

def build_dualpath_v2(num_classes: int) -> tf.keras.Model:
    """
    Merges the best of dualpath + bigru_v2:

    FROM dualpath:   Two-stream input (Conv1D motion features ∥ raw positions)
    FROM bigru_v2:   Deep 2-layer BiGRU with LayerNorm + wider head

    Architecture:
        raw (256-d) ──────────────────────────┐
            │                                  │
            ├→ Conv1D(128, k=3) → BN           │
            ├→ Conv1D(128, k=5) → BN           │ (raw positions)
            ├→ Conv1D(64, k=3)  → BN           │
            │    = motion features (64-d)       │
            │                                  │
            └──── Concatenate ─────────────────┘
                        │
                   (256 + 64 = 320-d per frame)
                        │
                   BiGRU(256) → LayerNorm      ← from bigru_v2
                        │
                   BiGRU(128) → LayerNorm      ← from bigru_v2
                        │
                   Dropout(0.4) → Dense(256)   ← wider head from bigru_v2
                   Dropout(0.3) → Dense(128)
                        │
                      output
    """
    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")

    raw = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(input_data)  # (128, 256)

    # ── Stream 1: Conv1D motion feature extractor ──
    m = tf.keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(raw)
    m = tf.keras.layers.BatchNormalization()(m)
    m = tf.keras.layers.Conv1D(128, kernel_size=5, padding="same", activation="relu")(m)
    m = tf.keras.layers.BatchNormalization()(m)
    m = tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(m)
    m = tf.keras.layers.BatchNormalization()(m)
    m = tf.keras.layers.Dropout(0.2)(m)  # (128, 64)

    # ── Merge streams ──
    merged = tf.keras.layers.Concatenate()([raw, m])  # (128, 320)

    # ── Deep BiGRU backbone (from bigru_v2) ──
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(256, return_sequences=True)
    )(merged, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(128)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)

    # ── Wider classification head (from bigru_v2) ──
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(
        inputs=[input_data, input_mask], outputs=outputs, name="dualpath_v2"
    )


# ══════════════════════════════════════════════════════════════════════
# Model registry — for easy switching in train_v2.py
# ══════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "original": build_original,
    "bigru_v2": build_bigru_v2,
    "bigru_v3": build_bigru_v3,
    "bigru_bigger_v1": build_bigru_bigger_v1,
    "bigru_bigger_v2": build_bigru_bigger_v2,
    "conv_bigru": build_conv_bigru,
    "tcn": build_tcn,
    "conv1d": build_conv1d,
    "dualpath": build_dualpath,
    "dualpath_v2": build_dualpath_v2,
}


def build_model(name: str, num_classes: int) -> tf.keras.Model:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](num_classes)


# ══════════════════════════════════════════════════════════════════════
# Callbacks
# ══════════════════════════════════════════════════════════════════════

def get_callbacks(model_dir: str, model_name: str | None = None, patience: int = 15):
    if model_name is None:
        model_name = datetime.datetime.now().strftime("%d-%m-%y__%H-%M")

    os.makedirs(model_dir, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, f"asl_{model_name}_best.keras"),
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=patience,
            restore_best_weights=True,
            monitor="val_loss",
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(model_dir, f"asl_{model_name}_log.csv"),
            separator=",",
            append=False,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
