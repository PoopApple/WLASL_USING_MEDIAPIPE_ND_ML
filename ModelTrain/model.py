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

MAX_FRAMES = 64   # 95.4% of trimmed segments fit in 64f; median=37f (was 128)


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


@tf.keras.utils.register_keras_serializable(package="asl")
def calc_velocity(x):
    diff = x[:, 1:, :, :] - x[:, :-1, :, :]
    zeros = tf.zeros_like(x[:, 0:1, :, :])
    return tf.concat([zeros, diff], axis=1)



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

@tf.keras.utils.register_keras_serializable(package="asl")
def compute_kinematic_features(x):
    # x: (batch, 128, 64, 4)

    # ---- remove visibility & padding (keep only the 63 used landmarks) ----
    x = x[:, :, :63, :3]   # (batch, 128, 63, 3)

    # ---- split ----
    body = x[:, :, :21, :]
    left = x[:, :, 21:42, :]
    right = x[:, :, 42:63, :]


    def angle(a, b, c):
        ba = a - b
        bc = c - b

        # Use a larger epsilon to prevent exploding gradients when vectors are near zero
        # Gradient of l2_normalize(x) is unstable for small x
        ba = ba / tf.maximum(tf.norm(ba, axis=-1, keepdims=True), 1e-4)
        bc = bc / tf.maximum(tf.norm(bc, axis=-1, keepdims=True), 1e-4)

        cos = tf.reduce_sum(ba * bc, axis=-1)
        # Tighter clipping to avoid infinite gradients in acos derivative
        # d/dx acos(x) = -1 / sqrt(1 - x^2)
        # If x=0.999, d/dx is ~22. If x=0.99999, d/dx is ~223.
        cos = tf.clip_by_value(cos, -0.999, 0.999) 

        return tf.acos(cos)

    def bone(p1, p2):
        v = p2 - p1
        # Safer normalization for bone directions
        return v / tf.maximum(tf.norm(v, axis=-1, keepdims=True), 1e-4)

    # =========================
    # BODY
    # =========================

    L_SHOULDER = 7
    R_SHOULDER = 8
    L_ELBOW = 9
    R_ELBOW = 10
    L_WRIST = 11
    R_WRIST = 12
    L_HIP = 19
    R_HIP = 20

    left_elbow = angle(body[:, :, L_SHOULDER], body[:, :, L_ELBOW], body[:, :, L_WRIST])
    right_elbow = angle(body[:, :, R_SHOULDER], body[:, :, R_ELBOW], body[:, :, R_WRIST])

    left_shoulder = angle(body[:, :, L_HIP], body[:, :, L_SHOULDER], body[:, :, L_ELBOW])
    right_shoulder = angle(body[:, :, R_HIP], body[:, :, R_SHOULDER], body[:, :, R_ELBOW])

    # =========================
    # HANDS
    # =========================

    def hand_angles(hand):
        fingers = [
            (1,2,3,4),
            (5,6,7,8),
            (9,10,11,12),
            (13,14,15,16),
            (17,18,19,20)
        ]

        out = []

        wrist = hand[:, :, 0]

        for mcp, pip, dip, tip in fingers:
            # internal bends
            out.append(angle(hand[:, :, mcp], hand[:, :, pip], hand[:, :, dip]))
            out.append(angle(hand[:, :, pip], hand[:, :, dip], hand[:, :, tip]))

            # global orientation
            out.append(angle(wrist, hand[:, :, mcp], hand[:, :, tip]))

        return tf.stack(out, axis=-1)
    
    left_hand_angles = hand_angles(left)
    right_hand_angles = hand_angles(right)

    # =========================
    # BONE DIRECTIONS
    # =========================

    left_upper_arm = bone(body[:, :, L_SHOULDER], body[:, :, L_ELBOW])
    left_forearm = bone(body[:, :, L_ELBOW], body[:, :, L_WRIST])

    right_upper_arm = bone(body[:, :, R_SHOULDER], body[:, :, R_ELBOW])
    right_forearm = bone(body[:, :, R_ELBOW], body[:, :, R_WRIST])

    # =========================
    # CONCAT
    # =========================

    features = tf.concat([
        tf.stack([left_elbow, right_elbow, left_shoulder, right_shoulder], axis=-1),
        left_hand_angles,
        right_hand_angles,
        left_upper_arm,     # 3 components (x, y, z)
        left_forearm,       # 3 components 
        right_upper_arm,    # 3 components
        right_forearm       # 3 components
    ], axis=-1)

    return features

# ══════════════════════════════════════════════════════════════════════
# bigru_angular (PURE ANGLES ONLY)
# ══════════════════════════════════════════════════════════════════════

def build_bigru_angular(num_classes: int) -> tf.keras.Model:
    """
    Theoretical test model that completely discards raw coordinates and 
    relies 100% on the mathematically computed 46 kinematic angles.
    """
    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")
    
    # 1. Extract exactly 46 angles and combine with raw coordinates
    angles = KinematicFeatureLayer()(input_data)
    x_raw = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(input_data)
    
    x = tf.keras.layers.Concatenate(axis=-1)([x_raw, angles])
    
    # 2. BiGRU Funnel Stack
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(512, return_sequences=True)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(256)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # 4. Dense Classification Head
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(
        inputs=[input_data, input_mask], outputs=outputs, name="bigru_angular"
    )


def compute_kinematic_features_global(x):
    return compute_kinematic_features(x)

def build_bigru_angular_v1(num_classes: int) -> tf.keras.Model:
    """
    BiGRU with angle/distance constraints, scaled down for 500 words.
    Uses 256 and 128 GRU units (similar to bigru_v2 capacity).
    """
    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")
    
    spatial_features = tf.keras.layers.Lambda(
        compute_kinematic_features,
        output_shape=(MAX_FRAMES, 46)
    )(input_data)
    
    # Base flattened raw coords
    x_raw = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(input_data)

    # Flattened combined with newly computed angles and distances
    x = tf.keras.layers.Concatenate(axis=-1)([x_raw, spatial_features])

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(256, return_sequences=True)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x) 

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
        inputs=[input_data, input_mask], outputs=outputs, name="bigru_angular_v1"
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

def build_bigru_bigger_angular_v1(num_classes: int) -> tf.keras.Model:
    """
    Scaled-up version of bigru_v2 for 1000+ words.
    Uses 512 and 256 GRU units instead of 256 and 128.
    Heavy dropout added to prevent overfitting on the long tail of rare words.
    Adds spatial tracking features: Angle constraints combining left & right sides.
    """
    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")
    
    spatial_features = tf.keras.layers.Lambda(
        compute_kinematic_features,
        output_shape=(MAX_FRAMES, 46)
    )(input_data)
    
    # Base flattened raw coords
    x_raw = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(input_data)

    # Flattened combined with newly computed angles and distances
    x = tf.keras.layers.Concatenate(axis=-1)([x_raw, spatial_features])

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
        inputs=[input_data, input_mask], outputs=outputs, name="bigru_bigger_angular_v1"
    )


# ══════════════════════════════════════════════════════════════════════
# FIXED KINEMATIC LAYER — proper Keras Layer instead of Lambda
# Fixes: acos clip tightened to ±0.99, proper serialization, no Lambda trace issues
# ══════════════════════════════════════════════════════════════════════

@tf.keras.utils.register_keras_serializable(package="asl")
class KinematicFeatureLayer(tf.keras.layers.Layer):
    """
    Computes kinematic features (joint angles + bone directions) as a proper
    Keras Layer instead of Lambda, fixing gradient tracing and serialization.

    Output shape: (batch, T, 46)
      - 4 elbow/shoulder angles
      - 15 left hand angles
      - 15 right hand angles
      - 4×3 = 12 bone direction components (left/right upper arm + forearm)
    """
    def call(self, x):
        x = x[:, :, :63, :3]                  # (batch, T, 63, 3)
        body  = x[:, :, :21, :]
        left  = x[:, :, 21:42, :]
        right = x[:, :, 42:63, :]

        def _safe_norm(v):
            norm_sq = tf.reduce_sum(v * v, axis=-1, keepdims=True)
            # Use tf.maximum to strictly enforce a lower bound without distorting real values!
            # If norm_sq < 1e-7 (like padded zero-vectors), the gradient stops here (becomes 0).
            # This prevents both NaN gradients and global clipnorm explosion, while
            # keeping real vectors mathematically perfect unit vectors.
            safe_norm_sq = tf.maximum(norm_sq, 1e-7)
            return v / tf.sqrt(safe_norm_sq)

        def _angle(a, b, c):
            ba = _safe_norm(a - b)
            bc = _safe_norm(c - b)
            cos = tf.reduce_sum(ba * bc, axis=-1)
            cos = tf.clip_by_value(cos, -0.99, 0.99)
            return tf.acos(cos)

        def _bone(p1, p2):
            return _safe_norm(p2 - p1)

        # Body angles
        L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 7, 8, 9, 10, 11, 12, 19, 20
        body_angles = tf.stack([
            _angle(body[:,:,L_HIP],  body[:,:,L_SH], body[:,:,L_EL]),
            _angle(body[:,:,R_HIP],  body[:,:,R_SH], body[:,:,R_EL]),
            _angle(body[:,:,L_SH],   body[:,:,L_EL], body[:,:,L_WR]),
            _angle(body[:,:,R_SH],   body[:,:,R_EL], body[:,:,R_WR]),
        ], axis=-1)   # (batch, T, 4)

        # Hand angles (15 per hand: 2 bends + 1 global per finger)
        def _hand_angles(hand):
            fingers = [(1,2,3,4),(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20)]
            wrist = hand[:, :, 0]
            out = []
            for mcp, pip, dip, tip in fingers:
                out.append(_angle(hand[:,:,mcp], hand[:,:,pip], hand[:,:,dip]))
                out.append(_angle(hand[:,:,pip], hand[:,:,dip], hand[:,:,tip]))
                out.append(_angle(wrist,         hand[:,:,mcp], hand[:,:,tip]))
            return tf.stack(out, axis=-1)   # (batch, T, 15)

        # Bone directions (3 components each → 12 total)
        bones = tf.concat([
            _bone(body[:,:,L_SH], body[:,:,L_EL]),
            _bone(body[:,:,L_EL], body[:,:,L_WR]),
            _bone(body[:,:,R_SH], body[:,:,R_EL]),
            _bone(body[:,:,R_EL], body[:,:,R_WR]),
        ], axis=-1)   # (batch, T, 12)

        return tf.concat([body_angles, _hand_angles(left), _hand_angles(right), bones], axis=-1)
        # → (batch, T, 4 + 15 + 15 + 12) = (batch, T, 46)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 46)


@tf.keras.utils.register_keras_serializable(package="asl")
def compute_angular_velocity(angles):
    """Frame-to-frame angular velocity. Same shape as input, first frame is zeros."""
    diff  = angles[:, 1:, :] - angles[:, :-1, :]
    zeros = tf.zeros_like(angles[:, :1, :])
    return tf.concat([zeros, diff], axis=1)

@tf.keras.utils.register_keras_serializable(package="asl")
def compute_velocity(x):
    """Frame-to-frame raw coordinate velocity."""
    diff  = x[:, 1:, :, :] - x[:, :-1, :, :]
    zeros = tf.zeros_like(x[:, :1, :, :])
    return tf.concat([zeros, diff], axis=1)

# ══════════════════════════════════════════════════════════════════════
# bigru_velocity_v1  (500 words)
#   Raw coordinates + Raw velocity (NO Conv1D, NO BatchNorm)
#   Exact clone of the proven bigru_bigger_v1 backbone
# ══════════════════════════════════════════════════════════════════════

def build_bigru_velocity_v1(num_classes: int) -> tf.keras.Model:
    """
    Minimal upgrade over bigru_bigger_v1 (the 88% champion):
    - Raw coordinates  (T, 256)   — WHERE hands are
    - Raw velocity     (T, 256)   — HOW hands move (simple frame diffs)
    - NO Conv1D, NO BatchNorm, NO angles, NO SpatialDropout
    - Same 512→256 BiGRU backbone that hit 88% val accuracy
    
    Input to GRU: raw(256) + velocity(256) = 512-d
    """
    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")

    # ── Position: where the landmarks are ──
    x_raw = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(input_data)       # (T, 256)

    # ── Velocity: how the landmarks move (simple frame diffs, no processing) ──
    velocity = tf.keras.layers.Lambda(compute_velocity)(input_data)          # (T, 64, 4)
    x_vel = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(velocity)         # (T, 256)

    # ── Merge: just concatenate, let the BiGRU figure it out ──
    x = tf.keras.layers.Concatenate(axis=-1)([x_raw, x_vel])                # (T, 512)

    # ── Proven BiGRU backbone (identical to bigru_bigger_v1) ──
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(512, return_sequences=True)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(256)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)

    # ── Classification head (identical to bigru_bigger_v1) ──
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(
        inputs=[input_data, input_mask], outputs=outputs,
        name="bigru_velocity_v1"
    )


# ══════════════════════════════════════════════════════════════════════
# bigru_velocity_biggest_v1  (2731 words)
#   Same idea but 3-layer BiGRU 512→256→256 for the larger class space
# ══════════════════════════════════════════════════════════════════════

def build_bigru_velocity_biggest_v1(num_classes: int) -> tf.keras.Model:
    """
    Scaled-up version for 2731 words.
    Same raw + velocity input, 3-layer BiGRU backbone.
    
    Input to GRU: raw(256) + velocity(256) = 512-d
    """
    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")

    # ── Position + Velocity ──
    x_raw = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(input_data)
    velocity = tf.keras.layers.Lambda(compute_velocity)(input_data)
    x_vel = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(velocity)
    x = tf.keras.layers.Concatenate(axis=-1)([x_raw, x_vel])                # (T, 512)

    # ── 3-layer BiGRU backbone ──
    # Layer 1
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(512, return_sequences=True)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Layer 2
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(256, return_sequences=True)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Layer 3
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(256)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)

    # ── Classification head ──
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(
        inputs=[input_data, input_mask], outputs=outputs,
        name="bigru_velocity_biggest_v1"
    )


# ══════════════════════════════════════════════════════════════════════
# bigru_motion_v1
#   Raw coordinates + CNN on Velocity + BiGRU
# ══════════════════════════════════════════════════════════════════════

def build_bigru_motion_v1(num_classes: int) -> tf.keras.Model:
    """
    Motion-aware BiGRU:
    - Raw coordinates (position)
    - Velocity (frame-to-frame motion)
    - Conv1D extracts local motion patterns
    - BiGRU models long-term temporal structure (scaled to 512->256)
    """

    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")

    # ─────────────────────────────
    # RAW FEATURES
    # ─────────────────────────────
    x_raw = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(input_data)  # (T, 256)

    # ─────────────────────────────
    # VELOCITY FEATURES
    # ─────────────────────────────
    velocity = tf.keras.layers.Lambda(compute_velocity)(input_data)
    x_vel = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(velocity)

    # Local motion extraction (scaled to 256 to match capacity)
    x_vel = tf.keras.layers.Conv1D(256, kernel_size=5, padding="same", activation="relu")(x_vel)
    x_vel = tf.keras.layers.BatchNormalization()(x_vel)

    x_vel = tf.keras.layers.Conv1D(256, kernel_size=3, padding="same", activation="relu")(x_vel)
    x_vel = tf.keras.layers.BatchNormalization()(x_vel)

    x_vel = tf.keras.layers.Dropout(0.3)(x_vel)

    # ─────────────────────────────
    # MERGE RAW + MOTION
    # ─────────────────────────────
    x = tf.keras.layers.Concatenate(axis=-1)([x_raw, x_vel])

    # Note: SpatialDropout1D omitted intentionally to prevent the underfitting we just analyzed!

    # ─────────────────────────────
    # TEMPORAL MODELING
    # ─────────────────────────────
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(512, return_sequences=True)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(256)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)

    # ─────────────────────────────
    # HEAD
    # ─────────────────────────────
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(
        inputs=[input_data, input_mask],
        outputs=outputs,
        name="bigru_motion_v1"
    )


# ══════════════════════════════════════════════════════════════════════
# bigru_bigger_angular_flash_v1  (500 words)
#   2 BiGRU layers (256 → 128)  |  angles + angular-velocity  |  SpatialDropout
# ══════════════════════════════════════════════════════════════════════

def build_bigru_bigger_angular_flash_v1(num_classes: int) -> tf.keras.Model:
    """
    For ~500 words.
    Fixes over angular_v1:
      - KinematicFeatureLayer (proper Keras layer, no Lambda)
      - Tighter acos clip (±0.99) → stable gradients
      - Angular velocity concatenated (how fast angles change)
      - SpatialDropout1D(0.2) before GRU
    Input to GRU: raw(256) + angles(46) + ang_vel(46) = 348-d
    """
    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")

    x_raw   = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(input_data)       # (T, 256)
    angles  = KinematicFeatureLayer()(input_data)                              # (T, 46)
    ang_vel = tf.keras.layers.Lambda(compute_angular_velocity)(angles)         # (T, 46)

    x = tf.keras.layers.Concatenate(axis=-1)([x_raw, angles, ang_vel])        # (T, 348)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(512, return_sequences=True)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(256)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(
        inputs=[input_data, input_mask], outputs=outputs,
        name="bigru_bigger_angular_flash_v1"
    )


# ══════════════════════════════════════════════════════════════════════
# bigru_biggest_angular_flash_v1  (2731 words)
#   3 BiGRU layers (512 → 512 → 256)  |  angles + angular-velocity  |  SpatialDropout
# ══════════════════════════════════════════════════════════════════════

def build_bigru_biggest_angular_flash_v1(num_classes: int) -> tf.keras.Model:
    """
    For 2731 words. Same fixes as bigru_bigger_angular_flash_v1 but
    scaled up with a 3-layer BiGRU backbone to handle the larger class space.
    Input to GRU: raw(256) + angles(46) + ang_vel(46) = 348-d
    """
    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")

    x_raw   = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(input_data)
    angles  = KinematicFeatureLayer()(input_data)
    ang_vel = tf.keras.layers.Lambda(compute_angular_velocity)(angles)

    x = tf.keras.layers.Concatenate(axis=-1)([x_raw, angles, ang_vel])

    # Layer 1
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(512, return_sequences=True)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Layer 2
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(256, return_sequences=True)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Layer 3
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(256)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(
        inputs=[input_data, input_mask], outputs=outputs,
        name="bigru_biggest_angular_flash_v1"
    )



def build_bigru_biggest_v1(num_classes: int) -> tf.keras.Model:

    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")

    x   = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(input_data)

    # Layer 1
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(512, return_sequences=True)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Layer 2
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(256, return_sequences=True)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    # Layer 3
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(256)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(
        inputs=[input_data, input_mask], outputs=outputs,
        name="bigru_biggest_v1"
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
        tf.keras.layers.GRU(1024, return_sequences=True)#, unroll=True)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Layer 2
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(512)#, unroll=True)  #unroll = False if breaks
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
def build_conv_only_v2(num_classes: int) -> tf.keras.Model:
    """
    Pure Convolutional model based on removing the BiGRU part.
    """
    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")

    x = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(input_data)

    x = tf.keras.layers.Conv1D(128, kernel_size=10, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=20, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Pooling to remove the temporal dimension
    float_mask = tf.keras.layers.Lambda(lambda m: tf.cast(m, tf.float32))(input_mask)
    mask_expanded = tf.keras.layers.Lambda(lambda m: tf.expand_dims(m, -1))(float_mask)
    x = tf.keras.layers.Multiply()([x, mask_expanded])
    x_sum = tf.keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)
    mask_count = tf.keras.layers.Lambda(
        lambda m: tf.maximum(tf.reduce_sum(m, axis=1, keepdims=True), 1.0)
    )(float_mask)
    x = tf.keras.layers.Lambda(lambda args: args[0] / args[1])([x_sum, mask_count])

    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(
        inputs=[input_data, input_mask], outputs=outputs, name="conv_only_v2"
    )

def build_conv_bigru_v3(num_classes: int) -> tf.keras.Model:
    """
    Conv1D with kernel size 20 into a 128 BiGRU.
    """
    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")

    x = tf.keras.layers.Reshape((MAX_FRAMES, 64 * 4))(input_data)

    x = tf.keras.layers.Conv1D(128, kernel_size=20, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(128)
    )(x, mask=input_mask)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(
        inputs=[input_data, input_mask], outputs=outputs, name="conv_bigru_v3"
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

def _tcn_block_v2(x, filters, kernel_size, dilation_rate):
    residual = x

    out = tf.keras.layers.Conv1D(
        filters,
        kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        activation="relu",
    )(x)

    # optional: lighter normalization
    # out = tf.keras.layers.LayerNormalization()(out)

    if residual.shape[-1] != filters:
        residual = tf.keras.layers.Conv1D(filters, 1, padding="same")(residual)

    return tf.keras.layers.Add()([residual, out])

@tf.keras.utils.register_keras_serializable(package="asl")
def cast_mask_to_float(m):
    return tf.cast(m, tf.float32)


@tf.keras.utils.register_keras_serializable(package="asl")
def expand_mask_last_dim(m):
    return tf.expand_dims(m, -1)


@tf.keras.utils.register_keras_serializable(package="asl")
def reduce_sum_over_time(t):
    return tf.reduce_sum(t, axis=1)


@tf.keras.utils.register_keras_serializable(package="asl")
def safe_mask_count(m):
    return tf.maximum(tf.reduce_sum(m, axis=1, keepdims=True), 1.0)


@tf.keras.utils.register_keras_serializable(package="asl")
def divide_with_mask(args):
    return args[0] / args[1]


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
    x = _tcn_block_v2(x, filters=128, kernel_size=3, dilation_rate=1)
    x = _tcn_block_v2(x, filters=128, kernel_size=3, dilation_rate=2)
    x = _tcn_block_v2(x, filters=128, kernel_size=3, dilation_rate=4)
    x = _tcn_block_v2(x, filters=128, kernel_size=3, dilation_rate=8)

    # ── Masked global average pooling ──
    # Zero out padded frames before averaging
    float_mask = tf.keras.layers.Lambda(
        cast_mask_to_float,
        output_shape=(MAX_FRAMES,),
    )(input_mask)  # (batch, 128)
    mask_expanded = tf.keras.layers.Lambda(
        expand_mask_last_dim,
        output_shape=(MAX_FRAMES, 1),
    )(float_mask)  # (batch, 128, 1)

    x = tf.keras.layers.Multiply()([x, mask_expanded])

    # Sum then divide by number of valid frames
    x_sum = tf.keras.layers.Lambda(
        reduce_sum_over_time,
        output_shape=(128,),
    )(x)
    mask_count = tf.keras.layers.Lambda(
        safe_mask_count,
        output_shape=(1,),
    )(float_mask)
    x = tf.keras.layers.Lambda(
        divide_with_mask,
        output_shape=(128,),
    )([x_sum, mask_count])

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
    "bigru_flash": build_bigru_v3,
    "bigru_angular_v1": build_bigru_angular_v1,
    "bigru_bigger_v1": build_bigru_bigger_v1,
    "bigru_bigger_angular_v1": build_bigru_bigger_angular_v1,
    "bigru_bigger_angular_flash_v1": build_bigru_bigger_angular_flash_v1,
    "bigru_biggest_angular_flash_v1": build_bigru_biggest_angular_flash_v1,
    "bigru_angular": build_bigru_angular,
    "bigru_motion_v1": build_bigru_motion_v1,
    "bigru_velocity_v1": build_bigru_velocity_v1,
    "bigru_velocity_biggest_v1": build_bigru_velocity_biggest_v1,
    "bigru_bigger_v2": build_bigru_bigger_v2,
    "conv_bigru": build_conv_bigru,
    "conv_only_v2": build_conv_only_v2,
    "conv_bigru_v3": build_conv_bigru_v3,
    "tcn": build_tcn,
    "conv1d": build_conv1d,
    "dualpath": build_dualpath,
    "dualpath_v2": build_dualpath_v2,
    "bigru_biggest_v1":build_bigru_biggest_v1,
}


def build_model(name: str, num_classes: int) -> tf.keras.Model:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](num_classes)


# ══════════════════════════════════════════════════════════════════════
# Callbacks
# ══════════════════════════════════════════════════════════════════════

class OverfitGuard(tf.keras.callbacks.Callback):
    """Hard-stop when (train_acc - val_acc) exceeds max_gap.

    Semantics across training phases:
    - Early epochs (val_acc > train_acc due to augmentation):
        gap = train_acc - val_acc  →  NEGATIVE  →  never triggers.
        The math handles this naturally; no special casing needed.
    - Post-crossover (train_acc > val_acc):
        gap becomes positive. Once it exceeds max_gap → stop.

    grace_epochs is a safety buffer for the first few noisy epochs
    where both accuracies are near zero and the ratio is unstable.
    The crossover in typical runs happens around epoch 40-50, so
    grace_epochs=50 means the guard only arms after we've confirmed
    normal training dynamics are established.

    Args:
        max_gap:      Max allowed (train_acc - val_acc). Default 0.08 (8%).
        grace_epochs: Epochs to skip before activating. Default 50.
    """

    def __init__(self, max_gap: float = 0.08, grace_epochs: int = 50):
        super().__init__()
        self.max_gap = max_gap
        self.grace_epochs = grace_epochs

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.grace_epochs:
            return
        train_acc = logs.get("accuracy", 0.0)
        val_acc   = logs.get("val_accuracy", 0.0)
        gap = train_acc - val_acc  # negative when val > train → safe
        if gap > self.max_gap:
            print(
                f"\n[OverfitGuard] Stopping at epoch {epoch}: "
                f"train={train_acc:.4f}, val={val_acc:.4f}, "
                f"gap={gap:.4f} > threshold={self.max_gap:.4f}"
            )
            self.model.stop_training = True


def get_callbacks(model_dir: str, model_name: str | None = None, patience: int = 15):
    """Callback suite designed to stop training at the right time.

    ── Why val_accuracy (not val_loss + min_delta) for EarlyStopping? ──
    Under label smoothing the loss surface is shifted: val_loss keeps
    making micro-improvements (e.g. 1.7249 → 1.7240 over 50 epochs)
    even after val_accuracy has completely flatlined. With or without
    min_delta, those micro-improvements reset EarlyStopping's counter
    and the run never terminates. val_accuracy is immune to this because
    it is discrete and actually plateaus when learning stops.

    ── Why min_delta=0.001 on val_accuracy? ──
    Without min_delta, a 0.0001 acc improvement (noise-level) resets the
    counter. With min_delta=0.001 the model must gain ≥0.1 pp per patience
    window or EarlyStopping fires.

    ── Why is ReduceLROnPlateau still on val_loss? ──
    LR scheduling works better on a smooth signal (loss) rather than a
    noisy one (accuracy). But min_delta=0.001 prevents it from triggering
    on pure noise fluctuations.

    ── OverfitGuard gap threshold = 8% ──
    Empirically, train-val acc gap grows from ~7% at epoch 82 to ~13%
    by epoch 176 with essentially zero val_acc gain. The guard fires at
    8% to cut the wasted tail. grace_epochs=50 arms it after the natural
    crossover point (~epoch 44) where val_acc stops exceeding train_acc.
    """
    if model_name is None:
        model_name = datetime.datetime.now().strftime("%d-%m-%y__%H-%M")

    os.makedirs(model_dir, exist_ok=True)

    return [
        # ── Primary checkpoint: best val_accuracy ──
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, f"asl_{model_name}_best.keras"),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
        ),
        # ── All-time-best checkpoint (separate file, never regresses) ──
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, f"asl_{model_name}_alltimebest.keras"),
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=0,
        ),
        # ── Early stopping on val_accuracy ──
        # monitor="val_accuracy" + mode="max": stops when acc stops growing.
        # min_delta=0.001: a 0.1 pp improvement is the minimum that "counts".
        # restore_best_weights: rolls back to the epoch with best val_acc.
        tf.keras.callbacks.EarlyStopping(
            patience=patience,
            restore_best_weights=True,
            monitor="val_accuracy",
            mode="max",
            min_delta=0.001,
            verbose=1,
        ),
        # ── CSV logger ──
        tf.keras.callbacks.CSVLogger(
            os.path.join(model_dir, f"asl_{model_name}_log.csv"),
            separator=",",
            append=False,
        ),
        # ── LR scheduler (on val_loss for smooth signal) ──
        # patience=5 fires well before EarlyStopping patience=15, giving
        # LR reduction a fair chance before the run is terminated.
        # min_delta=0.001: must improve val_loss by at least 0.001 absolute.
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            min_delta=0.001,
            verbose=1,
        ),
        # ── Overfit guard: hard-stop at 8% train-val gap ──
        OverfitGuard(max_gap=0.02, grace_epochs=50),
    ]
