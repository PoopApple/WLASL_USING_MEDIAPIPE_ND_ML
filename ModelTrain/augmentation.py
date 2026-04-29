"""
Data augmentation for ASL landmark sequences.

All augmentations operate on SINGLE SAMPLES (before batching) using pure
TF ops so they work inside tf.data.Dataset.map() without hitting the GIL.

Each augmentation is designed for normalised landmark data:
    shape: (128, 64, 4)  — 128 frames, 64 landmarks, 4 values (x, y, z, visibility)
    mask:  (128,)         — True for real frames, False for padding

Augmentations applied (training only):
    1. Gaussian noise    — simulates natural pose estimation variance
    2. Spatial scaling   — slight size variation (0.9× to 1.1×)
    3. Temporal masking  — randomly zero out some frames (simulate occlusion)
    4. Landmark dropout  — randomly zero out entire landmarks across all frames
    5. Time warping      — randomly speed up / slow down segments
"""

import tensorflow as tf


# ══════════════════════════════════════════════════════════════════════
# CONFIG — augmentation hyperparameters
# ══════════════════════════════════════════════════════════════════════

NOISE_STDDEV = 0.015          # Gaussian noise σ (landmarks are ~0-1 range)
SCALE_RANGE = (0.90, 1.10)    # Random spatial scale factor range
FRAME_DROP_RATE = 0.05        # Probability of dropping each frame
LANDMARK_DROP_RATE = 0.05     # Probability of dropping each landmark
TIME_WARP_PROB = 0.3          # Probability of applying time warping


# ══════════════════════════════════════════════════════════════════════
# Individual augmentations
# ══════════════════════════════════════════════════════════════════════

def add_gaussian_noise(data, mask):
    """
    Add small Gaussian noise to landmark coordinates.
    
    WHY: MediaPipe outputs aren't perfectly consistent frame-to-frame.
    This simulates that natural jitter so the model doesn't memorise
    exact coordinate values.
    
    Only applied to real frames (respects mask).
    """
    noise = tf.random.normal(shape=tf.shape(data), stddev=NOISE_STDDEV)
    # Zero out noise for padded frames
    mask_expanded = tf.cast(mask, tf.float32)[:, tf.newaxis, tf.newaxis]  # (128, 1, 1)
    noise = noise * mask_expanded
    return data + noise, mask


def random_spatial_scale(data, mask):
    """
    Scale all landmark coordinates by a random factor.
    
    WHY: Different signers have different body sizes. Even after normalisation,
    slight scale variation helps the model be more robust.
    
    Applies a single scale factor to x, y, z (not visibility).
    """
    scale = tf.random.uniform([], SCALE_RANGE[0], SCALE_RANGE[1])
    # Only scale x, y, z (first 3 channels), leave visibility (channel 4) alone
    xyz = data[:, :, :3] * scale
    vis = data[:, :, 3:]
    return tf.concat([xyz, vis], axis=-1), mask


def random_frame_drop(data, mask):
    """
    Randomly zero out some frames and mark them as padded.
    
    WHY: Simulates frame drops, occlusion, and missing detections that
    happen in real-world video capture. Forces the model to be robust
    to incomplete sequences.
    """
    # Generate per-frame drop probability
    drop_mask = tf.random.uniform([tf.shape(data)[0]]) < FRAME_DROP_RATE
    # Only drop real frames, don't "drop" already-padded frames
    drop_mask = tf.logical_and(drop_mask, mask)
    
    # Zero out dropped frames
    keep_mask = tf.logical_not(drop_mask)
    keep_float = tf.cast(keep_mask, tf.float32)[:, tf.newaxis, tf.newaxis]
    data = data * keep_float
    
    # Do NOT update the mask. cuDNN requires masks to be strictly right-padded
    # (i.e. all True followed by all False). If we put False in the middle, cuDNN crashes.
    # By zeroing the data but leaving the mask True, the GRU processes a frame of pure zeros,
    # which is exactly what we want to simulate dropped frames/dropout!
    return data, mask


def random_landmark_dropout(data, mask):
    """
    Randomly zero out entire landmarks across ALL frames.
    
    WHY: Forces the model to not rely on any single landmark.
    If landmark #23 (right wrist) is always present, the model might
    overfit to wrist-only patterns. Dropping it sometimes forces the
    model to learn from other landmarks too.
    """
    # Generate per-landmark drop probability
    drop_landmarks = tf.random.uniform([tf.shape(data)[1]]) < LANDMARK_DROP_RATE  # (64,)
    drop_expanded = tf.cast(tf.logical_not(drop_landmarks), tf.float32)
    # Shape: (1, 64, 1) — broadcast across frames and features
    drop_expanded = drop_expanded[tf.newaxis, :, tf.newaxis]
    data = data * drop_expanded
    return data, mask


def random_time_warp(data, mask):
    """
    Randomly stretch or compress the temporal axis by resampling frames.
    
    WHY: Different signers perform the same sign at different speeds.
    Time warping simulates faster/slower signing without changing the
    actual motion pattern.
    
    Implementation: resample the valid frames using linear interpolation
    at slightly shifted indices.
    """
    should_warp = tf.random.uniform([]) < TIME_WARP_PROB
    
    if not should_warp:
        return data, mask
    
    num_frames = tf.shape(data)[0]
    num_valid = tf.reduce_sum(tf.cast(mask, tf.int32))
    
    # Only warp if we have enough valid frames
    if num_valid < 10:
        return data, mask
    
    # Generate warped indices: slightly randomised version of [0, 1, ..., num_valid-1]
    warp_factor = tf.random.uniform([], 0.85, 1.15)
    original_indices = tf.cast(tf.range(num_valid), tf.float32)
    center = tf.cast(num_valid, tf.float32) / 2.0
    warped_indices = center + (original_indices - center) * warp_factor
    warped_indices = tf.clip_by_value(warped_indices, 0.0, tf.cast(num_valid - 1, tf.float32))
    
    # Get integer indices for gather (nearest-neighbour resampling)
    warped_indices_int = tf.cast(tf.round(warped_indices), tf.int32)
    
    # Gather warped frames from valid portion
    valid_data = data[:num_valid]
    warped_data = tf.gather(valid_data, warped_indices_int)
    
    # Reconstruct: warped valid frames + original padding
    padding = data[num_valid:]
    data = tf.concat([warped_data, padding], axis=0)
    
    return data, mask


# ══════════════════════════════════════════════════════════════════════
# Combined augmentation function
# ══════════════════════════════════════════════════════════════════════

def augment_sample(inputs, label):
    """
    Apply all augmentations to a single sample.
    
    Args:
        inputs: dict with "input_data" (128, 64, 4) and "input_mask" (128,)
        label: int class label
    
    Returns:
        (augmented_inputs, label)
    """
    data = inputs["input_data"]
    mask = inputs["input_mask"]
    
    # Apply augmentations in sequence
    data, mask = add_gaussian_noise(data, mask)
    data, mask = random_spatial_scale(data, mask)
    
    # Disabled: Zeroing out frames/landmarks teleports them to (0,0,0) 
    # which is the chest in our normalized data. This breaks trajectories.
    # data, mask = random_frame_drop(data, mask)
    # data, mask = random_landmark_dropout(data, mask)
    
    # Time warp disabled by default — can cause shape issues with tf.data
    # Uncomment if needed: data, mask = random_time_warp(data, mask)
    
    return {"input_data": data, "input_mask": mask}, label
