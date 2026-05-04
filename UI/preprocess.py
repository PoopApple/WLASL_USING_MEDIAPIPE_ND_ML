import numpy as np

def process_landmarks(lms_75):
    """
    lms_75: numpy array of shape (75, 4)
    Performs spatial normalization as done in normalise_data.py but for a single frame.
    """
    if lms_75 is None:
        return None

    needed_poses = [
        0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    ]
    lmsneeded = np.concatenate([needed_poses, np.arange(33, 75)])
    
    lms_63 = lms_75[lmsneeded, :]
    
    out_arr = np.zeros((64, 4), dtype=np.float32)
    out_arr[:63, :] = lms_63
    
    lms = out_arr
    
    LEFT_SHOULDER = lms[7][:3]
    RIGHT_SHOULDER = lms[8][:3]

    CENTER_OF_SHOULDER = (LEFT_SHOULDER + RIGHT_SHOULDER) / 2.0

    SHOULDER_LENGTH = (
        np.linalg.norm(LEFT_SHOULDER - RIGHT_SHOULDER) + 1e-8
    )

    for i in range(63):
        xx = lms[i][0] - CENTER_OF_SHOULDER[0]
        yy = lms[i][1] - CENTER_OF_SHOULDER[1]
        zz = lms[i][2] - CENTER_OF_SHOULDER[2]
        vv = lms[i][3]

        out_arr[i] = [
            xx / SHOULDER_LENGTH,
            yy / SHOULDER_LENGTH,
            zz,
            vv,
        ]

    xyz_of_left_wrist = lms[11, :3]
    xyz_of_right_wrist = lms[12, :3]

    diff_bw_wrists = xyz_of_left_wrist - xyz_of_right_wrist
    vis_wrists = (lms[11, 3] + lms[12, 3]) / 2

    out_arr[63, :3] = diff_bw_wrists
    out_arr[63, 3] = vis_wrists

    return out_arr

def normalise_lm_arr_temporally(arr):
    total_num_frames = arr.shape[0]

    normalised_numb_of_frames = 128
    arr_padded = np.zeros((normalised_numb_of_frames, 64, 4), dtype=np.float32)
    mask = np.zeros(normalised_numb_of_frames, dtype=bool)

    if total_num_frames == normalised_numb_of_frames:
        arr_padded[:] = arr[:]
        mask[:] = True
    elif total_num_frames > normalised_numb_of_frames:
        indices = np.round(
            np.linspace(0, total_num_frames - 1, normalised_numb_of_frames)
        ).astype(int)
        arr_padded[:] = arr[indices]
        mask[:] = True
    else:
        arr_padded[:total_num_frames, :, :] = arr
        mask[:total_num_frames] = True

    return arr_padded, mask


# Pose landmark indices in the PROCESSED array (mapped via needed_poses list).
# needed_poses = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
# Maps raw index → processed index (only pairs that both appear in needed_poses matter).
_POSE_LR_PAIRS_PROCESSED = [
    # (left_processed_idx, right_processed_idx)
    # raw (7,8)   → processed (3, 4)
    (3, 4),
    # raw (9,10)  → processed (5, 6)
    (5, 6),
    # raw (11,12) → processed (7, 8)
    (7, 8),
    # raw (13,14) → processed (9, 10)
    (9, 10),
    # raw (15,16) → processed (11, 12)
    (11, 12),
    # raw (17,18) → processed (13, 14)
    (13, 14),
    # raw (19,20) → processed (15, 16)
    (15, 16),
    # raw (21,22) → processed (17, 18)
    (17, 18),
    # raw (23,24) → processed (19, 20)
    (19, 20),
]

# In the processed array:
#   indices 21-41  = left hand  (raw 33-53)
#   indices 42-62  = right hand (raw 54-74)
_LEFT_HAND_SLICE  = slice(21, 42)
_RIGHT_HAND_SLICE = slice(42, 63)


def flip_processed_arr(arr):
    """
    Horizontally flip a processed (T, 64, 4) landmark array.
    Mirrors x-coordinates and swaps left/right body-side features,
    matching the flip_raw_arr transform applied during training.

    Args:
        arr: np.ndarray of shape (T, 64, 4)
    Returns:
        flipped: same shape (T, 64, 4)
    """
    flipped = arr.copy()

    # 1. Mirror x for all non-zero landmarks
    nonzero = flipped[:, :, 0] != 0.0
    flipped[nonzero, 0] = -flipped[nonzero, 0]  # already shoulder-centred, so negate x

    # 2. Swap left/right hand segments
    left_hand  = flipped[:, _LEFT_HAND_SLICE, :].copy()
    flipped[:, _LEFT_HAND_SLICE, :]  = flipped[:, _RIGHT_HAND_SLICE, :]
    flipped[:, _RIGHT_HAND_SLICE, :] = left_hand

    # 3. Swap paired left/right pose landmarks
    for l_idx, r_idx in _POSE_LR_PAIRS_PROCESSED:
        left_lm = flipped[:, l_idx, :].copy()
        flipped[:, l_idx, :] = flipped[:, r_idx, :]
        flipped[:, r_idx, :] = left_lm

    # 4. Negate the wrist-diff feature's x component (index 63, channel 0)
    flipped[:, 63, 0] = -flipped[:, 63, 0]

    return flipped