import numpy as np

TARGET_FRAMES = 64  # Must match MAX_FRAMES in ModelTrain/model.py

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

        # Ghost-zero guard: only normalise if landmark is visible
        if vv > 0.0:
            out_arr[i] = [
                xx / SHOULDER_LENGTH,
                yy / SHOULDER_LENGTH,
                zz,
                vv,
            ]
        else:
            out_arr[i] = [0.0, 0.0, 0.0, 0.0]

    xyz_of_left_wrist = lms[11, :3]
    xyz_of_right_wrist = lms[12, :3]

    diff_bw_wrists = xyz_of_left_wrist - xyz_of_right_wrist
    vis_wrists = (lms[11, 3] + lms[12, 3]) / 2

    out_arr[63, :3] = diff_bw_wrists
    out_arr[63, 3] = vis_wrists

    return out_arr


def normalise_lm_arr_temporally(arr, target_frames=TARGET_FRAMES):
    """
    Temporally resample/pad a (T, 64, 4) landmark array to (target_frames, 64, 4).
    
    - If T > target_frames : uniformly subsample (speedup)
    - If T < target_frames : zero-pad with mask=False for padded frames
    - If T == target_frames: pass through
    
    Returns
    -------
    arr_padded : np.ndarray, shape (target_frames, 64, 4)
    mask       : np.ndarray, shape (target_frames,), dtype=bool
                 True where real data exists, False for padding.
    """
    total_num_frames = arr.shape[0]

    arr_padded = np.zeros((target_frames, 64, 4), dtype=np.float32)
    mask = np.zeros(target_frames, dtype=bool)

    if total_num_frames == target_frames:
        arr_padded[:] = arr[:]
        mask[:] = True
    elif total_num_frames > target_frames:
        # Uniform subsampling — spreads evenly across the sequence
        indices = np.round(
            np.linspace(0, total_num_frames - 1, target_frames)
        ).astype(int)
        arr_padded[:] = arr[indices]
        mask[:] = True
    else:
        arr_padded[:total_num_frames, :, :] = arr
        mask[:total_num_frames] = True

    return arr_padded, mask


# ── Pose landmark indices in the PROCESSED array ──────────────────────────────
# needed_poses = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
_POSE_LR_PAIRS_PROCESSED = [
    (3, 4), (5, 6), (7, 8), (9, 10), (11, 12),
    (13, 14), (15, 16), (17, 18), (19, 20),
]

_LEFT_HAND_SLICE  = slice(21, 42)
_RIGHT_HAND_SLICE = slice(42, 63)


def flip_processed_arr(arr):
    """
    Horizontally flip a processed (T, 64, 4) landmark array.
    """
    flipped = arr.copy()
    nonzero = flipped[:, :, 0] != 0.0
    flipped[nonzero, 0] = -flipped[nonzero, 0]
    left_hand  = flipped[:, _LEFT_HAND_SLICE, :].copy()
    flipped[:, _LEFT_HAND_SLICE, :]  = flipped[:, _RIGHT_HAND_SLICE, :]
    flipped[:, _RIGHT_HAND_SLICE, :] = left_hand
    for l_idx, r_idx in _POSE_LR_PAIRS_PROCESSED:
        left_lm = flipped[:, l_idx, :].copy()
        flipped[:, l_idx, :] = flipped[:, r_idx, :]
        flipped[:, r_idx, :] = left_lm
    flipped[:, 63, 0] = -flipped[:, 63, 0]
    return flipped