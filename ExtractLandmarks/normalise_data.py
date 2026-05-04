from re import T
import numpy as np
from visualise_lms_in_3d import (
    infinite_plot_landmarks,
    infinite_plot_landmarks_compare2,
    infinite_plot_landmarks_compare2_sidebyside,
)
import os
import shutil

# motion_detection.py is a local copy of MotionTesting/motion_detection.py
from motion_detection import get_final_extraction, DEFAULTS as MOTION_DEFAULTS

"""
needed_poses = [0,2,5,7,8,9,10,11,12,13,15,14,16,23,24,17-22] #15+6 = 21

array of landmarks need --  21x4 = 21landmarks with 4 dimensions [x y z visibility]
"""
"""
since gpus are optimised for powers of 2 calculations

and percentiles are: 
      85% = 116.0
      90% = 136.0

128frames is good

fir we can do 64x4 features   and in the last extra feature we can simply do difference between wrists or something

"""


def normalise_lm_arr_spatially(basic_arr):
    needed_poses = [
        0,  # 0
        2,
        5,
        7,
        8,
        9,
        10,
        11,  # 7
        12,  # 8
        13,
        14,
        15,  # 11
        16,  # 12
        17,
        18,
        19,
        20,
        21,
        22,
        23,  # 19
        24,  # 20
    ]
    total_num_frames = int(basic_arr[0][0][0])
    print(int(total_num_frames))
    # print(basic_arr[0])
    # print(basic_arr.shape)
    lmsneeded = np.concatenate([needed_poses, np.arange(33, 75)])

    # print(lmsneeded)
    # exit()

    arr_reduced_to_63 = basic_arr[1:, lmsneeded, :]
    """
    11 => 7
    12 => 8
    23 => 13
    24 => 14
    """

    arr = np.zeros((total_num_frames, 64, 4), dtype=np.float32)
    arr[:, :63, :] = arr_reduced_to_63
    # print(arr.shape)
    # infinite_plot_landmarks(arr)
    # exit()
    for frame_ind in range(0, total_num_frames):
        lms = arr[frame_ind]

        LEFT_SHOULDER = lms[7][:3]
        RIGHT_SHOULDER = lms[8][:3]

        CENTER_OF_SHOULDER = (LEFT_SHOULDER + RIGHT_SHOULDER) / 2.0

        SHOULDER_LENGTH = (
            np.linalg.norm(LEFT_SHOULDER - RIGHT_SHOULDER) + 1e-8
        )  # agar bychance distance 0 agaya

        for i in range(63):
            vv = lms[i][3]  # visibility / presence score

            # BUG FIX: If the landmark was not detected (visibility == 0), MediaPipe
            # stores (0, 0, 0, 0). Subtracting the shoulder center would turn this into
            # a non-zero vector (-shoulder_x, -shoulder_y, ...) which fakes displacement
            # and creates spurious energy spikes when a hand first appears. Keep as zero.
            if vv == 0.0:
                arr[frame_ind, i] = [0.0, 0.0, 0.0, 0.0]
                continue

            xx = lms[i][0] - CENTER_OF_SHOULDER[0]
            yy = lms[i][1] - CENTER_OF_SHOULDER[1]
            zz = lms[i][2] - CENTER_OF_SHOULDER[2]

            arr[frame_ind, i] = [
                xx / SHOULDER_LENGTH,
                yy / SHOULDER_LENGTH,
                zz,
                vv,
            ]

        
        "64th landmark ko difference bw wrists rakhlete h"
        xyz_of_left_wrist  = arr[frame_ind, 11, :3]   # already normalised above
        xyz_of_right_wrist = arr[frame_ind, 12, :3]
        diff_bw_wrists     = xyz_of_left_wrist - xyz_of_right_wrist
        vis_wrists         = (lms[11, 3] + lms[12, 3]) / 2
        arr[frame_ind, 63, :3] = diff_bw_wrists
        arr[frame_ind, 63, 3] = vis_wrists
    return arr


def normalise_lm_arr_temporally(arr):
    total_num_frames = arr.shape[0]
    # print(total_num_frames)

    normalised_numb_of_frames = 64   # 95.4% of trimmed segments fit in 64f; median=37f
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


def flip_raw_arr(arr):
    flipped = arr.copy()
    # print(arr[0])
    notzero_mask = flipped[:, :, 0] != 0.0
    notzero_mask[0, :] = 0

    # print(notzero_mask)
    # print(notzero_mask.shape)

    flipped[notzero_mask, 0] = 1 - flipped[notzero_mask, 0]

    left_lms = flipped[1:, 33:54, :].copy()
    flipped[1:, 33:54, :] = flipped[1:, 54:75, :]
    flipped[1:, 54:75, :] = left_lms

    # https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
    left_right_lm_pairs = [
        (1, 4),
        (2, 5),
        (3, 6),
        (7, 8),
        (9, 10),
        (11, 12),
        (23, 24),
        (25, 26),
        (27, 28),
        (29, 30),
        (31, 32),
        (13, 14),
        (15, 16),
        (17, 18),
        (19, 20),
        (21, 22),
    ]

    for l_ind, r_ind in left_right_lm_pairs:
        left_lm = flipped[1:, l_ind, :].copy()
        flipped[1:, l_ind, :] = flipped[1:, r_ind, :]
        flipped[1:, r_ind, :] = left_lm

    return flipped


def process_one(input_path, output_path, flip=False):
    og_array = None
    normalised_arr = None
    try:
        og_array = np.load(input_path)
        if flip:
            og_array = flip_raw_arr(og_array)

        # ── Motion-based trimming ──────────────────────────────────────
        # Extract only the active signing segment before normalisation.
        # This removes idle frames at the start/end and camera-off sweeps.
        total_frames = int(og_array[0, 0, 0])
        frames_raw   = og_array[1:total_frames + 1]   # shape (T, 75, 4)

        seg = get_final_extraction(frames_raw)         # uses DEFAULTS params
        if seg is None:
            # ── No signing segment: quarantine to NO_SEGMENTS/<word>/ ──
            # output_path layout: <op_dataset_path>/<WORD>/<filename>.npy
            word_name  = os.path.basename(os.path.dirname(output_path))
            no_seg_dir = os.path.join(
                os.path.dirname(os.path.dirname(output_path)),
                "NO_SEGMENTS",
                word_name,
            )
            os.makedirs(no_seg_dir, exist_ok=True)
            dest = os.path.join(no_seg_dir, os.path.basename(input_path))
            shutil.copy2(input_path, dest)
            with open("normalising_logs.log", "a") as f:
                f.write(f"NO_SEGMENT :: '{input_path}' -> '{dest}'\n")
            return   # skip normalisation

        seg_start, seg_end = seg
        trimmed_raw = frames_raw[seg_start:seg_end]    # (T', 75, 4)

        # Rebuild a minimal array that normalise_lm_arr_spatially expects:
        # Row 0 = metadata [total_frames, fps, ...], rows 1..N = frames.
        trimmed_len = len(trimmed_raw)
        meta_row    = og_array[0:1].copy()
        meta_row[0, 0, 0] = trimmed_len           # update frame count
        og_array    = np.concatenate([meta_row, trimmed_raw], axis=0)
        # ──────────────────────────────────────────────────────────────

        spatially_normalised_arr = normalise_lm_arr_spatially(og_array)
        temporally_spatially_normalised_arr, mask = normalise_lm_arr_temporally(
            spatially_normalised_arr
        )
        if flip:
            npz_path = output_path[:-4] + "_flipped.npz"
        else:
            npz_path = output_path[:-4] + ".npz"
        print(npz_path)
        np.savez(npz_path, data=temporally_spatially_normalised_arr, mask=mask)
    except Exception as e:
        with open("normalising_logs.log", "a") as f:
            f.write(f"path:'{input_path}' :: error:'{e}'\n")
    finally:
        del og_array, normalised_arr


def process_all(
    dataset_path="./dataset1.0/landmarks/",
    op_dataset_path="./dataset1.0/landmarks_npz/",
    flip=False,
):
    os.makedirs(op_dataset_path, exist_ok=True)
    for word in os.listdir(dataset_path):
        wordpath = os.path.join(dataset_path, word)
        op_wordpath = os.path.join(op_dataset_path, word)
        os.makedirs(op_wordpath, exist_ok=True)
        for file in os.listdir(wordpath):
            if file.endswith(".npy"):
                filepath = os.path.join(wordpath, file)
                op_filepath = os.path.join(op_wordpath, file)
                process_one(filepath, op_filepath, flip=flip)


if __name__ == "__main__":
    dataset_path = "/run/media/aryan/PoopiDrive/projects_linux/microsoft asl citizen/landmarks/"
    op_dataset_path = "./dataset4.0/landmarks_npz/"

    # process_one("./test_actor.npy", "./test_actor.npz")

    # exit()
    process_all(dataset_path=dataset_path, op_dataset_path=op_dataset_path, flip=False)
    process_all(dataset_path=dataset_path, op_dataset_path=op_dataset_path, flip=True)
    exit(0)
    # arr = np.load("./dataset1.0/landmarks/8HOUR/8HOUR_07243138866452936-8 HOUR.npy")
    # # flipped = flip_raw_arr(arr)
    #
    # arr = normalise_lm_arr_spatially(arr)
    # flipped = normalise_lm_arr_spatially(flipped)
    #
    # infinite_plot_landmarks_compare2_sidebyside(arr, flipped)
    # exit()
    # dataset_path = "./dataset1.0/landmarks/"
    # process_all(dataset_path=dataset_path)
    # exit(0)
    SHAPE_OF_LANDMARK_IN_ONE_FRAME = (75, 4)
    # NP_LANDMARK_ALL_FRAMES = np.zeros(
    # (total_num_frames + 1, *SHAPE_OF_LANDMARK_IN_ONE_FRAME), dtype=np.float32
    # )  # mediapipe xyz float64 me dera h
    # frame_index = 1

    """ 
    first frame can store metadata like  fps frames etc

    """

    # NP_LANDMARK_ALL_FRAMES[0][0] = total_num_frames
    # NP_LANDMARK_ALL_FRAMES[0][1] = fps
    pathh = "./MS_ASL_Result/1DOLLAR_641444141964802-1 DOLLAR.npy"
    pathh = "./MS_ASL_Result/ABOUT1_718305545661964-ABOUT.npy"
    pathh = "./MS_ASL_Result/5DOLLARS_2660970218292622-5 DOLLARS.npy"
    pathh = "./dataset1.0/landmarks/ACT/ACT_4016028161461893-ACT.npy"
    pathh1 = "./dataset1.0/landmarks/ABOVE/ABOVE_12573880874826582-ABOVE.npy"
    pathh2 = "./dataset1.0/landmarks/ABOVE/ABOVE_18500814503221719-ABOVE.npy"

    pathh1 = "./dataset1.0/landmarks/ACT/ACT_4016028161461893-ACT.npy"
    # infinite_plot_landmarks(arr)

    basicarr1 = np.load(pathh1)
    arr1 = normalise_lm_arr_spatially(basicarr1)
    # arr1_128 = normalise_lm_arr_temporally(arr1)[0]
    # print(arr1_128.shape)
    # infinite_plot_landmarks(arr1_128)
    # normalise_lm_arr_for_time(arr1)
    # exit()

    basicarr2 = np.load(pathh2)
    arr2 = normalise_lm_arr_spatially(basicarr2)
    infinite_plot_landmarks_compare2(arr1, arr2)
    """have some videos that are 100 frames


    some that are 70


    avg is 82

    median is 74


    im thinking of normalising to 80frames


    by smapling the larger ones with start and middle getting priorityy

    80% being the first 70%

    and 20% being last 30%


    and interpolating between frames to get a larger array"""
