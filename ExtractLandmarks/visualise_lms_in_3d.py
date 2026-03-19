import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np


def plot_landmarks(plt, ax, landmarks, visibility_th=0.5, pause=True, clr=True):

    # Helper function to extract points
    def extract_points(ll):
        xs, ys, zs = [], [], []
        for i in ll:
            x, y, z, vis = landmarks[i]
            # if vis < visibility_th:
            #     continue
            xs.append(x)
            ys.append(y * (-1))
            zs.append(z)
        return xs, zs, ys

    # waist_x, waist_y, waist_z = extract_points(waist_index_list)
    needed_poses = [
        0,
        2,
        5,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
    ]

    map_pose_index = {}

    for i, index in enumerate(needed_poses):
        map_pose_index[index] = i
    # print(map_pose_index)

    face_index_list = [8, 5, 0, 2, 7, 9, 10]
    right_arm_index_list = [11, 13, 15, 17, 19, 21]
    left_arm_index_list = [12, 14, 16, 18, 20, 22]
    right_body_side_index_list = [11, 23]
    left_body_side_index_list = [12, 24]
    shoulder_index_list = [11, 12]
    waist_index_list = [23, 24]

    left_hand_index_list = range(21, 42)
    right_hand_index_list = range(42, 42 + 21)

    face_index_list = [map_pose_index[x] for x in face_index_list]
    right_arm_index_list = [map_pose_index[x] for x in right_arm_index_list]
    left_arm_index_list = [map_pose_index[x] for x in left_arm_index_list]
    right_body_side_index_list = [map_pose_index[x] for x in right_body_side_index_list]
    left_body_side_index_list = [map_pose_index[x] for x in left_body_side_index_list]
    shoulder_index_list = [map_pose_index[x] for x in shoulder_index_list]
    waist_index_list = [map_pose_index[x] for x in waist_index_list]

    face_x, face_y, face_z = extract_points(face_index_list)
    right_arm_x, right_arm_y, right_arm_z = extract_points(right_arm_index_list)
    left_arm_x, left_arm_y, left_arm_z = extract_points(left_arm_index_list)
    right_body_side_x, right_body_side_y, right_body_side_z = extract_points(
        right_body_side_index_list
    )
    left_body_side_x, left_body_side_y, left_body_side_z = extract_points(
        left_body_side_index_list
    )
    shoulder_x, shoulder_y, shoulder_z = extract_points(shoulder_index_list)
    waist_x, waist_y, waist_z = extract_points(waist_index_list)

    lefthand_x, lefthand_y, lefthand_z = extract_points(left_hand_index_list)
    righthand_x, righthand_y, righthand_z = extract_points(right_hand_index_list)
    if clr:
        ax.cla()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    ax.scatter(face_x, face_y, face_z)
    ax.plot(right_arm_x, right_arm_y, right_arm_z)
    ax.plot(left_arm_x, left_arm_y, left_arm_z)
    ax.plot(right_body_side_x, right_body_side_y, right_body_side_z)
    ax.plot(left_body_side_x, left_body_side_y, left_body_side_z)
    ax.plot(shoulder_x, shoulder_y, shoulder_z)
    ax.plot(waist_x, waist_y, waist_z)

    ax.plot(righthand_x, righthand_y, righthand_z)
    ax.plot(lefthand_x, lefthand_y, lefthand_z)

    if pause:
        plt.pause(0.001)


def infinite_plot_landmarks(arr: np.ndarray | str = "../gte9_test_vid/bed.npy"):
    if type(arr) is str:
        arrr = np.load(arr)
    else:
        arrr = arr

    frames = arrr.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    while True:
        try:
            for fr in range(frames):
                # print(arrr[fr])
                # print(arrr[fr].shape)
                plot_landmarks(plt, ax, arrr[fr])
        except Exception as e:
            print(e)
            break


def infinite_plot_landmarks_compare2(arr1="../gte9_test_vid/bed.npy", arr2=""):
    if type(arr1) is str:
        arrr1 = np.load(arr1)
    else:
        arrr1 = arr1

    if type(arr2) is str:
        arrr2 = np.load(arr2)
    else:
        arrr2 = arr2

    frames1 = arrr1.shape[0]
    frames2 = arrr2.shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    while True:
        try:
            for fr in range(max(frames1, frames2)):
                # print(arrr[fr])
                # print(arrr[fr].shape)

                if fr < frames1:
                    plot_landmarks(plt, ax, arrr1[fr], clr=True, pause=False)
                if fr < frames2:
                    plot_landmarks(plt, ax, arrr2[fr], clr=False, pause=False)
                plt.pause(0.001)

        except Exception as e:
            print(e)
            break


def infinite_plot_landmarks_compare2_sidebyside(
    arr1: np.ndarray | str = "../gte9_test_vid/bed.npy", arr2: np.ndarray | str = ""
):
    if type(arr1) is str:
        arrr1 = np.load(arr1)
    else:
        arrr1 = arr1

    if type(arr2) is str:
        arrr2 = np.load(arr2)
    else:
        arrr2 = arr2

    frames1 = arrr1.shape[0]
    frames2 = arrr2.shape[0]

    # 1. Make the figure wider to accommodate two plots side-by-side
    fig = plt.figure(figsize=(12, 6))

    # 2. Add two separate 3D subplots (1 row, 2 columns, index 1 and 2)
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    is_running = [True]

    def close_viewer(event):
        # Triggered if user presses 'q', 'escape', or clicks the 'X' button
        if hasattr(event, "key") and event.key in ["q", "escape"]:
            is_running[0] = False
            plt.close("all")
        elif event.name == "close_event":
            is_running[0] = False

    fig.canvas.mpl_connect("key_press_event", close_viewer)
    fig.canvas.mpl_connect("close_event", close_viewer)

    while is_running[0]:
        try:
            for fr in range(max(frames1, frames2)):
                # 3. Pass ax1 to the first and ax2 to the second.
                # Note: clr=True is used for BOTH now, assuming your plot_landmarks
                # function clears the specific axis passed to it before drawing the new frame.
                if fr < frames1:
                    plot_landmarks(plt, ax1, arrr1[fr], clr=True, pause=False)
                if fr < frames2:
                    plot_landmarks(plt, ax2, arrr2[fr], clr=True, pause=False)

                plt.pause(0.001)

        except Exception as e:
            print(e)
            break


# Example usage:
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plot_world_landmarks(plt, ax, your_mediapipe_landmarks)


if __name__ == "__main__":
    datasetpath = "../gte9_landmarks"

    arr_path = "./gte9_landmarks/animal/02583.npy"
    arr_path = "./smaller_dataset_landmarks/about/00414.npy"
    arr_path = "./gte9_test_vid/bed.npy"
    arrr = np.load(arr_path)

    infinite_plot_landmarks(arr="./gte9_landmarks/baby/04505.npy")
    exit()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    num_frames = 70

    def stickyhands(NP_LANDMARK_ALL_FRAMES, total_num_frames=num_frames):
        lastknownleft = np.zeros((21, 4))
        lastknownright = np.zeros((21, 4))
        allzeros = np.zeros((21, 4))

        for fram in range(total_num_frames):
            left = NP_LANDMARK_ALL_FRAMES[fram][21:42]
            right = NP_LANDMARK_ALL_FRAMES[fram][42:63]

            if (not np.all(lastknownleft == allzeros)) and np.all(left == allzeros):
                NP_LANDMARK_ALL_FRAMES[fram][21:42] = lastknownleft
            elif np.all(lastknownleft == allzeros) and (not np.all(left == allzeros)):
                lastknownleft = left

            if (not np.all(lastknownright == allzeros)) and np.all(right == allzeros):
                NP_LANDMARK_ALL_FRAMES[fram][42:63] = lastknownright
            elif np.all(lastknownright == allzeros) and (not np.all(right == allzeros)):
                lastknownright = right

    # stickyhands(arrr)

    # for i in range(70):
    #     print(arrr[i][42:63])
    # print(arrr.shape)

    # exit()
    while True:
        for fr in range(70):
            # print(arrr[fr])
            # print(arrr[fr].shape)
            plot_landmarks(plt, ax, arrr[fr])

    for word in os.listdir(datasetpath):
        word_path = os.path.join(datasetpath, word)
        for arrs in os.listdir(word_path):
            arr_path = os.path.join(word_path, arrs)
            arrr = np.load(arr_path)

            print(arrr.shape)

            for fr in range(70):
                # print(arrr[fr])
                # print(arrr[fr].shape)
                plot_landmarks(plt, ax, arrr[fr])
            exit()
