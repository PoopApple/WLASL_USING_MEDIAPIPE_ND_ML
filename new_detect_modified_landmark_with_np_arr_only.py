"""https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md
https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md#model_complexity
"""

"""
MARKERS

#REPLACED PYTHON LISTS WITH NUMPY ARRAYS
NOTE numpy not used


"""


"""FALTU WARNIGN HATAO"""

# os.environ['GLOG_minloglevel'] = '2'  # Suppresses INFO and WARNING
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import os, sys

# Redirect stderr to null at the OS level BEFORE mediapipe loads
stderr_fd = sys.stderr.fileno()
null_fd = open(os.devnull, "w")
os.dup2(null_fd.fileno(), stderr_fd)




os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["GLOG_minloglevel"] = "3"        # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # Hide TF Lite warnings too
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ["ABSL_LOGGING_THRESHOLD"] = "FATAL"


# import absl.logging
# absl.logging.set_verbosity(absl.logging.ERROR)
# absl.logging.set_stderrthreshold('fatal')


import cv2


import mediapipe as mp


import numpy as np
import multiprocessing as multip
import gc


NORMALISED_FRAMES = 70
SHAPE_OF_LANDMARK_IN_ONE_FRAME = (63, 4)


def doshit(
    vid_path="./smaller_dataset/above/00430.mp4",
    showvid=False,
    dopythonlists=False,
    printdebugdata=False,
    normalisearray=True,
    solidbg=False
):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    """https://camo.githubusercontent.com/d3afebfc801ee1a094c28604c7a0eb25f8b9c9925f75b0fff4c8c8b4871c0d28/68747470733a2f2f6d65646961706970652e6465762f696d616765732f6d6f62696c652f706f73655f747261636b696e675f66756c6c5f626f64795f6c616e646d61726b732e706e67
    https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

    lower right torso 24
    lower left torso 23
    up r t 12
    up l t 11 


    arm left 13 15
    right 14 16



    face-
    lips = 9 10
    nos = 0


    ears = 7 8
    eyes = 5 2
    
    
    
    
    
    ADDING HANDS to poses SINCE HAND LANDMARKS ARE NOT ALWAYS VISIBLE DUE TO OVERLAPPING
    17 SE 22
    
    17,18,19,20,21,22
    
    
    

    """
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
        15,
        14,
        16,
        23,
        24,
        17,
        18,
        19,
        20,
        21,
        22,
    ]  # 15+6 = 21

    # For webcam input:
    cap = cv2.VideoCapture(vid_path)

    hands = mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.2,
        max_num_hands=2,
    )
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.2,
    )

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    """
    70 frames
    (57+6)x4 landmarks = 63x4 landmarks
    32bit each float = 4byte
    
    1 vid = print(70*63*4*4) = 70560 bytes = print(70560/1024) = 68.90625 KB
    
    11980vids = print(68.90625 * 11980) = 825496.875 KB     
    
    we use float32 bcuz tensorflow is optimised for it and not for float64
    
    
    
    BUT WE ALSO DISCARDED SOME VIDS, SO 11980 IS NOT THE FINAL
    
    """

    NP_LANDMARK_ALL_FRAMES = np.zeros(
        (total_num_frames, *SHAPE_OF_LANDMARK_IN_ONE_FRAME), dtype=np.float32
    )  # mediapipe xyz float64 me dera h
    frame_index = 0

    landmark_for_all_frames = []

        
    bad_torso = False
    while cap.isOpened():

        if cv2.waitKey(5) & 0xFF == 27:
            break

        success, image = cap.read()
        # success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        result_pose = pose.process(image)

        if showvid:
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            
            
            if solidbg:
                blue_color = (255, 0, 0) # BGR: Blue=255, Green=0, Red=0

                # Fill the entire image with the specified color
                image[:] = blue_color

        landmark_for_one_frame = []
        """ format = all 15+6 = 21 poses(face and arms and wrists and torso) then left hand then right hand"""
        if result_pose.pose_landmarks:
            # print(f"pose landmarks: {len(result_pose.pose_landmarks.landmark)}")

            # print(f"nose: {result_pose.pose_landmarks.landmark[0]}")
            # print(f"le ear: {result_pose.pose_landmarks.landmark[7]}")
            # print(f"ri ear: {result_pose.pose_landmarks.landmark[8]}")

            """
            needed_poses = [0,2,5,7,8,9,10,11,12,13,15,14,16,23,24,17-22] #15+6 = 21

            array of landmarks need --  21x4 = 21landmarks with 4 dimensions [x y z visibility]
            """

            CENTER_OF_SHOULDER = [
                (
                    result_pose.pose_landmarks.landmark[11].x
                    + result_pose.pose_landmarks.landmark[12].x
                )
                / 2,
                (
                    result_pose.pose_landmarks.landmark[11].y
                    + result_pose.pose_landmarks.landmark[12].y
                )
                / 2,
                (
                    result_pose.pose_landmarks.landmark[11].z
                    + result_pose.pose_landmarks.landmark[12].z
                )
                / 2,
            ]

            # print(f"11:\t{result_pose.pose_landmarks.landmark[11]}")
            # print(f"12:\t{result_pose.pose_landmarks.landmark[12]}")
            # print(CENTER_OF_SHOULDER)

            """TODO try scale with shoulder length 
            
            
            mediapipe predicts the position of hips if not visible..... only visibiltiy changes
            and so y becomes >1 whichmeans not in the actual img
            
            """

            CENTER_OF_HIPS = [
                (
                    result_pose.pose_landmarks.landmark[23].x
                    + result_pose.pose_landmarks.landmark[24].x
                )
                / 2,
                (
                    result_pose.pose_landmarks.landmark[23].y
                    + result_pose.pose_landmarks.landmark[24].y
                )
                / 2,
                (
                    result_pose.pose_landmarks.landmark[23].z
                    + result_pose.pose_landmarks.landmark[24].z
                )
                / 2,
            ]

            # print(f"23:\t{result_pose.pose_landmarks.landmark[23]}")
            # print(f"24:\t{result_pose.pose_landmarks.landmark[24]}")

            # print(CENTER_OF_HIPS)

            # print(CENTER_OF_SHOULDER == [0, 0, 0], CENTER_OF_SHOULDER)
            # print(CENTER_OF_HIPS == [0, 0, 0], CENTER_OF_HIPS)

            diff_s = [
                (
                    result_pose.pose_landmarks.landmark[11].x
                    - result_pose.pose_landmarks.landmark[12].x
                ),
                (
                    result_pose.pose_landmarks.landmark[11].y
                    - result_pose.pose_landmarks.landmark[12].y
                ),
            ]
            SHOULDER_LENGTH = np.linalg.norm(diff_s)

            diff_t = [CENTER_OF_HIPS[i] - CENTER_OF_SHOULDER[i] for i in [0, 1]]
            # print("diff t:\t\t",diff_t)
            TORSO_HEIGHT = np.linalg.norm(diff_t)
            
            
            if printdebugdata:
                print("Shoulder:\t",SHOULDER_LENGTH)
                print("torso he:\t",TORSO_HEIGHT)
                
            if not bad_torso and TORSO_HEIGHT < 0.31:
                bad_torso=True

            # pose_

            """TODO TODO TODO TODO
            
            now divide x by should len and y by trso h
            
            """

            for landmark_index, i in enumerate(needed_poses):
                xx = result_pose.pose_landmarks.landmark[i].x - CENTER_OF_SHOULDER[0]
                yy = result_pose.pose_landmarks.landmark[i].y - CENTER_OF_SHOULDER[1]
                zz = result_pose.pose_landmarks.landmark[i].z - CENTER_OF_SHOULDER[2]
                vv = result_pose.pose_landmarks.landmark[i].visibility
                
                
                NP_LANDMARK_ALL_FRAMES[frame_index, landmark_index] = [
                    xx/SHOULDER_LENGTH,
                    yy/TORSO_HEIGHT,
                    zz,
                    vv
                ]

            """NOTE numpy not used"""
            if dopythonlists:
                
                landmark_for_one_frame.extend(
                    [
                        [
                            (result_pose.pose_landmarks.landmark[i].x
                            - CENTER_OF_SHOULDER[0]) / SHOULDER_LENGTH,
                            (result_pose.pose_landmarks.landmark[i].y
                            - CENTER_OF_SHOULDER[1]) / TORSO_HEIGHT,
                            result_pose.pose_landmarks.landmark[i].z
                            - CENTER_OF_SHOULDER[2],
                            result_pose.pose_landmarks.landmark[i].visibility,
                        ]
                        for i in needed_poses
                    ]
                )

            # print(landmark_for_one_frame)
            # print(NP_LANDMARK_ALL_FRAMES[0][:15])
            # print(landmark_for_one_frame  == NP_LANDMARK_ALL_FRAMES[0][:15] )

            # break
            # print(landmark_for_one_frame[0][0])
            # print(NP_LANDMARK_ALL_FRAMES[0][0][0])

            # np_pose_landmark_for_one_frame = np.array(pose_landmark_for_one_frame,dtype=np.float32)
            # print(np_pose_landmark_for_one_frame.shape) ==== 15x4
            if showvid:
                mp_drawing.draw_landmarks(
                    image,
                    result_pose.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )
        else:
            """NOTE numpy not used"""
            if dopythonlists:
                landmark_for_one_frame.extend([[0, 0, 0, 0] for i in needed_poses])

        """https://stackoverflow.com/questions/67455791/mediapipe-python-link-landmark-with-handedness"""

        if results.multi_hand_landmarks:
            """NOTE numpy not used"""
            if dopythonlists:
                hand_landmarks_list = {"Left": [], "Right": []}

            # print(len(results.multi_hand_landmarks))

            # for handedness in results.multi_handedness:
            #     # print(handedness)
            #     idx = handedness.classification[0].index
            #     # print(idx)

            # if len(results.multi_hand_landmarks) == 2:
            #         print(results.multi_hand_landmarks)

            """
            0-14+6 = pose = 0-20
            15+6-35+6 = lefthand = 21-41
            36+6-56+6 = righthand = 42-62
            
            
            """

            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                lbl = results.multi_handedness[idx].classification[0].label

                landmark_index = 21 if lbl == "Left" else 42

                for i in range(21):
                    NP_LANDMARK_ALL_FRAMES[frame_index][landmark_index + i] = [
                        (hand_landmarks.landmark[i].x - CENTER_OF_SHOULDER[0])/SHOULDER_LENGTH,
                        (hand_landmarks.landmark[i].y - CENTER_OF_SHOULDER[1])/TORSO_HEIGHT,
                        hand_landmarks.landmark[i].z - CENTER_OF_SHOULDER[2],
                        1,
                    ]

                # print()
                # print(lbl)

                # print(hand_landmarks)
                # print(f"hand landmarks: {len(hand_landmarks.landmark)}")
                # for ind in range(21):
                # print([hand_landmarks.landmark[ind].x , hand_landmarks.landmark[ind].y , hand_landmarks.landmark[ind].z , 1])
                """NOTE numpy not used """
                if dopythonlists:
                    hand_landmarks_list[lbl] = [
                        [
                            hand_landmarks.landmark[ind].x - CENTER_OF_SHOULDER[0],
                            hand_landmarks.landmark[ind].y - CENTER_OF_SHOULDER[1],
                            hand_landmarks.landmark[ind].z - CENTER_OF_SHOULDER[2],
                            1,
                        ]
                        for ind in range(21)
                    ]
                # print(hand_landmarks_list)

                # print(hand_landmarks.landmark[0])
                """
                NOTE
                NOTE
                NOTE
                NOTE
                
                done --- ADD LEFT HAND LANDMARKS TO LEFTHAND LIST AND RIGHT TO RIGHT HAND USING IF ELSE OR WHATEVER
                
                
                THEN CHECK ALL DIMESNIONS
                total landmarks = 15+21+21 = 57   +6  = 63
                each has x,y,z,visibilty 
                
                done --- shape of each frame data = 57+6x4 ====> 63X4
                
                
                THEN NORMALISE DATA TO 70 FRAMES USING LIN SPACE
                
                THEN STORE IN NPY
                
                
                THEN RUN LTSM
                
                THEN TRY OUT 3DCNN
                
                
                
                
                """

                if showvid:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

            """NOTE numpy not used """
            if dopythonlists:
                if hand_landmarks_list["Left"] == []:
                    landmark_for_one_frame.extend([[0, 0, 0, 0] for i in range(21)])
                    # print([[0,0,0,0] for i in range(21)])
                    # print(landmark_for_one_frame)
                else:
                    landmark_for_one_frame.extend(hand_landmarks_list["Left"])

                if hand_landmarks_list["Right"] == []:
                    landmark_for_one_frame.extend([[0, 0, 0, 0] for i in range(21)])
                    # print([[0,0,0,0] for i in range(21)])
                    # print(landmark_for_one_frame)
                else:
                    landmark_for_one_frame.extend(hand_landmarks_list["Right"])

            # print([[0,0,0,0] for i in range(21)])
            # print(landmark_for_one_frame)
            # print(len(landmark_for_one_frame))

        else:
            """NOTE numpy not used"""
            if dopythonlists:
                landmark_for_one_frame.extend([[0, 0, 0, 0] for i in range(21)])
                landmark_for_one_frame.extend([[0, 0, 0, 0] for i in range(21)])

        """NOTE numpy not used """
        if dopythonlists:
            landmark_for_all_frames.extend([landmark_for_one_frame])

        # print(landmark_for_one_frame[15:])
        # print(NP_LANDMARK_ALL_FRAMES[0][15:])
        # print((landmark_for_one_frame[15:]  == NP_LANDMARK_ALL_FRAMES[0][15:]).all() )
        #
        # break
        # print(landmark_for_one_frame[0][0])
        # print(NP_LANDMARK_ALL_FRAMES[0][0][0])

        if printdebugdata:
            for i in range(63):
                print(f"{i} =  {NP_LANDMARK_ALL_FRAMES[frame_index][i]}")

        frame_index += 1

        if showvid:
            original_height, original_width = image.shape[:2]

            # Define new width while maintaining the aspect ratio
            new_width = 800
            aspect_ratio = new_width / original_width
            # Compute height based on aspect ratio
            new_height = int(original_height * aspect_ratio)

            # Resize the image
            resized_image = cv2.resize(cv2.flip(image, 1), (new_width, new_height))

            # Display the resized image
            cv2.imshow("Resized Image", resized_image)

            # cv2.imshow("MediaPipe Hands", cv2.flip(image, 1))

    cap.release()
    cv2.destroyAllWindows()
    
    
    if bad_torso:
        with open("./stats/bad_torso.txt","a") as f:
            f.write(vid_path)

    """
    medipipe uses internal cpp calculator or smthng so each usage increases ram usage by 150MB or so
    
    toh harr call ke baad destroy krna hota
    
    ek soln h ki common mediapipe use krlo 
    
    pr fir multi processing nhi ho payegi
    """

    hands.close()
    pose.close()

    gc.collect()

    if dopythonlists:
        print("SAHI AAYA:")
        print((landmark_for_all_frames == NP_LANDMARK_ALL_FRAMES).all())

    # print(total_num_frames)
    # print(len(landmark_for_all_frames))

    def normalise_array(nparr, normalised_frames=NORMALISED_FRAMES):
        linear_indices = np.linspace(0, len(nparr) - 1, normalised_frames, dtype=int)

        NP_NORMALISED_LANDMARK_ALL_FRAMES = nparr[linear_indices]

        return NP_NORMALISED_LANDMARK_ALL_FRAMES

    if normalisearray:
        return (total_num_frames, vid_path, normalise_array(NP_LANDMARK_ALL_FRAMES))
    else:
        return (total_num_frames, vid_path, NP_LANDMARK_ALL_FRAMES)


def process_one(input_path, output_path):
    total_num_frames = vid_path = NP_NORMALISED_LANDMARK_ALL_FRAMES = None
    try:
        total_num_frames, vid_path, NP_NORMALISED_LANDMARK_ALL_FRAMES = doshit(
            input_path
        )
        np.save(output_path, NP_NORMALISED_LANDMARK_ALL_FRAMES)
        print(f"Saved:\t{output_path}")
    except KeyboardInterrupt:
        print("exit")
    except Exception as e:
        print(f"Bhai <{input_path}> me error agaya: {e}")
    finally:
        del total_num_frames, vid_path, NP_NORMALISED_LANDMARK_ALL_FRAMES
        gc.collect()


"""
import os
print(os.cpu_count())

12??????

logical processors h 12
"""


def process_all(
    input_folder, output_folder, printdebug=True, num_workers=os.cpu_count() - 1
):
    os.makedirs(output_folder, exist_ok=True)

    ip_op_path_list_of_tasks = []

    for word_folder in os.listdir(input_folder):
        if printdebug:
            print()
            print(word_folder)

        input_word_folder = os.path.join(input_folder, word_folder)
        output_word_folder = os.path.join(output_folder, word_folder)

        os.makedirs(output_word_folder, exist_ok=True)

        for filename in os.listdir(input_word_folder):

            if filename.lower().endswith(".mp4"):
                input_path = os.path.join(input_word_folder, filename)
                output_path = os.path.join(output_word_folder, filename[:-4] + ".npy")
                if printdebug:
                    print(input_path)
                    print(output_path)

                if not os.path.exists(output_path):
                    ip_op_path_list_of_tasks.append((input_path, output_path))
                    # process_one(input_path,output_path)
                else:
                    if printdebug:
                        print("already saved")

    """
    each processvid meri 200-300MB ram leta h
    agar mere paas 1000vids h
    aur num worker 11 h
    
    toh
    
    naaa
    
    
    agar mere paas 11 workers h toh total ram use hogi 11*300 jo h 3GB
    
    agar mujhe 2 rakhni h toh      2000/300 = 7
    
    
    10 dunga toh 3GB
    
    """

    """iske binna bhai band nhi hora tha crazyyyy
    
    exit nhi hora
    
    task manager se kill krna padhra h
    
    koi nhi
    
    """
    try:
        with multip.Pool(processes=num_workers) as pool:
            pool.starmap(process_one, ip_op_path_list_of_tasks, chunksize=2)
    except KeyboardInterrupt:
        print("exiiitttt")


if __name__ == "__main__":

    """
    agar model abhi bhi galat ho
    
    toh masking try for start and end 10% kyuki usme haath position pe aare hote
    
    """
    
    
    
    
    # doshit(vid_path="./dataset/across/00832.mp4", showvid=True, printdebugdata=False)
    # doshit(vid_path="./dataset/across/00834.mp4", showvid=True)
    # doshit(vid_path="./dataset/across/00836.mp4", showvid=True)

    dataset_folder = ".\\smaller_dataset"
    np_output_folder = ".\\smaller_dataset_landmarks"
    process_all(dataset_folder,np_output_folder,num_workers=10)

    exit()

    input_path = "./test_vid/about.mp4"
    output_path = "./test_vid/adout.npy"

    total_num_frames = vid_path = NP_NORMALISED_LANDMARK_ALL_FRAMES = None
    try:
        total_num_frames, vid_path, NP_NORMALISED_LANDMARK_ALL_FRAMES = doshit(
            vid_path=input_path, showvid=True
        )
        np.save(output_path, NP_NORMALISED_LANDMARK_ALL_FRAMES)

    finally:
        del total_num_frames, vid_path, NP_NORMALISED_LANDMARK_ALL_FRAMES
        gc.collect()


# total_num_frames2,landmark_for_all_frames2 = doshit(showvid=True)


# print(landmark_for_all_frames == landmark_for_all_frames2)


# linear_indices = np.linspace(0,total_num_frames-1,NORMALISED_FRAMES,dtype=int)

# print(linear_indices)


# normalised_landmark_for_all_frames = landmark_for_all_frames[linear_indices.tolist()]

# print(normalised_landmark_for_all_frames)
# print(len(normalised_landmark_for_all_frames))
