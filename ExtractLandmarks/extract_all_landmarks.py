import os

# # Redirect stderr to null at the OS level BEFORE mediapipe loads
# stderr_fd = sys.stderr.fileno()
# null_fd = open(os.devnull, "w")
# os.dup2(null_fd.fileno(), stderr_fd)


import datetime

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GLOG_minloglevel"] = "3"  # 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide TF Lite warnings too
# os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
os.environ["ABSL_LOGGING_THRESHOLD"] = "FATAL"


# import absl.logging
# absl.logging.set_verbosity(absl.logging.ERROR)
# absl.logging.set_stderrthreshold('fatal')


import cv2

import time
import mediapipe as mp
from mediapipe.tasks.python import vision
# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2

# import matplotlib.pyplot as plt


import numpy as np
import multiprocessing as multip
import gc

# from visualise_lms_in_3d import infinite_plot_landmarks


from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

# NORMALISED_FRAMES = 70
# SHAPE_OF_LANDMARK_IN_ONE_FRAME = (63, 4)


pose_model_path = "./vision_models/pose_landmarker_heavy.task"
hand_model_path = "./vision_models/hand_landmarker.task"


"""https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md
https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md#model_complexity
"""

"""

W0000 00:00:1763102308.732913   31240 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1763102308.762379   23596 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.

"""

"""
TODO
TODO

now try flipping the videos too



"""


"""
MARKERS

#REPLACED PYTHON LISTS WITH NUMPY ARRAYS
NOTE numpy not used


"""


"""FALTU WARNIGN HATAO"""

# os.environ['GLOG_minloglevel'] = '2'  # Suppresses INFO and WARNING
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/pose_landmarker/python/[MediaPipe_Python_Tasks]_Pose_Landmarker.ipynb#scrollTo=s3E6NFV-00Qt"


def draw_pose_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    for pose_landmarks in pose_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style,
        )

    return annotated_image


#
# def draw_pose_landmarks_on_image(rgb_image, detection_result):
#   pose_landmarks_list = detection_result.pose_landmarks
#   annotated_image = np.copy(rgb_image)
#
#   # Loop through the detected poses to visualize.
#   for idx in range(len(pose_landmarks_list)):
#     pose_landmarks = pose_landmarks_list[idx]
#
#     # Draw the pose landmarks.
#     pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#     pose_landmarks_proto.landmark.extend([
#       landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
#     ])
#     solutions.drawing_utils.draw_landmarks(
#       annotated_image,
#       pose_landmarks_proto,
#       solutions.pose.POSE_CONNECTIONS,
#       solutions.drawing_styles.get_default_pose_landmarks_style())
#   return annotated_image
#
#
# import mediapipe as mp
# import numpy as np
#


mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

"https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=s3E6NFV-00Qt&uniqifier=1"


def draw_hands_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image


#
# MARGIN = 10  # pixels
# FONT_SIZE = 1
# FONT_THICKNESS = 1
# HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
# "https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=s3E6NFV-00Qt&uniqifier=1"
# def draw_hands_landmarks_on_image(rgb_image, detection_result):
#   hand_landmarks_list = detection_result.hand_landmarks
#   handedness_list = detection_result.handedness
#   annotated_image = np.copy(rgb_image)
#
#   # Loop through the detected hands to visualize.
#   for idx in range(len(hand_landmarks_list)):
#     hand_landmarks = hand_landmarks_list[idx]
#     handedness = handedness_list[idx]
#
#     # Draw the hand landmarks.
#     hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#     hand_landmarks_proto.landmark.extend([
#       landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
#     ])
#     solutions.drawing_utils.draw_landmarks(
#       annotated_image,
#       hand_landmarks_proto,
#       solutions.hands.HAND_CONNECTIONS,
#       solutions.drawing_styles.get_default_hand_landmarks_style(),
#       solutions.drawing_styles.get_default_hand_connections_style())
#
#     # Get the top left corner of the detected hand's bounding box.
#     height, width, _ = annotated_image.shape
#     x_coordinates = [landmark.x for landmark in hand_landmarks]
#     y_coordinates = [landmark.y for landmark in hand_landmarks]
#     text_x = int(min(x_coordinates) * width)
#     text_y = int(min(y_coordinates) * height) - MARGIN
#
#     # Draw handedness (left or right hand) on the image.
#     cv2.putText(annotated_image, f"{handedness[0].category_name}",
#                 (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
#                 FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
#
#   return annotated_image
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


def doshit(
    vid_path="/media/aryan/PoopiDrive/projects_linux/microsoft asl citizen/ASL_Citizen/videos/8HOUR_07243138866452936-8 HOUR.mp4",
    showvid=False,
    printdebugdata=False,
    solidbg=False,
):

    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

    # Create a hand landmarker instance with the video mode:
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=hand_model_path  # , delegate=BaseOptions.Delegate.GPU
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
    )

    # Create a pose landmarker instance with the video mode:
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=pose_model_path,  # delegate=BaseOptions.Delegate.GPU
        ),
        running_mode=VisionRunningMode.VIDEO,
    )

    with (
        PoseLandmarker.create_from_options(pose_options) as pose_landmarker,
        HandLandmarker.create_from_options(hand_options) as hand_landmarker,
    ):
        # num_of_poses=33

        cap = cv2.VideoCapture(vid_path)

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
        """
                0-32 => poses
                33-53 => left hand
                54-74 => left hand

                total = 75

        """
        SHAPE_OF_LANDMARK_IN_ONE_FRAME = (75, 4)
        NP_LANDMARK_ALL_FRAMES = np.zeros(
            (total_num_frames + 1, *SHAPE_OF_LANDMARK_IN_ONE_FRAME), dtype=np.float32
        )  # mediapipe xyz float64 me dera h
        frame_index = 1

        """
            first frame can store metadata like  fps frames etc

        """

        NP_LANDMARK_ALL_FRAMES[0][0] = total_num_frames
        NP_LANDMARK_ALL_FRAMES[0][1] = fps

        # landmark_for_all_frames = []

        bad_torso = False

        """ek baar krna padha kyukiwarna time -40ms aara"""
        cap.read()
        while cap.isOpened():
            if cv2.waitKey(5) & 0xFF == 27:
                break

            success, image = cap.read()
            # cv2.imshow("Image", image)

            """https://stackoverflow.com/questions/47743246/getting-timestamp-of-each-frame-in-a-video"""
            curr_frame_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            # print(f"time in ms\t{curr_frame_ms}")

            # success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

            pose_landmarker_result = pose_landmarker.detect_for_video(
                mp_image, curr_frame_ms
            )

            # ops = pose_landmarker_result.pose_landmarks[0]

            # for lm in ops:
            #     print(lm)

            # print(len(ops))

            # results = hands.process(image)
            # result_pose = pose.process(image)

            if showvid:
                # Draw the hand annotations on the image.
                image.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if solidbg:
                    blue_color = (255, 0, 0)  # BGR: Blue=255, Green=0, Red=0

                    # Fill the entire image with the specified color
                    image[:] = blue_color

            if pose_landmarker_result.pose_landmarks:
                arr = pose_landmarker_result.pose_landmarks[0]

                # for i in range(33):
                #     print(arr[i].x)

                # print(f"pose landmarks: {len(pose_landmarker_result.pose_landmarks)}")

                # print(f"nose: {pose_landmarker_result.pose_landmarks[0]}")
                # print(f"le ear: {pose_landmarker_result.pose_landmarks[7]}")
                # print(f"ri ear: {pose_landmarker_result.pose_landmarks[8]}")

                """
                --needed_poses = [0,2,5,7,8,9,10,11,12,13,15,14,16,23,24,17-22] #15+6 = 21--

                --array of landmarks need --  21x4 = 21landmarks with 4 dimensions [x y z visibility]--

                taking all 33 poses for now




                """
                for i in range(33):
                    xx = arr[i].x
                    yy = arr[i].y
                    zz = arr[i].z
                    vv = arr[i].visibility

                    NP_LANDMARK_ALL_FRAMES[frame_index, i] = [xx, yy, zz, vv]

                if showvid:
                    annotated_image = draw_pose_landmarks_on_image(
                        image, pose_landmarker_result
                    )

            """https://stackoverflow.com/questions/67455791/mediapipe-python-link-landmark-with-handedness"""

            hand_landmarker_result = hand_landmarker.detect_for_video(
                mp_image, curr_frame_ms
            )

            # print(hand_landmarker_result)
            """HandLandmarkerResult(handedness=[], hand_landmarks=[], hand_world_landmarks=[])"""

            # if False:
            if hand_landmarker_result.hand_landmarks:
                """NOTE numpy not used"""

                # print(hand_landmarker_result)

                for idx, handedd in enumerate(hand_landmarker_result.handedness):
                    handedd = handedd[0]
                    lbl = handedd.display_name

                    # print("aa")
                    # print(lbl)
                    # print(idx)

                    handLMs = hand_landmarker_result.hand_landmarks[idx]

                    landmark_index = 33 if lbl == "Left" else 54

                    for i in range(21):
                        NP_LANDMARK_ALL_FRAMES[frame_index][landmark_index + i] = [
                            handLMs[i].x,
                            handLMs[i].y,
                            handLMs[i].z,
                            1,
                        ]

                    # print()
                    # print(lbl)

                    # print(hand_landmarks)
                    # print(f"hand landmarks: {len(hand_landmarks.landmark)}")
                    # for ind in range(21):
                    # print([hand_landmarks.landmark[ind].x , hand_landmarks.landmark[ind].y , hand_landmarks.landmark[ind].z , 1])
                    """NOTE numpy not used """
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
                        annotated_image = draw_hands_landmarks_on_image(
                            annotated_image, hand_landmarker_result
                        )

            # print([[0,0,0,0] for i in range(21)])
            # print(landmark_for_one_frame)
            # print(len(landmark_for_one_frame))

            # print(landmark_for_one_frame[15:])
            # print(NP_LANDMARK_ALL_FRAMES[0][15:])
            # print((landmark_for_one_frame[15:]  == NP_LANDMARK_ALL_FRAMES[0][15:]).all() )
            #
            # break
            # print(landmark_for_one_frame[0][0])
            # print(NP_LANDMARK_ALL_FRAMES[0][0][0])

            if printdebugdata:
                for i in range(75):
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
                resized_image = cv2.resize(annotated_image, (new_width, new_height))

                cv2.imshow("a", cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR))
                # Display the resized image
                # cv2.imshow("Resized Image", resized_image)

                # cv2.imshow("MediaPipe Hands", cv2.flip(image, 1))

        cap.release()
        cv2.destroyAllWindows()

        if bad_torso:
            with open("./stats/bad_torso.txt", "a") as f:
                f.write(vid_path)

    """
    medipipe uses internal cpp calculator or smthng so each usage increases ram usage by 150MB or so
    
    toh harr call ke baad destroy krna hota
    
    ek soln h ki common mediapipe use krlo 
    
    pr fir multi processing nhi ho payegi
    """

    # hands.close()
    # pose_landmarker.close()
    # hand_landmarker.close()

    gc.collect()

    # print(total_num_frames)
    # print(len(landmark_for_all_frames))

    # if show3d:
    #     infinite_plot_landmarks(NP_LANDMARK_ALL_FRAMES)

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
        print(f"{output_path} done!!")
        del total_num_frames, vid_path, NP_NORMALISED_LANDMARK_ALL_FRAMES
        gc.collect()


"""
import os
print(os.cpu_count())

12??????

logical processors h 12
"""


def process_all(
    input_folder, output_folder, printdebug=True, num_workers=2, maxlimit=None
):

    with open("processing.logs", "a") as logf:
        logf.write(
            f"\n\nStart:\t\t{datetime.datetime.now().strftime(format='%d/%m/%Y, %H:%M:%S')}\nMaxLimit:{maxlimit}"
        )
    os.makedirs(output_folder, exist_ok=True)

    ip_op_path_list_of_tasks = []
    processed = 0
    for filename in os.listdir(input_folder):
        if maxlimit and processed < maxlimit:
            pass
        else:
            break

        if filename.lower().endswith(".mp4"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename[:-4] + ".npy")
            if printdebug:
                print(input_path)
                print(output_path)

            if not os.path.exists(output_path):
                ip_op_path_list_of_tasks.append((input_path, output_path))
                processed += 1
                # process_one(input_path,output_path)
            else:
                if printdebug:
                    print("already saved")
    pool = None
    try:
        with multip.Pool(processes=num_workers, maxtasksperchild=1) as pool:
            # chunksize=1 ensures videos are handed out one-by-one to the fresh workers
            pool.starmap(process_one, ip_op_path_list_of_tasks, chunksize=1)
        with open("processing.logs", "a") as logf:
            logf.write(
                f"\n\nEnd:\t\t{datetime.datetime.now().strftime(format='%d/%m/%Y, %H:%M:%S')}\nProcessed:{processed}"
            )

    except KeyboardInterrupt:
        print("\n\n\nSTOPPPPP PLSSSS!!!!!")
        if pool:
            pool.terminate()
            pool.join()

        with open("processing.logs", "a") as logf:
            logf.write(
                f"\n\nError:\t\t{datetime.datetime.now().strftime(format='%d/%m/%Y, %H:%M:%S')}\nProcessed:{processed}"
            )

    # try:
    #     with multip.Pool(processes=num_workers) as pool:
    #         pool.starmap(process_one, ip_op_path_list_of_tasks, chunksize=2)
    # except KeyboardInterrupt:
    #     print("exiiitttt")
    #
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


def blahblah():
    input_paths = [
        "./gte9_test_vid/ball.mp4",
        "./gte9_test_vid/cat.mp4",
        "./gte9_test_vid/bed.mp4",
    ]
    input_paths = ["./gte9_test_vid/country.mp4"]
    output_paths = [x[:-4] + ".npy" for x in input_paths]

    for input_path, output_path in zip(input_paths, output_paths):
        total_num_frames = vid_path = NP_NORMALISED_LANDMARK_ALL_FRAMES = None
        try:
            total_num_frames, vid_path, NP_NORMALISED_LANDMARK_ALL_FRAMES = doshit(
                vid_path=input_path, showvid=True
            )
            np.save(output_path, NP_NORMALISED_LANDMARK_ALL_FRAMES)

        finally:
            del total_num_frames, vid_path, NP_NORMALISED_LANDMARK_ALL_FRAMES
            gc.collect()


def measure_time(func, *args, **kwargs):
    start = time.perf_counter()

    result = func(*args, **kwargs)

    end = time.perf_counter()

    print(f"Time taken: {(end - start) * 1000:.3f} ms")

    return result


def check200vids(input_folder="/run/media/aryan/PoopiDrive/projects_linux/microsoft asl citizen/ASL_Citizen/videos/"):
    
    print(datetime.datetime.now().strftime("%H:%M:%S"))
    
    for filename in os.listdir(input_folder)[:200]:
        
        if filename.lower().endswith(".mp4"):
            input_path = os.path.join(input_folder, filename)
            doshit(vid_path=input_path)
    
    print(datetime.datetime.now().strftime("%H:%M:%S"))
    


if __name__ == "__main__":
    
    # start = 03:19:03
    # end = 03:36:48
    # Time taken: 1065220.291 ms
    
    # measure_time(check200vids)
    
    # timeofwords200 = 1065220 
    # timeofword1 = timeofwords200/200
    
    # numofvids = 83399 
    
    # timeoftotal = numofvids*timeofword1
    
    # insec = timeoftotal/1000
    # inmin = insec/60
    # inhour = inmin/60
    # indays = inhour/24
    
    # print(f"inmin {inmin}")
    # print(f"inhours {inhour}")
    # print(f"indays {indays}")
     
    # exit()
    """
    agar model abhi bhi galat ho
    
    toh masking try for start and end 10% kyuki usme haath position pe aare hote
    
    """
    doshit("/run/media/aryan/PoopiDrive/projects_linux/microsoft asl citizen/ASL_Citizen/videos/BASKETBALL1_57337857006252-BASKETBALL.mp4", showvid=True)
    # process_one("./test_actor.webm", "./test_actor.npy")
    exit()
    # exit()
    # vidpath = "./TestingVids/ACCENT_674051280092625-ACCENT.mp4"
    # vidpath = "./TestingVids/ACCESS_20630301986906696-ACCESS.mp4"
    # vidpath = "./TestingVids/ACCENT_006466386479419883-ACCENT.mp4"
    # doshit(vid_path=vidpath,
    #        showvid=True,
    #        printdebugdata=   True)
    # doshit(vid_path="./dataset/across/00832.mp4", showvid=True, fliptheimg=True)
    # doshit(vid_path="./dataset/across/00834.mp4", showvid=True)
    # doshit(vid_path="./dataset/bed/05634.mp4",show3d=True,showvid=True)
    # doshit(vid_path="./gte9_dataset/ball/04852.mp4", showvid=True,show3d=True)

    # exit()
    dataset_folder = "./TestingVids/dataset_testing_248vids/"
    dataset_folder = "/media/aryan/PoopiDrive/projects_linux/microsoft asl citizen/ASL_Citizen/videos/"
    np_output_folder = "./MS_ASL_Result/"
    process_all(dataset_folder, np_output_folder, num_workers=5, maxlimit=20000)
    
    
    """
    1hour approx = 2000vids
    on 6 workers
    
    11hours = 22000vids !!!!!
    """
    
    
    # doshit("./TestingVids/dataset_testing_248vids/1DOLLAR_023931338852502426-1 DOLLAR.mp4")
    # doshit("./TestingVids/dataset_testing_248vids/1DOLLAR_023931338852502426-1 DOLLAR.mp4")
    # doshit("./TestingVids/dataset_testing_248vids/1DOLLAR_023931338852502426-1 DOLLAR.mp4")
    # doshit("./TestingVids/dataset_testing_248vids/1DOLLAR_023931338852502426-1 DOLLAR.mp4")
    # doshit("./TestingVids/dataset_testing_248vids/5DOLLARS_16319399751609986-5 DOLLARS.mp4")
    # doshit("./TestingVids/dataset_testing_248vids/1DOLLAR_09105279664955357-1 DOLLAR.mp4")
    # doshit("./TestingVids/dataset_testing_248vids/1DOLLAR_3758657990369867-1 DOLLAR.mp4")
    # doshit("./TestingVids/dataset_testing_248vids/1DOLLAR_3758657990369867-1 DOLLAR.mp4")
    exit()

    # dataset_folder = ".\\smaller_dataset"
    # np_output_folder = ".\\smaller_dataset_landmarks"
    # process_all(dataset_folder,np_output_folder,num_workers=10)

    # exit()

    # blahblah()


# total_num_frames2,landmark_for_all_frames2 = doshit(showvid=True)


# print(landmark_for_all_frames == landmark_for_all_frames2)


# linear_indices = np.linspace(0,total_num_frames-1,NORMALISED_FRAMES,dtype=int)

# print(linear_indices)


# normalised_landmark_for_all_frames = landmark_for_all_frames[linear_indices.tolist()]

# print(normalised_landmark_for_all_frames)
# print(len(normalised_landmark_for_all_frames))
