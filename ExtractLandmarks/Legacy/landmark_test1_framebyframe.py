"""https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md
https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md#model_complexity"""


import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

vid_path = "./smaller_dataset/abdomen/00335.mp4"
vid_path = "./smaller_dataset/abdomen/00336.mp4"
# vid_path = "./smaller_dataset/abdomen/00338.mp4"
vid_path = "./smaller_dataset/above/00430.mp4"

"""https://camo.githubusercontent.com/d3afebfc801ee1a094c28604c7a0eb25f8b9c9925f75b0fff4c8c8b4871c0d28/68747470733a2f2f6d65646961706970652e6465762f696d616765732f6d6f62696c652f706f73655f747261636b696e675f66756c6c5f626f64795f6c616e646d61726b732e706e67"""
"""https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

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

"""
needed_poses = [0,2,5,7,8,9,10,11,12,13,15,14,16,23,24] #15


# For webcam input:
cap = cv2.VideoCapture(vid_path)



fps = cap.get(cv2.CAP_PROP_FPS)
total_num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

print(cap.get(cv2.CAP_PROP_FPS))
print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
hands = mp_hands.Hands(static_image_mode = False,  model_complexity=1, min_detection_confidence=0.4, min_tracking_confidence=0.4, max_num_hands=2)
pose = mp_pose.Pose(static_image_mode = False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

success, image = cap.read()

landmark_for_all_frames = []

while cap.isOpened():
    if cv2.waitKey(5) & 0xFF == 27:
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

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        landmark_for_one_frame = []
        """ format = all 15 poses(face and arms and wrists and torso) then left hand then right hand"""
        if result_pose:
            # print(f"pose landmarks: {len(result_pose.pose_landmarks.landmark)}")

            # print(f"nose: {result_pose.pose_landmarks.landmark[0]}")
            # print(f"le ear: {result_pose.pose_landmarks.landmark[7]}")
            # print(f"ri ear: {result_pose.pose_landmarks.landmark[8]}")

            """
            needed_poses = [0,2,5,7,8,9,10,11,12,13,15,14,16,23,24] #15
            
            array of landmarks need --  15x4 = 15landmarks with 4 dimensions [x y z visibility]
            """

            # pose_
            landmark_for_one_frame.extend(
                [
                    [
                        result_pose.pose_landmarks.landmark[i].x,
                        result_pose.pose_landmarks.landmark[i].y,
                        result_pose.pose_landmarks.landmark[i].z,
                        result_pose.pose_landmarks.landmark[i].visibility,
                    ]
                    for i in needed_poses
                ]
            )
            # np_pose_landmark_for_one_frame = np.array(pose_landmark_for_one_frame,dtype=np.float32)
            # print(np_pose_landmark_for_one_frame.shape) ==== 15x4

            mp_drawing.draw_landmarks(
                image,
                result_pose.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        else:
            landmark_for_one_frame.extend(
                [
                    [
                        0,0,0,0
                    ]
                    for i in needed_poses
                ]
            )

        """https://stackoverflow.com/questions/67455791/mediapipe-python-link-landmark-with-handedness"""

        if results.multi_hand_landmarks:

            hand_landmarks_list = {"Left": [], "Right": []}

            # print(len(results.multi_hand_landmarks))

            for handedness in results.multi_handedness:
                # print(handedness)
                idx = handedness.classification[0].index
                # print(idx)

            # if len(results.multi_hand_landmarks) == 2:
            #         print(results.multi_hand_landmarks)
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                lbl = results.multi_handedness[idx].classification[0].label
                # print()
                # print(lbl)

                # print(hand_landmarks)
                # print(f"hand landmarks: {len(hand_landmarks.landmark)}")
                # for ind in range(21):
                    # print([hand_landmarks.landmark[ind].x , hand_landmarks.landmark[ind].y , hand_landmarks.landmark[ind].z , 1])

                hand_landmarks_list[lbl] = [[hand_landmarks.landmark[ind].x , hand_landmarks.landmark[ind].y , hand_landmarks.landmark[ind].z , 1] for ind in range(21)]
                # print(hand_landmarks_list)

                # print(hand_landmarks.landmark[0])
                """
                NOTE
                NOTE
                NOTE
                NOTE
                
                done --- ADD LEFT HAND LANDMARKS TO LEFTHAND LIST AND RIGHT TO RIGHT HAND USING IF ELSE OR WHATEVER
                
                
                THEN CHECK ALL DIMESNIONS
                total landmarks = 15+21+21 = 57
                each has x,y,z,visibilty 
                
                done --- shape of each frame data = 57x4 
                
                
                THEN NORMALISE DATA TO 70 FRAMES USING LIN SPACE
                
                THEN STORE IN NPY
                
                
                THEN RUN LTSM
                
                THEN TRY OUT 3DCNN
                
                
                
                
                """

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
            
            if hand_landmarks_list["Left"] == []:
                landmark_for_one_frame.extend([[0,0,0,0] for i in range(21)])
                # print([[0,0,0,0] for i in range(21)])
                # print(landmark_for_one_frame)
            else:
                landmark_for_one_frame.extend(hand_landmarks_list["Left"])
                
            if hand_landmarks_list["Right"] == []:
                landmark_for_one_frame.extend([[0,0,0,0] for i in range(21)])
                # print([[0,0,0,0] for i in range(21)])
                # print(landmark_for_one_frame)
            else:
                landmark_for_one_frame.extend(hand_landmarks_list["Right"])
            
            # print([[0,0,0,0] for i in range(21)])
            # print(landmark_for_one_frame)
            # print(len(landmark_for_one_frame))
        
        else:
            landmark_for_one_frame.extend([[0,0,0,0] for i in range(21)])
            landmark_for_one_frame.extend([[0,0,0,0] for i in range(21)])
            
        
        
        
        
        
        landmark_for_all_frames.extend([landmark_for_one_frame])
    print(total_num_frames)
    print(len(landmark_for_all_frames))
            
            
            
    original_height, original_width = image.shape[:2]

    # Define new width while maintaining the aspect ratio
    new_width = 800
    aspect_ratio = new_width / original_width
    new_height = int(original_height * aspect_ratio)  # Compute height based on aspect ratio

    # Resize the image
    resized_image = cv2.resize(cv2.flip(image, 1), (new_width, new_height))

    # Display the resized image
    cv2.imshow("Resized Image", resized_image)

    # cv2.imshow("MediaPipe Hands", cv2.flip(image, 1))

    # if cv2.waitKey(5) & 0xFF == 27:
    #     break
cap.release()
