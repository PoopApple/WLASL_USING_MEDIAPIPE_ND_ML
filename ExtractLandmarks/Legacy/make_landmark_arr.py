import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from PIL import Image
import os



"""
NOTE
https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/index

nose tip  = 4
left_eye 473
left ear 454
right_eye 468
right ear 234
upper lip 0

"""
Face_Landmark_FACE_POINTS = [4,473,454,468,234,0]
"^^^^^^^^^NOT NEEDED XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"


"""
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

"""
POSE_POINTS_NEEDED = [24,23,12,11,13,15,14,16,8,10,0,7,8,5,2]

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)




"""
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
"""
# all needed
HAND_POINTS_NEEDED = range(21)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

input_folder = "smaller_dataset"

def process_video(input_path, output_folder):
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"not open")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    
    base_name = os.path.basename(input_path)
    # video_out_path = os.path.join(output_folder, f"processed_{base_name}")
    npy_out_path = os.path.join(output_folder, f"{os.path.splitext(base_name)[0]}.npy")

    # out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))


    all_frames = []
    num_of_frames = 0
    while cap.isOpened():
        num_of_frames+=1
        # print(num_of_frames)
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR → RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(image_rgb)
        hand_results = hands.process(image_rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw pose landmarks
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Draw minimal face points
            # h, w, _ = image_bgr.shape
            # for idx in FACE_POINTS.values():
            #     lm = pose_results.pose_landmarks.landmark[idx]
            #     x, y = int(lm.x * w), int(lm.y * h)
            #     cv2.circle(image_bgr, (x, y), 5, (0, 255, 255), -1)

        # Draw hands
        # if hand_results.multi_hand_landmarks:
        #     for hand_landmarks in hand_results.multi_hand_landmarks:
        #         mp_drawing.draw_landmarks(
        #             image_bgr,
        #             hand_landmarks,
        #             mp_hands.HAND_CONNECTIONS,
        #             mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        #             mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
        #         )

        # ----- Extract landmarks for ML -----
        frame_landmarks = []

        # Pose landmarks (33)
        if pose_results.pose_landmarks:
            print(f"num of pose landmarks: \t{len(pose_results.pose_landmarks.landmark)}")
            for lm in pose_results.pose_landmarks.landmark:
                frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            frame_landmarks.extend([0.0]*(33*4))

        # Minimal face points (5)
        if pose_results.pose_landmarks:
            
            for idx in FACE_POINTS:
                lm = pose_results.pose_landmarks.landmark[idx]
                frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            frame_landmarks.extend([0.0]*(5*4))

        # Hands (2 hands x 21 landmarks)
        if hand_results.multi_hand_landmarks:
            print(f"num of pose landmarks: \t{len(hand_results.multi_hand_landmarks)}")
            for hand_landmarks in hand_results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z, 1.0])
        # pad if no hands detected
        if not hand_results.multi_hand_landmarks:
            frame_landmarks.extend([0.0]*(21*2*4))

        all_frames.append(frame_landmarks)
        print(len(frame_landmarks))

        # Write frame to output video
        # out.write(image_bgr)
    print(num_of_frames)
    print(len(frame_landmarks))
    # Cleanup
    pose.close()
    hands.close()
    cap.release()
    # out.release()

    # Save landmarks as numpy file
    all_frames_np = np.array(all_frames, dtype=np.float32)
    np.save(npy_out_path, all_frames_np)

    # print(f"✅ Processed: {video_out_path}")
    print(f"✅ Landmarks saved: {npy_out_path} (shape={all_frames_np.shape})")


def process_multiple_videos(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            input_path = os.path.join(input_folder, filename)
            process_video(input_path, output_folder)


# Example usage
if __name__ == "__main__":
    input_folder = "vids"           # folder with raw videos
    output_folder = "landmarked"    # folder for videos + landmarks
    process_video("smaller_dataset/abdomen/00335.mp4" , "landmarked/abdomen/")
    # process_multiple_videos(input_folder, output_folder)


















def play_videos_loop(input_folder):
   

    video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]
    if not video_files:
        # print(" No video files found")
        return

    # print("Playing videos. Press Ctrl+C in terminal to stop.")

    try:
        while True:  # infinite loop
            for filename in video_files:
                cap = cv2.VideoCapture(os.path.join(input_folder, filename))
                if not cap.isOpened():
                    print(f"❌ Could not open {filename}")
                    continue

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(image_rgb)
                    hand_results = hands.process(image_rgb)
                    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                    # Pose landmarks
                    if pose_results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            image_bgr,
                            pose_results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                        )
                        # minimal face points
                        h, w, _ = image_bgr.shape
                        for idx in FACE_POINTS:
                            lm = pose_results.pose_landmarks.landmark[idx]
                            x, y = int(lm.x * w), int(lm.y * h)
                            cv2.circle(image_bgr, (x, y), 5, (0, 255, 255), -1)

                    # Hands
                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                image_bgr,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                            )

                    cv2.imshow("Landmarks Viewer", image_bgr)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to skip video
                        break

                cap.release()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")

    pose.close()
    hands.close()
    cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    play_videos_loop("vids")
