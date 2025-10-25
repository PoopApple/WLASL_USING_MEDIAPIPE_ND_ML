import cv2
import mediapipe as mp
import numpy as np
import os

mp_holistic = mp.solutions.holistic

# Select only relevant landmark indices
POSE_LANDMARKS = [
    11, 12,   # shoulders
    13, 14,   # elbows
    15, 16,   # wrists
    23, 24    # upper torso (optional)
]

FACE_LANDMARKS = [1, 2, 4, 5, 6, 9, 10, 13]  # rough face center (nose, eyes, mouth corners)
HAND_LANDMARKS = list(range(21))  # all 21 hand landmarks

def extract_relevant_landmarks(results):
    frame_landmarks = []

    # Pose (arms + shoulders + torso)
    if results.pose_landmarks:
        for idx in POSE_LANDMARKS:
            lm = results.pose_landmarks.landmark[idx]
            frame_landmarks.extend([lm.x, lm.y, lm.z])
    else:
        frame_landmarks.extend([0]*len(POSE_LANDMARKS)*3)

    # Face (key points only)
    if results.face_landmarks:
        for idx in FACE_LANDMARKS:
            lm = results.face_landmarks.landmark[idx]
            frame_landmarks.extend([lm.x, lm.y, lm.z])
    else:
        frame_landmarks.extend([0]*len(FACE_LANDMARKS)*3)

    # Left hand
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            frame_landmarks.extend([lm.x, lm.y, lm.z])
    else:
        frame_landmarks.extend([0]*len(HAND_LANDMARKS)*3)

    # Right hand
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            frame_landmarks.extend([lm.x, lm.y, lm.z])
    else:
        frame_landmarks.extend([0]*len(HAND_LANDMARKS)*3)

    return frame_landmarks


def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    all_frames = []

    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1, enable_segmentation=False) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            landmarks = extract_relevant_landmarks(results)
            all_frames.append(landmarks)

            # Draw only selected parts for visualization
            annotated = frame.copy()
            mp.solutions.drawing_utils.draw_landmarks(
                annotated, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(
                annotated, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(
                annotated, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            cv2.imshow('Video', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return np.array(all_frames, dtype=np.float32)


def process_dataset(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(input_folder):
        if file.endswith(".mp4"):
            path = os.path.join(input_folder, file)
            print(f"🎥 Processing: {path}")
            arr = extract_landmarks_from_video(path)
            np.save(os.path.join(output_folder, file.replace(".mp4", ".npy")), arr)
            print(f"✅ Saved {file.replace('.mp4', '.npy')} shape: {arr.shape}")


if __name__ == "__main__":
    folder = "smaller_dataset/a"
    out = "landmarks_out"
    process_dataset(folder, out)
