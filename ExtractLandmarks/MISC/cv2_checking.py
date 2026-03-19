import cv2

video_path = "/media/aryan/PoopiDrive/projects_linux/microsoft asl citizen/ASL_Citizen/videos/ACTOR_16964925362473782-ACTOR.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video.")
    exit()

print("Playing video... Press 'q' to quit.")

while True:
    success, frame = cap.read()

    # If we run out of frames, stop the loop
    if not success:
        print("Video finished.")
        break

    # Show the raw frame
    cv2.imshow("Barebones Viewer", frame)

    # Wait 25ms to simulate normal playback speed
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
