import os
import cv2

path = "./all_videos"


minfps = 1e9
maxfps = -1e9
totalfps = 0

minframes = 1e9
maxframes = -1e9
totalframes = 0



vids_done = 0


for filename in os.listdir(path):
        if filename.lower().endswith('.mp4'):
            
            vid_path = os.path.join(path, filename)
            

            cap = cv2.VideoCapture(vid_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            minfps = min(minfps, fps)
            maxfps = max(maxfps,fps)
            totalfps += fps
            
            
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            maxframes = max(maxframes,frames)
            minframes = min(maxframes,frames)
            totalframes += frames
            if frames > 80:
                print(vid_path)
            
            
            cap.release()
            
            vids_done +=1
            print(vids_done)
            
            
print(minfps)
print(maxfps)
print(totalfps/vids_done)
print(minframes)
print(maxframes)
print(totalframes/vids_done)


"""
12.0
59.94
28.530534785231108
57.0
233.0
69.12295492487479
"""