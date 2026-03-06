import os
import cv2
import statistics


path = "./all_videos"


def main():
    
    fps_list = []
    frames_list = []
    
    vids_done = 0


    for filename in os.listdir(path):
            if filename.lower().endswith('.mp4'):
                
                vid_path = os.path.join(path, filename)
                

                cap = cv2.VideoCapture(vid_path)
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                fps_list.append(fps)
                
                
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                frames_list.append(frames)
                
                
                
                cap.release()
                
                vids_done +=1
                print(vids_done)
                
                
    print(len(fps_list))    
    print(len(frames_list))
    print("FPS")
    getallstats(fps_list)
    print("\n\n")
    
    print("FRAMES")
    getallstats(frames_list)
    
main()
"""
11980
11980
FPS
Mean: 28.530534785232394
Median: 29.97
Mode: 29.97
Standard Deviation (sample): 3.101906651411547
Standard Deviation (population): 3.1017771868295334
Variance (sample): 9.621824874071198
Variance (population): 9.621021716736132
Harmonic Mean: 28.20762657332378



FRAMES
Mean: 69.12295492487479
Median: 70.0
Mode: 85.0
Standard Deviation (sample): 26.64041389834537
Standard Deviation (population): 26.639302004778727
Variance (sample): 709.7116526751531
Variance (population): 709.652411301808
Harmonic Mean: 58.530410392450605

"""

def main1():
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




def getallstats(data):
    # Calculate the mean (average)
    mean_value = statistics.mean(data)
    print(f"Mean: {mean_value}")

    # Calculate the median (middle value)
    median_value = statistics.median(data)
    print(f"Median: {median_value}")

    # Calculate the mode (most common value)
    # Note: statistics.mode() raises a StatisticsError if there is no unique mode.
    try:
        mode_value = statistics.mode(data)
        print(f"Mode: {mode_value}")
    except statistics.StatisticsError as e:
        print(f"Mode error: {e}")

    # Calculate the standard deviation (sample)
    stdev_value = statistics.stdev(data)
    print(f"Standard Deviation (sample): {stdev_value}")

    # Calculate the population standard deviation
    pstdev_value = statistics.pstdev(data)
    print(f"Standard Deviation (population): {pstdev_value}")

    # Calculate the variance (sample)
    variance_value = statistics.variance(data)
    print(f"Variance (sample): {variance_value}")

    # Calculate the population variance
    pvariance_value = statistics.pvariance(data)
    print(f"Variance (population): {pvariance_value}")

    # Calculate the harmonic mean
    harmonic_mean_value = statistics.harmonic_mean(data)
    print(f"Harmonic Mean: {harmonic_mean_value}")