import os
import time

os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*=false"
import cv2
import numpy as np
from extract_all_landmarks import doshit
import matplotlib.pyplot as plt

from normalise_data import normalise_lm_arr_spatially, normalise_lm_arr_temporally


from visualise_lms_in_3d import (
    infinite_plot_landmarks_compare2,
    infinite_plot_landmarks,
    infinite_plot_landmarks_compare2_sidebyside,
)

SIGN_WAVE_ne = r"""
  /$$$$$$  /$$                           /$$      /$$                              
 /$$__  $$|__/                          | $$  /$ | $$                              
| $$  \__/ /$$  /$$$$$$  /$$$$$$$       | $$ /$$$| $$  /$$$$$$  /$$    /$$ /$$$$$$ 
|  $$$$$$ | $$ /$$__  $$| $$__  $$      | $$/$$ $$ $$ |____  $$|  $$  /$$//$$__  $$
 \____  $$| $$| $$  \ $$| $$  \ $$      | $$$$_  $$$$  /$$$$$$$ \  $$/$$/| $$$$$$$$
 /$$  \ $$| $$| $$  | $$| $$  | $$      | $$$/ \  $$$ /$$__  $$  \  $$$/ | $$_____/
|  $$$$$$/| $$|  $$$$$$$| $$  | $$      | $$/   \  $$|  $$$$$$$   \  $/  |  $$$$$$$
 \______/ |__/ \____  $$|__/  |__/      |__/     \__/ \_______/    \_/    \_______/
               /$$  \ $$                                                           
              |  $$$$$$/                                                           
               \______/                                                            
"""


SIGN_WAVE_sw = r"""
  ______   __                            __       __                               
 /      \ /  |                          /  |  _  /  |                              
/$$$$$$  |$$/   ______   _______        $$ | / \ $$ |  ______   __     __  ______  
$$ \__$$/ /  | /      \ /       \       $$ |/$  \$$ | /      \ /  \   /  |/      \ 
$$      \ $$ |/$$$$$$  |$$$$$$$  |      $$ /$$$  $$ | $$$$$$  |$$  \ /$$//$$$$$$  |
 $$$$$$  |$$ |$$ |  $$ |$$ |  $$ |      $$ $$/$$ $$ | /    $$ | $$  /$$/ $$    $$ |
/  \__$$ |$$ |$$ \__$$ |$$ |  $$ |      $$$$/  $$$$ |/$$$$$$$ |  $$ $$/  $$$$$$$$/ 
$$    $$/ $$ |$$    $$ |$$ |  $$ |      $$$/    $$$ |$$    $$ |   $$$/   $$       |
 $$$$$$/  $$/  $$$$$$$ |$$/   $$/       $$/      $$/  $$$$$$$/     $/     $$$$$$$/ 
              /  \__$$ |                                                           
              $$    $$/                                                            
               $$$$$$/                                                             
"""

SIGN_WAVE_doh = r"""
                                                                                                                                                                       
                                                                                                                                                                       
   SSSSSSSSSSSSSSS   iiii                                             WWWWWWWW                           WWWWWWWW                                                      
 SS:::::::::::::::S i::::i                                            W::::::W                           W::::::W                                                      
S:::::SSSSSS::::::S  iiii                                             W::::::W                           W::::::W                                                      
S:::::S     SSSSSSS                                                   W::::::W                           W::::::W                                                      
S:::::S            iiiiiii    ggggggggg   gggggnnnn  nnnnnnnn          W:::::W           WWWWW           W:::::Waaaaaaaaaaaaavvvvvvv           vvvvvvv eeeeeeeeeeee    
S:::::S            i:::::i   g:::::::::ggg::::gn:::nn::::::::nn         W:::::W         W:::::W         W:::::W a::::::::::::av:::::v         v:::::vee::::::::::::ee  
 S::::SSSS          i::::i  g:::::::::::::::::gn::::::::::::::nn         W:::::W       W:::::::W       W:::::W  aaaaaaaaa:::::av:::::v       v:::::ve::::::eeeee:::::ee
  SS::::::SSSSS     i::::i g::::::ggggg::::::ggnn:::::::::::::::n         W:::::W     W:::::::::W     W:::::W            a::::a v:::::v     v:::::ve::::::e     e:::::e
    SSS::::::::SS   i::::i g:::::g     g:::::g   n:::::nnnn:::::n          W:::::W   W:::::W:::::W   W:::::W      aaaaaaa:::::a  v:::::v   v:::::v e:::::::eeeee::::::e
       SSSSSS::::S  i::::i g:::::g     g:::::g   n::::n    n::::n           W:::::W W:::::W W:::::W W:::::W     aa::::::::::::a   v:::::v v:::::v  e:::::::::::::::::e 
            S:::::S i::::i g:::::g     g:::::g   n::::n    n::::n            W:::::W:::::W   W:::::W:::::W     a::::aaaa::::::a    v:::::v:::::v   e::::::eeeeeeeeeee  
            S:::::S i::::i g::::::g    g:::::g   n::::n    n::::n             W:::::::::W     W:::::::::W     a::::a    a:::::a     v:::::::::v    e:::::::e           
SSSSSSS     S:::::Si::::::ig:::::::ggggg:::::g   n::::n    n::::n              W:::::::W       W:::::::W      a::::a    a:::::a      v:::::::v     e::::::::e          
S::::::SSSSSS:::::Si::::::i g::::::::::::::::g   n::::n    n::::n               W:::::W         W:::::W       a:::::aaaa::::::a       v:::::v       e::::::::eeeeeeee  
S:::::::::::::::SS i::::::i  gg::::::::::::::g   n::::n    n::::n                W:::W           W:::W         a::::::::::aa:::a       v:::v         ee:::::::::::::e  
 SSSSSSSSSSSSSSS   iiiiiiii    gggggggg::::::g   nnnnnn    nnnnnn                 WWW             WWW           aaaaaaaaaa  aaaa        vvv            eeeeeeeeeeeeee  
                                       g:::::g                                                                                                                         
                           gggggg      g:::::g                                                                                                                         
                           g:::::gg   gg:::::g                                                                                                                         
                            g::::::ggg:::::::g                                                                                                                         
                             gg:::::::::::::g                                                                                                                          
                               ggg::::::ggg                                                                                                                            
                                  gggggg
"""


proj_made_by = r"""
  _____           _           _     __  __           _        ____            
 |  __ \         (_)         | |   |  \/  |         | |      |  _ \         _ 
 | |__) | __ ___  _  ___  ___| |_  | \  / | __ _  __| | ___  | |_) |_   _  (_)
 |  ___/ '__/ _ \| |/ _ \/ __| __| | |\/| |/ _` |/ _` |/ _ \ |  _ <| | | |    
 | |   | | | (_) | |  __/ (__| |_  | |  | | (_| | (_| |  __/ | |_) | |_| |  _ 
 |_|   |_|  \___/| |\___|\___|\__| |_|  |_|\__,_|\__,_|\___| |____/ \__, | (_)
                _/ |                                                 __/ |    
               |__/                                                 |___/
"""


actor_text = r"""
      __  ___  __   __  
 /\  /  `  |  /  \ |__) 
/~~\ \__,  |  \__/ |  \ 
"""
mp_text = r"""
       ___  __          __     __   ___                    __              __        __  
 |\/| |__  |  \ |  /\  |__) | |__) |__     |     /\  |\ | |  \  |\/|  /\  |__) |__/ /__` 
 |  | |___ |__/ | /~~\ |    | |    |___    |___ /~~\ | \| |__/  |  | /~~\ |  \ |  \ .__/ 
"""

lm_heatmap_text = r"""
                __              __        __           ___      ___             __  
|     /\  |\ | |  \  |\/|  /\  |__) |__/ /__`    |__| |__   /\   |   |\/|  /\  |__) 
|___ /~~\ | \| |__/  |  | /~~\ |  \ |  \ .__/    |  | |___ /~~\  |   |  | /~~\ |    
"""


letsnormalise = r"""
      ___ ___  __           __   __                     __   ___ 
|    |__   |  /__`    |\ | /  \ |__)  |\/|  /\  |    | /__` |__  
|___ |___  |  .__/    | \| \__/ |  \  |  | /~~\ |___ | .__/ |___ 
"""

skelview = r"""
 __        ___       ___ ___                      ___      
/__` |__/ |__  |    |__   |   /\  |       \  / | |__  |  | 
.__/ |  \ |___ |___ |___  |  /~~\ |___     \/  | |___ |/\| 
"""


movingonto_text = r"""
                                  ##                                                                          
 ###  ###                         ##                                                                          
 ###  ###                         ##                                                         ##               
 ###::###                                                                                    ##               
 ###  ###   .####.   ##:  :##   ####     ##.####    :###:##             .####.   ##.####   #######    .####.  
 ## ## ##  .######.   ##  ##    ####     #######   .#######            .######.  #######   #######   .######. 
 ##:##:##  ###  ###  :##  ##:     ##     ###  :##  ###  ###            ###  ###  ###  :##    ##      ###  ### 
 ##.##.##  ##.  .##   ##..##      ##     ##    ##  ##.  .##            ##.  .##  ##    ##    ##      ##.  .## 
 ## ## ##  ##    ##   ##::##      ##     ##    ##  ##    ##            ##    ##  ##    ##    ##      ##    ## 
 ##    ##  ##.  .##   :####:      ##     ##    ##  ##.  .##            ##.  .##  ##    ##    ##      ##.  .## 
 ##    ##  ###  ###    ####       ##     ##    ##  ###  ###            ###  ###  ##    ##    ##.     ###  ### 
 ##    ##  .######.    ####    ########  ##    ##  .#######            .######.  ##    ##    #####   .######. 
 ##    ##   .####.     :##:    ########  ##    ##   :###:##             .####.   ##    ##    .####    .####.  
                                                    #.  :##                                                   
                                                    ######                                                    
                                                    :####:                                                    
"""

deeplearning_text = r"""
                                                                                                        ##                        
 #####:                                            ##                                                   ##                        
 #######                                           ##                                                   ##                        
 ##  :##:                                          ##                                                                             
 ##   :##   .####:    .####:   ##.###:             ##         .####:    :####     ##.####  ##.####    ####     ##.####    :###:## 
 ##   .##  .######:  .######:  #######:            ##        .######:   ######    #######  #######    ####     #######   .####### 
 ##    ##  ##:  :##  ##:  :##  ###  ###            ##        ##:  :##   #:  :##   ###.     ###  :##     ##     ###  :##  ###  ### 
 ##    ##  ########  ########  ##.  .##            ##        ########    :#####   ##       ##    ##     ##     ##    ##  ##.  .## 
 ##   .##  ########  ########  ##    ##            ##        ########  .#######   ##       ##    ##     ##     ##    ##  ##    ## 
 ##   :##  ##        ##        ##.  .##            ##        ##        ## .  ##   ##       ##    ##     ##     ##    ##  ##.  .## 
 ##  :##:  ###.  :#  ###.  :#  ###  ###            ##        ###.  :#  ##:  ###   ##       ##    ##     ##     ##    ##  ###  ### 
 #######   .#######  .#######  #######:            ########  .#######  ########   ##       ##    ##  ########  ##    ##  .####### 
 #####:     .#####:   .#####:  ##.###:             ########   .#####:    ###.##   ##       ##    ##  ########  ##    ##   :###:## 
                               ##                                                                                         #.  :## 
                               ##                                                                                         ######  
                               ##                                                                                         :####:  
"""


SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080


def printascii(lines):
    for line in lines.splitlines():
        print(line.center(terminal_width))


def show_vid(vid_path):
    playing = True
    while playing:
        cap = cv2.VideoCapture(vid_path)

        if not cap.isOpened():
            print("Cannot open video.")
            exit()

        while True:
            success, frame = cap.read()

            # If we run out of frames, stop the loop
            if not success:
                break

            # Show the raw frame
            cv2.imshow("Actor", frame)

            # Wait 25ms to simulate normal playback speed
            if cv2.waitKey(25) & 0xFF == ord("q"):
                playing = False
                break

        cap.release()
    cv2.destroyAllWindows()


def visualise_lm_output(data):
    landmark_idx = 16

    # Slice out the X, Y, and Z coordinates across ALL frames for this one landmark
    x_coords = data[:, landmark_idx, 0]
    y_coords = data[:, landmark_idx, 1]
    z_coords = data[:, landmark_idx, 2]

    plt.figure(figsize=(10, 4))
    plt.plot(x_coords, label="X (Left/Right)", color="blue", linewidth=2)
    plt.plot(y_coords, label="Y (Up/Down)", color="red", linewidth=2)
    plt.plot(z_coords, label="Z (Depth)", color="green", linewidth=2)

    plt.title(f"Trajectory of Landmark {landmark_idx} over Time")
    plt.xlabel("Frame Number")
    plt.ylabel("Coordinate Value (Normalized 0.0 to 1.0)")
    plt.legend()
    plt.grid(True)
    plt.show()


def heatmaps(data):
    x_data = data[1:, :, 0]
    y_data = data[1:, :, 1]
    z_data = data[1:, :, 2]

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    im_x = axes[0].imshow(x_data.T, aspect="auto", cmap="viridis")

    def on_press(event):
        if event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_press)

    axes[0].set_title("X Coordinates (Left/Right Screen Position)")
    axes[0].set_ylabel("75 Landmarks")
    fig.colorbar(im_x, ax=axes[0], label="Value")

    im_y = axes[1].imshow(y_data.T, aspect="auto", cmap="plasma")
    axes[1].set_title("Y Coordinates (Up/Down Screen Position)")
    axes[1].set_ylabel("75 Landmarks")
    fig.colorbar(im_y, ax=axes[1], label="Value")

    im_z = axes[2].imshow(z_data.T, aspect="auto", cmap="coolwarm")
    axes[2].set_title("Z Coordinates (Relative Depth / Distance from Camera)")
    axes[2].set_xlabel("Time (Frames)")
    axes[2].set_ylabel("75 Landmarks")
    fig.colorbar(im_z, ax=axes[2], label="Relative Depth")

    plt.tight_layout()
    plt.show()


def heatmaps_comparison(data1, data2, title1="Actor1", title2="Actor2"):
    # Extract data for the first array
    x_data1 = data1[1:, :, 0]
    y_data1 = data1[1:, :, 1]
    z_data1 = data1[1:, :, 2]

    # Extract data for the second array
    x_data2 = data2[1:, :, 0]
    y_data2 = data2[1:, :, 1]
    z_data2 = data2[1:, :, 2]

    fig, axes = plt.subplots(3, 2, figsize=(14, 7), sharex=False, sharey=True)

    def on_press(event):
        if event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_press)

    im_x1 = axes[0, 0].imshow(x_data1.T, aspect="auto", cmap="viridis")
    axes[0, 0].set_title(f"{title1} - X Coordinates")
    axes[0, 0].set_ylabel("75 Landmarks")
    fig.colorbar(im_x1, ax=axes[0, 0], label="Value")

    im_y1 = axes[1, 0].imshow(y_data1.T, aspect="auto", cmap="plasma")
    axes[1, 0].set_title(f"{title1} - Y Coordinates")
    axes[1, 0].set_ylabel("75 Landmarks")
    fig.colorbar(im_y1, ax=axes[1, 0], label="Value")

    im_z1 = axes[2, 0].imshow(z_data1.T, aspect="auto", cmap="coolwarm")
    axes[2, 0].set_title(f"{title1} - Z Coordinates")
    axes[2, 0].set_xlabel("Time (Frames)")
    axes[2, 0].set_ylabel("75 Landmarks")
    fig.colorbar(im_z1, ax=axes[2, 0], label="Relative Depth")

    im_x2 = axes[0, 1].imshow(x_data2.T, aspect="auto", cmap="viridis")
    axes[0, 1].set_title(f"{title2} - X Coordinates")
    fig.colorbar(im_x2, ax=axes[0, 1], label="Value")

    im_y2 = axes[1, 1].imshow(y_data2.T, aspect="auto", cmap="plasma")
    axes[1, 1].set_title(f"{title2} - Y Coordinates")
    fig.colorbar(im_y2, ax=axes[1, 1], label="Value")

    im_z2 = axes[2, 1].imshow(z_data2.T, aspect="auto", cmap="coolwarm")
    axes[2, 1].set_title(f"{title2} - Z Coordinates")
    axes[2, 1].set_xlabel("Time (Frames)")
    fig.colorbar(im_z2, ax=axes[2, 1], label="Relative Depth")

    plt.tight_layout()
    plt.show()


# Example of how to call it:
# heatmaps_comparison(array_one, array_two, "Raw Data", "Processed Data")
if __name__ == "__main__":
    terminal_width = os.get_terminal_size().columns
    # print(terminal_width)

    # print(SIGN_WAVE_ne)
    # print(f"{SIGN_WAVE_ne:^236}")
    print("=" * terminal_width)

    printascii(SIGN_WAVE_ne)
    print()

    text_info = [
        "\n",
        "Project Made By:\n",
        # proj_made_by,
        # "Name\t\t\tEnrollment Number\t\tBatch",
        # "=" * 70,
        "Aryan Sethi"  # \t\t9923103030\t\t\tF2",
        # "Pratham Arora\t\t9923103044\t\t\tF2",
        # "Suvraaj Nandwani\t9923103052\t\t\tF2",
        "\n",
        "=" * terminal_width,
    ]

    print("=" * terminal_width)
    for _ in text_info:
        time.sleep(0.02)
        print(_)

    for _ in range(2):
        time.sleep(0.02)
        print("\n")

    input("continue?")

    sample_vid_text = "Playing Sample Video for the word:"

    print(sample_vid_text.center(terminal_width))

    printascii(actor_text)

    sample_vid_text = "press q to continue"
    print(sample_vid_text.center(terminal_width))

    for _ in range(25):
        time.sleep(0.02)
        print("\n")

    vid_path = "/media/aryan/PoopiDrive/projects_linux/microsoft asl citizen/ASL_Citizen/videos/ACTOR_5867233759174431-ACTOR.mp4"

    vid_path = "/media/aryan/PoopiDrive/projects_linux/microsoft asl citizen/ASL_Citizen/videos/ACTOR_2517131108269346-ACTOR.mp4"
    # doshit(vid_path=vid_path, showvid=True)
    show_vid(vid_path)

    sample_vid_text = "Running Google Mediapipe on the video to extract landmarks:"

    print(sample_vid_text.center(terminal_width))

    printascii(mp_text)
    print()
    # sample_vid_text = "press q to continue"
    # print(sample_vid_text.center(terminal_width))

    for _ in range(20):
        time.sleep(0.02)
        print("\n")

    print("These are harmless warnings from mediapipe\n")
    total_num_frames, vid_path, NP_NORMALISED_LANDMARK_ALL_FRAMES = doshit(
        vid_path=vid_path, showvid=True
    )
    printascii(lm_heatmap_text)
    print()

    sample_vid_text = "press q to continue"
    print(sample_vid_text.center(terminal_width))

    for _ in range(26):
        time.sleep(0.02)
        print("\n")

    heatmaps(NP_NORMALISED_LANDMARK_ALL_FRAMES)

    sample_vid_text = "Running Google Mediapipe on our own video to extract landmarks:"

    print(sample_vid_text.center(terminal_width))

    printascii(mp_text)
    print()
    # sample_vid_text = "press q to continue"
    # print(sample_vid_text.center(terminal_width))

    for _ in range(20):
        time.sleep(0.02)
        print("\n")

    print("These are harmless warnings from mediapipe\n")
    vid_path = "./test_actor.webm"
    total_num_frames, vid_path, NP_NORMALISED_LANDMARK_ALL_FRAMES2 = doshit(
        vid_path=vid_path, showvid=True
    )

    print("\n\n")
    printascii(lm_heatmap_text)

    print()
    sample_vid_text = "press q to continue"
    print(sample_vid_text.center(terminal_width))

    for _ in range(26):
        time.sleep(0.02)
        print("\n")

    heatmaps(NP_NORMALISED_LANDMARK_ALL_FRAMES2)

    heatmaps_comparison(
        NP_NORMALISED_LANDMARK_ALL_FRAMES, NP_NORMALISED_LANDMARK_ALL_FRAMES2
    )

    printascii(letsnormalise)
    print()

    sample_vid_text = "press q to continue"
    print(sample_vid_text.center(terminal_width))

    for _ in range(25):
        time.sleep(0.02)
        print("\n")

    spatially_normalised_arr1 = normalise_lm_arr_spatially(
        NP_NORMALISED_LANDMARK_ALL_FRAMES
    )

    spatially_normalised_arr2 = normalise_lm_arr_spatially(
        NP_NORMALISED_LANDMARK_ALL_FRAMES2
    )

    temporally_spatially_normalised_arr, mask = normalise_lm_arr_temporally(
        spatially_normalised_arr1
    )
    temporally_spatially_normalised_arr2, mask = normalise_lm_arr_temporally(
        spatially_normalised_arr2
    )

    heatmaps_comparison(
        temporally_spatially_normalised_arr, temporally_spatially_normalised_arr2
    )

    printascii(skelview)
    print()

    sample_vid_text = "press q to continue"
    print(sample_vid_text.center(terminal_width))

    for _ in range(26):
        time.sleep(0.02)
        print("\n")

    infinite_plot_landmarks_compare2_sidebyside(
        temporally_spatially_normalised_arr, temporally_spatially_normalised_arr2
    )

    printascii(movingonto_text)
    print()
    printascii(deeplearning_text)

    for _ in range(5):
        time.sleep(0.02)
        print("\n")
