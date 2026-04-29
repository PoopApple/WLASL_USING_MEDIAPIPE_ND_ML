import numpy as np
import os
import matplotlib.pyplot as plt
import shutil


def plot_graph_of_freq(frame_freq_arr):
    x = [frame for frame, freq in frame_freq_arr]
    y = [freq for frame, freq in frame_freq_arr]

    plt.plot(x, y)
    plt.show()


def getstatsoldfn():
    dataset_path = "./landmarks/"
    allfps = {}
    allframes = {}

    framesssssss = []
    n = len(os.listdir(dataset_path))

    for word in os.listdir(dataset_path):
        worddir = os.path.join(dataset_path, word)
        for file in os.listdir(worddir):
            filepath = os.path.join(worddir, file)
            arr = np.load(filepath)
            frames = int(arr[0][0][0])
            framesssssss.append(frames)

            # if frames < 40:
            #     with open("lessthan40frames.txt", "a") as f:
            #         f.write("\n")
            #         f.write(filepath)
            #         f.write("\n")
            #
            # print(frames)
            fps = int(arr[0][0][1])
            # print(fps)

            if allfps.get(fps):
                allfps[fps] += 1
            else:
                allfps[fps] = 1

            if allframes.get(frames):
                allframes[frames] += 1
            else:
                allframes[frames] = 1

    print(allframes)

    freq = [(fram, f) for fram, f in allframes.items()]
    freq.sort(reverse=True)

    plot_graph_of_freq(freq)

    for fr, f in freq:
        print(f"Frames={fr} :: Freq={f}")

    print()
    print()
    print()
    print()
    print()
    print()

    freq = [(f, fram) for fram, f in allframes.items()]
    freq.sort(reverse=True)

    for f, fr in freq:
        print(f"Frames={fr} :: Freq={f}")
    percentiles_to_take = [10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 99]

    print(f"Total Frames = {sum(framesssssss)}")
    print(f"Average = {np.average(framesssssss)}")
    print(f"Median = {np.median(framesssssss)}")
    print(f"Min = {np.min(framesssssss)}")
    print(f"Std Dev = {np.std(framesssssss)}")
    print(f"Sample Std Dev = {np.std(framesssssss, ddof=1)}")
    print("Percentiles =>")
    percentiles = np.percentile(framesssssss, percentiles_to_take)
    for ptt, p in zip(percentiles_to_take, percentiles):
        print(f"{ptt}% = {np.round(p, 2)}")
    print(f"Num = {n}")


def statsnew_lt20gte15():
    dataset_path = "./landmarks/"
    removed_dataset_path = "./landmarks_lt20gte15/"  # less than 14
    os.makedirs(removed_dataset_path, exist_ok=True)
    for folder in os.listdir(dataset_path):
        numbs = len(os.listdir(os.path.join(dataset_path, folder)))
        print(f"{folder} => {numbs}")

        if numbs < 20:
            og_path = os.path.join(dataset_path, folder)
            new_path = os.path.join(removed_dataset_path, folder)
            shutil.move(og_path, new_path)
            # os.rename(og_path, new_path)


def statsnew_lte14():
    dataset_path = "./landmarks/"
    removed_dataset_path = "./landmarks_lte14/"  # less than 14
    os.makedirs(removed_dataset_path, exist_ok=True)
    for folder in os.listdir(dataset_path):
        numbs = len(os.listdir(os.path.join(dataset_path, folder)))
        print(f"{folder} => {numbs}")

        if numbs < 15:
            og_path = os.path.join(dataset_path, folder)
            new_path = os.path.join(removed_dataset_path, folder)

            os.rename(og_path, new_path)

def getstatsfn():
    dataset_path = "./landmarks/"
    wordslens = {}
    alllens = []

    n = len(os.listdir(dataset_path))

    for word in os.listdir(dataset_path):
        worddir = os.path.join(dataset_path, word)
        wordsize = len(os.listdir(worddir))
        wordslens[word] = wordsize
        alllens.append(wordsize)
        

            
    print(wordslens)

    freq = [(fram, f) for fram, f in wordslens.items()]
    freq.sort(reverse=True)

    plot_graph_of_freq(freq)

    for fr, f in freq:
        print(f"Frames={fr} :: Freq={f}")

    print()
    print()
    print()
    print()
    print()
    print()

    freq = [(f, fram) for fram, f in wordslens.items()]
    freq.sort(reverse=True)

    for f, fr in freq:
        print(f"Frames={fr} :: Freq={f}")
    percentiles_to_take = [10, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 99]

    print(f"Total Frames = {sum(alllens)}")
    print(f"Average = {np.average(alllens)}")
    print(f"Median = {np.median(alllens)}")
    print(f"Min = {np.min(alllens)}")
    print(f"Std Dev = {np.std(alllens)}")
    print(f"Sample Std Dev = {np.std(alllens, ddof=1)}")
    print("Percentiles =>")
    percentiles = np.percentile(alllens, percentiles_to_take)
    for ptt, p in zip(percentiles_to_take, percentiles):
        print(f"{ptt}% = {np.round(p, 2)}")
    print(f"Num = {n}")


if __name__ == "__main__":
    # statsnew_lte14()
    # statsnew_lt20gte15()
    getstatsfn()
    print("\n\n\n\n\n\n\n")
    getstatsoldfn()
