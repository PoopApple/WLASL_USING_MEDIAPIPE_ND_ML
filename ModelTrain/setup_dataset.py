import numpy as np
import os
import json


def check_mask_validity(dataset_path="./dataset2.0/landmarks_npz/"):
    for word in os.listdir(dataset_path):
        word_path = os.path.join(dataset_path, word)
        for file in os.listdir(word_path):
            if file.endswith(".npz"):
                file_path = os.path.join(word_path, file)
                arr = np.load(file_path)
                maskarr = arr["mask"]

                foundFalse = False
                for b in maskarr:
                    if foundFalse and b:
                        with open("setup_dataset.log", "a") as f:
                            f.write(f"{file_path} :: found true after false\n")
                        break
                    if not b:
                        foundFalse = True


if __name__ == "__main__":
    check_mask_validity()

    dataset_path = "./dataset2.0/landmarks_npz/"
    combined_dataset_path = "./dataset2.0/dataset2-0.npz"

    classes = sorted(
        [
            word
            for word in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, word))
        ]
    )

    word_to_ind = {word: ind for ind, word in enumerate(classes)}
    with open("./dataset2.0/word_to_ind.json", "w") as f:
        json.dump(word_to_ind, f)

    x_data = []
    x_mask = []
    y_labels = []  # this is index

    for word, class_ind in word_to_ind.items():
        word_path = os.path.join(dataset_path, word)
        for file in os.listdir(word_path):
            if file.endswith(".npz"):
                file_path = os.path.join(word_path, file)

                filedata = np.load(file_path)
                # print(filedata.shape)
                # print(filedata)
                # exit()

                data = filedata["data"]
                mask = filedata["mask"]

                x_data.append(data)
                x_mask.append(mask)
                y_labels.append(class_ind)

    X_data = np.array(x_data)
    X_mask = np.array(x_mask)
    Y_labels = np.array(y_labels)

    print(f"Features:\t{X_data.shape}")
    print(f"Masks:\t{X_mask.shape}")
    print(f"Labels:\t{Y_labels.shape}")

    np.savez_compressed(
        combined_dataset_path, features=X_data, masks=X_mask, labels=Y_labels
    )
