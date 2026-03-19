import os
import shutil


if __name__ == "__main__":
    dataset_path = "./dataset1.0/landmarks/"
    npz_dataset_path = "./dataset1.0/landmarks_npz/"

    os.makedirs(npz_dataset_path, exist_ok=True)

    for word in os.listdir(dataset_path):
        npz_word_path = os.path.join(npz_dataset_path, word)
        word_path = os.path.join(dataset_path, word)

        os.makedirs(npz_word_path, exist_ok=True)

        for npzfile in os.listdir(word_path):
            if npzfile.endswith(".npz"):
                npz_file_path = os.path.join(npz_word_path, npzfile)
                og_file_path = os.path.join(word_path, npzfile)

                shutil.move(og_file_path, npz_file_path)
