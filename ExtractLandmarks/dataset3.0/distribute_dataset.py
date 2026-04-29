import os


if __name__ == "__main__":
    dataset_path = "./landmarks/"
    for filename in os.listdir(dataset_path):
        if filename.endswith(".npy"):
            word = filename.split("_")[0]
            word_dir_path = os.path.join(dataset_path, word)
            os.makedirs(word_dir_path, exist_ok=True)
            og_path = os.path.join(dataset_path, filename)
            new_path = os.path.join(dataset_path, word, filename)
            os.rename(og_path, new_path)
