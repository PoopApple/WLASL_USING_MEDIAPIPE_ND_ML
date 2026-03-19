import json
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


MODEL_PATH = "./dataset1.0/asl_bigru_16-03-26__19-50_best.keras"
LABEL_MAP_PATH = "./dataset1.0/word_to_ind.json"
TARGET_FILE = "./test_actor.npz"


model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

index_to_word = {v: k for k, v in label_map.items()}


def test_single_sample(file_path, runs=50):
    try:
        data = np.load(file_path)
        X_sample = data["data"].astype(np.float32)
        mask_sample = data["mask"].astype(bool)
    except Exception as e:
        print(f"Failed to load .npz file: {e}")
        return

    X_batch = np.expand_dims(X_sample, axis=0)
    mask_batch = np.expand_dims(mask_sample, axis=0)

    # 🔥 Warmup
    _ = model.predict([X_batch, mask_batch], verbose=0)

    times = []

    for _ in range(runs):
        start = time.perf_counter()
        preds = model.predict([X_batch, mask_batch], verbose=0)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)

    predictions = preds[0]
    predicted_idx = np.argmax(predictions)
    confidence = predictions[predicted_idx]
    predicted_word = index_to_word[predicted_idx]

    print("\n" + "=" * 50)
    print("SINGLE SAMPLE RESULT")
    print("=" * 50)

    print(f"PREDICTION:  {predicted_word.upper()}")
    print(f"CONFIDENCE:  {confidence * 100:.2f}%")

    print("\nTop 5 Guesses:")
    top_5_indices = np.argsort(predictions)[-5:][::-1]
    for i in top_5_indices:
        print(f"{index_to_word[i]}: {predictions[i] * 100:.2f}%")

    print("\n⏱️ Timing:")
    print(f"Avg time: {avg_time * 1000:.3f} ms")
    print(f"Min time: {min(times) * 1000:.3f} ms")
    print(f"Max time: {max(times) * 1000:.3f} ms")


def eval_model(dataset_path, map_path, model_path):

    dataset = np.load(dataset_path)
    X = dataset["features"]
    masks = dataset["masks"].copy()
    y = dataset["labels"]

    _, X_val, _, mask_val, _, y_val = train_test_split(
        X, masks, y, test_size=0.2, random_state=1234, stratify=y
    )

    model = tf.keras.models.load_model(model_path)

    with open(map_path, "r") as f:
        label_map = json.load(f)

    index_to_word = {v: k for k, v in label_map.items()}
    target_names = [index_to_word[i] for i in range(len(index_to_word))]

    _ = model.predict([X_val[:1], mask_val[:1]], verbose=0)

    start = time.perf_counter()
    y_pred_probs = model.predict([X_val, mask_val], verbose=0)
    end = time.perf_counter()

    total_time = end - start
    per_sample_time = total_time / len(X_val)

    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)

    report = classification_report(y_val, y_pred, target_names=target_names)
    print(report)

    print("\n⏱️ Timing:")
    print(f"Total time: {total_time:.3f} sec")
    print(f"Avg per sample: {per_sample_time * 1000:.3f} ms")

    report_dict = classification_report(
        y_val, y_pred, target_names=target_names, output_dict=True
    )

    plt.figure(figsize=(20, 8))
    f1_scores = [report_dict[word]["f1-score"] for word in target_names]

    plt.bar(target_names, f1_scores)
    plt.title("F1 Score per Word")
    plt.xticks(rotation=90)
    plt.tight_layout()

    cm = confusion_matrix(y_val, y_pred)

    plt.figure(figsize=(18, 15))
    sns.heatmap(
        cm,
        annot=False,
        xticklabels=target_names,
        yticklabels=target_names,
    )

    plt.title("Confusion Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    test_single_sample(TARGET_FILE)

    eval_model(
        dataset_path="./dataset1.0/dataset1-0.npz",
        map_path="./dataset1.0/word_to_ind.json",
        model_path="./dataset1.0/BiLSTM/asl_bilstm_16-03-26__22-38_best.keras",
    )
