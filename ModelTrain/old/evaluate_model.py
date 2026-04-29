import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def eval_model(dataset_path, map_path, model_path):

    dataset = np.load(dataset_path)
    X = dataset["features"]
    masks = dataset["masks"].copy()
    y = dataset["labels"]

    random_num = 1234
    _, X_val, _, mask_val, _, y_val = train_test_split(
        X, masks, y, test_size=0.2, random_state=random_num, stratify=y
    )

    model = tf.keras.models.load_model(model_path)

    with open(map_path, "r") as f:
        label_map = json.load(f)

    index_to_word = {v: k for k, v in label_map.items()}
    target_names = [index_to_word[i] for i in range(len(index_to_word))]

    y_pred_probs = model.predict([X_val, mask_val])
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT (Precision, Recall, F1-Score)")
    print("=" * 50)
    # This prints the massive text table to your terminal
    report = classification_report(y_val, y_pred, target_names=target_names)
    print(report[-5:])

    report_dict = classification_report(
        y_val, y_pred, target_names=target_names, output_dict=True
    )

    plt.figure(figsize=(20, 8))
    f1_scores = [report_dict[word]["f1-score"] for word in target_names]
    plt.bar(target_names, f1_scores, color="cornflowerblue")
    plt.title("Model Accuracy (F1-Score) per ASL Word", fontsize=16)
    plt.xlabel("ASL Words", fontsize=12)
    plt.ylabel("F1-Score (0.0 to 1.0)", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)  # Rotate words 90 degrees so they fit
    plt.tight_layout()
    # plt.savefig("./dataset2.0/f1_scores_per_word_bigru.png", dpi=300)

    plt.show()

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(18, 15))  # Massive canvas for 106 words
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title("Confusion Matrix: True Words vs Predicted Words", fontsize=20)
    plt.ylabel("What the word ACTUALLY was", fontsize=14)
    plt.xlabel("What the model PREDICTED", fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    # plt.savefig("./dataset2.0/confusion_matrix_bigru.png", dpi=300)

    plt.show()


def compare_models(dataset_path, map_path, model_path1, model_path2, name1, name2):
    dataset = np.load(dataset_path)
    X, masks, y = dataset["features"], dataset["masks"], dataset["labels"]

    _, X_test, _, m_test, _, y_test = train_test_split(
        X, masks, y, test_size=0.2, random_state=1234, stratify=y
    )

    with open(map_path, "r") as f:
        label_map = json.load(f)

    target_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]

    all_results = []
    for path in [model_path1, model_path2]:
        m = tf.keras.models.load_model(path)
        probs = m.predict([X_test, m_test])
        preds = np.argmax(probs, axis=1)

        report = classification_report(
            y_test, preds, target_names=target_names, output_dict=True
        )
        cm = confusion_matrix(y_test, preds)
        all_results.append({"report": report, "cm": cm, "acc": report["accuracy"]})

    plt.figure(figsize=(22, 8))
    x = np.arange(len(target_names))
    width = 0.35

    f1_m1 = [all_results[0]["report"][word]["f1-score"] for word in target_names]
    f1_m2 = [all_results[1]["report"][word]["f1-score"] for word in target_names]

    plt.bar(
        x - width / 2,
        f1_m1,
        width,
        label=f"{name1} (Overall: {all_results[0]['acc']:.2f})",
        color="dodgerblue",
    )
    plt.bar(
        x + width / 2,
        f1_m2,
        width,
        label=f"{name2} (Overall: {all_results[1]['acc']:.2f})",
        color="orange",
    )

    plt.title(f"F1-Score Comparison: {name1} vs {name2}", fontsize=18)
    plt.xticks(x, target_names, rotation=90, fontsize=8)
    plt.ylabel("F1-Score")
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    # plt.savefig("f1_comparison.png", dpi=300)

    fig_cm, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), sharey=True)

    sns.heatmap(
        all_results[0]["cm"],
        ax=ax1,
        cmap="Blues",
        cbar=False,
        xticklabels=target_names,
        yticklabels=target_names,
    )
    ax1.set_title(f"{name1} Confusion Matrix", fontsize=16)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    sns.heatmap(
        all_results[1]["cm"],
        ax=ax2,
        cmap="Oranges",
        cbar=False,
        xticklabels=target_names,
        yticklabels=target_names,
    )
    ax2.set_title(f"{name2} Confusion Matrix", fontsize=16)
    ax2.set_xlabel("Predicted")

    plt.tight_layout()

    print("\nCharts Generated. Press 'q' in any window to close all.")

    def on_close(event):
        if event.key in ["q", "escape"]:
            plt.close("all")

    plt.gcf().canvas.mpl_connect("key_press_event", on_close)
    plt.show()


if __name__ == "__main__":
    model_path = "./dataset1.0/asl_bigru_16-03-26__19-50_best.keras"
    map_path = "./dataset1.0/word_to_ind.json"
    dataset_path = "./dataset2.0/dataset2-0.npz"

    eval_model(dataset_path, map_path, model_path)
