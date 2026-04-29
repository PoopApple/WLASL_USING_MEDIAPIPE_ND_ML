import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def eval_and_compare_models(dataset_path, map_path, model_path_gru, model_path_lstm):
    # 1. Load Data
    dataset = np.load(dataset_path)
    X, masks, y = dataset["features"], dataset["masks"], dataset["labels"]

    # Use a fixed random_state so both models face the EXACT same test
    _, X_val, _, mask_val, _, y_val = train_test_split(
        X, masks, y, test_size=0.2, random_state=1234, stratify=y
    )

    with open(map_path, "r") as f:
        label_map = json.load(f)
    target_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]

    # 2. Evaluate both models
    model_data = []
    for path, name in [(model_path_gru, "BiGRU"), (model_path_lstm, "BiLSTM")]:
        model = tf.keras.models.load_model(path)

        # Capture model stats
        params = model.count_params()

        # Get Predictions
        y_pred_probs = model.predict([X_val, mask_val], verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Generate Reports
        report_dict = classification_report(
            y_val, y_pred, target_names=target_names, output_dict=True
        )
        cm = confusion_matrix(y_val, y_pred)

        model_data.append(
            {
                "name": name,
                "params": params,
                "report": report_dict,
                "cm": cm,
                "accuracy": report_dict["accuracy"],
            }
        )

    # ======================================================
    # ONE-PAGE REVIEWER PRINTOUT
    # ======================================================
    m1, m2 = model_data[0], model_data[1]
    param_saving = ((m2["params"] - m1["params"]) / m2["params"]) * 100

    print("\n" + "█" * 70)
    print(f"{'PROJECT REVIEWER SUMMARY: ASL ARCHITECTURE COMPARISON':^70}")
    print("█" * 70)

    print(f"\n[1] HARDWARE & EFFICIENCY")
    print(f"{'Model Architecture':<25} | {'Parameters':<15} | {'Estimated Size'}")
    print("-" * 70)
    print(f"{m1['name']:<25} | {m1['params']:,<15} | ~5.07 MB")
    print(f"{m2['name']:<25} | {m2['params']:,<15} | ~6.69 MB")
    print(
        f"👉 {m1['name']} is {param_saving:.1f}% more memory-efficient than {m2['name']}."
    )

    print(f"\n[2] PERFORMANCE METRICS")
    print(f"{'Metric':<25} | {m1['name']:<15} | {m2['name']:<15} | Winner")
    print("-" * 70)
    print(
        f"{'Overall Accuracy':<25} | {m1['accuracy']:<15.4f} | {m2['accuracy']:<15.4f} | {m1['name'] if m1['accuracy'] > m2['accuracy'] else m2['name']}"
    )
    print(
        f"{'Macro F1-Score':<25} | {m1['report']['macro avg']['f1-score']:<15.4f} | {m2['report']['macro avg']['f1-score']:<15.4f} | {m1['name'] if m1['report']['macro avg']['f1-score'] > m2['report']['macro avg']['f1-score'] else m2['name']}"
    )

    print(f"\n[3] FINAL VERDICT FOR PRODUCTION")
    if m1["accuracy"] >= (m2["accuracy"] - 0.01):
        print(
            f"Consensus: {m1['name']} is recommended due to its high efficiency and minimal accuracy trade-off."
        )
    else:
        print(
            f"Consensus: {m2['name']} is recommended for maximum recognition accuracy."
        )
    print("█" * 70 + "\n")

    # ======================================================
    # VISUALIZATIONS (F1 Comparison)
    # ======================================================
    plt.figure(figsize=(20, 8))
    x = np.arange(len(target_names))
    width = 0.35

    f1_gru = [m1["report"][w]["f1-score"] for w in target_names]
    f1_lstm = [m2["report"][w]["f1-score"] for w in target_names]

    plt.bar(x - width / 2, f1_gru, width, label=f"{m1['name']}", color="cornflowerblue")
    plt.bar(x + width / 2, f1_lstm, width, label=f"{m2['name']}", color="salmon")

    plt.title("Model Accuracy (F1-Score) Comparison per ASL Word", fontsize=16)
    plt.xticks(x, target_names, rotation=90, fontsize=7)
    plt.ylabel("F1-Score")
    plt.legend()
    plt.tight_layout()

    # ======================================================
    # VISUALIZATIONS (Confusion Matrices)
    # ======================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), sharey=True)
    sns.heatmap(
        m1["cm"],
        ax=ax1,
        cmap="Blues",
        cbar=False,
        xticklabels=target_names,
        yticklabels=target_names,
    )
    ax1.set_title(f"{m1['name']} Confusion Matrix")

    sns.heatmap(
        m2["cm"],
        ax=ax2,
        cmap="Oranges",
        cbar=False,
        xticklabels=target_names,
        yticklabels=target_names,
    )
    ax2.set_title(f"{m2['name']} Confusion Matrix")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    eval_and_compare_models(
        "./dataset1.0/dataset1-0.npz",
        "./dataset1.0/word_to_ind.json",
        "./dataset1.0/asl_bigru_16-03-26__19-50_best.keras",
        "./dataset1.0/BiLSTM/asl_bilstm_16-03-26__22-38_best.keras",
    )
