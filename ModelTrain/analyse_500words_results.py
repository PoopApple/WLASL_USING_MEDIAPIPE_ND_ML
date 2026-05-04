import csv
import json
import re
from pathlib import Path
from statistics import median, multimode, mean

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data_pipeline import ASLDataPipeline
from model import calc_velocity, build_model


REPORT_PATTERN = re.compile(r"^\s*(\S+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+(\d+)\s*$")
ACCURACY_PATTERN = re.compile(r"^\s*accuracy\s+([0-9]*\.?[0-9]+)\s+(\d+)\s*$")
MACRO_PATTERN = re.compile(r"^\s*macro avg\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+(\d+)\s*$")
WEIGHTED_PATTERN = re.compile(r"^\s*weighted avg\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+(\d+)\s*$")
HEADER_KV_PATTERN = re.compile(r"^\s*([A-Za-z_]+)\s*:\s*([0-9]*\.?[0-9]+)\s*$")


# CONFIG (edit directly here)
ROOT_DIR = "dataset3.0/500words"
OUTPUT_DIR = "../Results/dataset3_500words_analysis"
DATASET_DIR = "../ExtractLandmarks/dataset3.0/landmarks_npz"
JSON_FILE_NAME = "dataset3_500w_model_metrics.json"
NUM_WORDS = 2731
BATCH_SIZE = 32
VAL_SPLIT = 0.20
TEST_SPLIT = 0.00
SEED = 1234
LEARNING_RATE = 1e-4
USE_TFRECORD = True
USE_AUGMENTATION = True


def compute_kinematic_features(x):
    x = x[:, :, :63, :3]

    body = x[:, :, :21, :]
    left = x[:, :, 21:42, :]
    right = x[:, :, 42:63, :]

    def angle(a, b, c):
        ba = a - b
        bc = c - b

        ba = tf.math.l2_normalize(ba, axis=-1)
        bc = tf.math.l2_normalize(bc, axis=-1)

        cos = tf.reduce_sum(ba * bc, axis=-1)
        cos = tf.clip_by_value(cos, -0.99999, 0.99999)

        return tf.acos(cos)

    def bone(p1, p2):
        v = p2 - p1
        return tf.math.l2_normalize(v, axis=-1)

    l_shoulder = 7
    r_shoulder = 8
    l_elbow = 9
    r_elbow = 10
    l_wrist = 11
    r_wrist = 12
    l_hip = 19
    r_hip = 20

    left_elbow = angle(body[:, :, l_shoulder], body[:, :, l_elbow], body[:, :, l_wrist])
    right_elbow = angle(body[:, :, r_shoulder], body[:, :, r_elbow], body[:, :, r_wrist])
    left_shoulder = angle(body[:, :, l_hip], body[:, :, l_shoulder], body[:, :, l_elbow])
    right_shoulder = angle(body[:, :, r_hip], body[:, :, r_shoulder], body[:, :, r_elbow])

    def hand_angles(hand):
        fingers = [
            (1, 2, 3, 4),
            (5, 6, 7, 8),
            (9, 10, 11, 12),
            (13, 14, 15, 16),
            (17, 18, 19, 20),
        ]

        out = []
        wrist = hand[:, :, 0]

        for mcp, pip, dip, tip in fingers:
            out.append(angle(hand[:, :, mcp], hand[:, :, pip], hand[:, :, dip]))
            out.append(angle(hand[:, :, pip], hand[:, :, dip], hand[:, :, tip]))
            out.append(angle(wrist, hand[:, :, mcp], hand[:, :, tip]))

        return tf.stack(out, axis=-1)

    left_hand_angles = hand_angles(left)
    right_hand_angles = hand_angles(right)

    left_upper_arm = bone(body[:, :, l_shoulder], body[:, :, l_elbow])
    left_forearm = bone(body[:, :, l_elbow], body[:, :, l_wrist])
    right_upper_arm = bone(body[:, :, r_shoulder], body[:, :, r_elbow])
    right_forearm = bone(body[:, :, r_elbow], body[:, :, r_wrist])

    features = tf.concat(
        [
            tf.stack([left_elbow, right_elbow, left_shoulder, right_shoulder], axis=-1),
            left_hand_angles,
            right_hand_angles,
            left_upper_arm,
            left_forearm,
            right_upper_arm,
            right_forearm,
        ],
        axis=-1,
    )

    return features


def parse_report(report_path: Path) -> dict:
    lines = report_path.read_text(encoding="utf-8").splitlines()

    model_name = "unknown"
    timestamp = ""
    words = None

    if len(lines) >= 3:
        if lines[0].startswith("Model:"):
            model_name = lines[0].split(":", 1)[1].strip()
        if lines[1].startswith("Words:"):
            words_raw = lines[1].split(":", 1)[1].strip()
            words = int(words_raw) if words_raw.isdigit() else words_raw
        if lines[2].startswith("Timestamp:"):
            timestamp = lines[2].split(":", 1)[1].strip()

    scalar_metrics = {}
    per_word = []
    accuracy = None
    macro = None
    weighted = None

    for line in lines:
        kv_match = HEADER_KV_PATTERN.match(line)
        if kv_match:
            key = kv_match.group(1).strip().lower()
            val = float(kv_match.group(2))
            scalar_metrics[key] = val
            continue

        acc_match = ACCURACY_PATTERN.match(line)
        if acc_match:
            accuracy = {
                "accuracy": float(acc_match.group(1)),
                "support": int(acc_match.group(2)),
            }
            continue

        macro_match = MACRO_PATTERN.match(line)
        if macro_match:
            macro = {
                "precision": float(macro_match.group(1)),
                "recall": float(macro_match.group(2)),
                "f1": float(macro_match.group(3)),
                "support": int(macro_match.group(4)),
            }
            continue

        weighted_match = WEIGHTED_PATTERN.match(line)
        if weighted_match:
            weighted = {
                "precision": float(weighted_match.group(1)),
                "recall": float(weighted_match.group(2)),
                "f1": float(weighted_match.group(3)),
                "support": int(weighted_match.group(4)),
            }
            continue

        row_match = REPORT_PATTERN.match(line)
        if row_match:
            label = row_match.group(1)
            if label in {"accuracy", "macro", "weighted"}:
                continue
            per_word.append(
                {
                    "word": label,
                    "precision": float(row_match.group(2)),
                    "recall": float(row_match.group(3)),
                    "f1": float(row_match.group(4)),
                    "support": int(row_match.group(5)),
                }
            )

    return {
        "report_path": str(report_path),
        "model": model_name,
        "words": words,
        "timestamp": timestamp,
        "scalar_metrics": scalar_metrics,
        "accuracy": accuracy,
        "macro": macro,
        "weighted": weighted,
        "per_word": per_word,
    }


def find_model_path(report_path: Path, timestamp: str) -> Path:
    model_candidates = sorted(report_path.parent.glob("*.keras"))
    if not model_candidates:
        raise FileNotFoundError(f"No .keras model found in {report_path.parent}")

    if timestamp:
        ts_matches = [p for p in model_candidates if timestamp in p.name]
        if ts_matches:
            return ts_matches[0]

    best_matches = [p for p in model_candidates if p.name.endswith("_best.keras")]
    if best_matches:
        return sorted(best_matches)[-1]

    return model_candidates[-1]


def evaluate_topk(model_path: Path, test_ds, learning_rate: float, model_type: str, num_words: int) -> dict:
    model = build_model(model_type, num_classes=num_words)
    model.load_weights(model_path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=5.0),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name="top1_acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=7, name="top7_acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name="top10_acc"),
        ],
    )
    return model.evaluate(test_ds, verbose=0, return_dict=True)


def compute_f1_distribution_stats(per_word_rows: list[dict]) -> dict:
    f1_scores = [r["f1"] for r in per_word_rows]
    rounded = [round(v, 4) for v in f1_scores]
    modes = sorted(multimode(rounded))
    mode_value = modes[0] if modes else None
    return {
        "max": float(max(f1_scores)) if f1_scores else None,
        "min": float(min(f1_scores)) if f1_scores else None,
        "median": float(median(f1_scores)) if f1_scores else None,
        "mean": float(mean(f1_scores)) if f1_scores else None,
        "mode": float(mode_value) if mode_value is not None else None,
        "zero_count": len([f for f in f1_scores if f == 0.0]),
    }


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_markdown_table(path: Path, rows: list[dict], columns: list[str]) -> None:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(row.get(c, "")) for c in columns) + " |" for row in rows]
    path.write_text("\n".join([header, sep] + body) + "\n", encoding="utf-8")


def plot_grouped_bar(output_path: Path, models: list[str], series: dict[str, list[float]], title: str) -> None:
    labels = list(series.keys())
    x = np.arange(len(models))
    width = 0.8 / max(len(labels), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 8))
    for idx, key in enumerate(labels):
        offset = (idx - (len(labels) - 1) / 2) * width
        ax.bar(x + offset, series[key], width=width, label=key)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_grouped_barh(output_path: Path, models: list[str], series: dict[str, list[float]], title: str) -> None:
    labels = list(series.keys())
    y = np.arange(len(models))
    height = 0.8 / max(len(labels), 1)

    fig, ax = plt.subplots(figsize=(10, max(6, len(models) * 1.5)))
    for idx, key in enumerate(labels):
        offset = (idx - (len(labels) - 1) / 2) * height
        ax.barh(y + offset, series[key], height=height, label=key)

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1.0)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    root = Path(ROOT_DIR).resolve()
    output_dir = Path(OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / JSON_FILE_NAME 

    if json_path.exists():
        print(f"Loading existing results from {json_path}")
        with json_path.open("r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        report_paths = sorted(root.glob("*/*_report.txt"))
        report_paths = [p for p in report_paths if "old" not in p.parts]

        if not report_paths:
            raise FileNotFoundError(f"No report files found under {root}")

        pipeline = ASLDataPipeline(
            dataset_dir=DATASET_DIR,
            num_words=NUM_WORDS,
            batch_size=BATCH_SIZE,
            val_split=VAL_SPLIT,
            test_split=TEST_SPLIT,
            seed=SEED,
            output_dir=str(root),
        )
        _, val_ds, test_ds = pipeline.get_datasets(
            use_tfrecord=USE_TFRECORD,
            augment=USE_AUGMENTATION,
        )
        if test_ds is None:
            test_ds = val_ds

        results = []
        for report_path in report_paths:
            
            print(f"[REPORT PATH] {report_path}")
            
            parsed = parse_report(report_path)
            model_path = find_model_path(report_path, parsed.get("timestamp", ""))
            topk = evaluate_topk(model_path, test_ds, LEARNING_RATE, parsed["model"], NUM_WORDS)

            f1_stats = compute_f1_distribution_stats(parsed["per_word"])

            row = {
                "model": parsed["model"],
                "model_dir": report_path.parent.name,
                "report_path": str(report_path),
                "model_path": str(model_path),
                "accuracy": parsed["accuracy"]["accuracy"] if parsed["accuracy"] else None,
                "precision": parsed["macro"]["precision"] if parsed["macro"] else None,
                "f1": parsed["macro"]["f1"] if parsed["macro"] else None,
                "weighted_f1": parsed["weighted"]["f1"] if parsed["weighted"] else None,
                "top1_acc": float(topk.get("top1_acc", np.nan)),
                "top3_acc": float(topk.get("top3_acc", np.nan)),
                "top5_acc": float(topk.get("top5_acc", np.nan)),
                "top10_acc": float(topk.get("top10_acc", np.nan)),
                "loss": float(topk.get("loss", np.nan)),
                "f1_word_max": f1_stats["max"],
                "f1_word_min": f1_stats["min"],
                "f1_word_median": f1_stats["median"],
                "f1_word_mean": f1_stats["mean"],
                "f1_word_mode": f1_stats["mode"],
                "f1_word_zero_count": f1_stats["zero_count"],
                "per_word_count": len(parsed["per_word"]),
            }
            results.append({"summary": row, "parsed_report": parsed, "topk_eval": topk})
            print(
                f"[DONE] {row['model_dir']}: "
                f"acc={row['accuracy']:.4f} macro_f1={row['f1']:.4f} "
                f"weighted_f1={row['weighted_f1']:.4f} mean_f1={row['f1_word_mean']:.4f} "
                f"0-F1_classes={row['f1_word_zero_count']} "
                f"top1={row['top1_acc']:.4f} top3={row['top3_acc']:.4f} "
                f"top5={row['top5_acc']:.4f} top10={row['top10_acc']:.4f}"
            )
            
            # Clear keras session to avoid memory leak with many models
            tf.keras.backend.clear_session()
            import gc
            gc.collect()

        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary_rows = [r["summary"] for r in results]
    summary_rows.sort(key=lambda x: (x["f1"] if x["f1"] is not None else -1), reverse=True)

    summary_csv = output_dir / "dataset3_500w_summary.csv"
    summary_fields = [
        "model",
        "model_dir",
        "accuracy",
        "precision",
        "f1",
        "weighted_f1",
        "top1_acc",
        "top3_acc",
        "top5_acc",
        "top10_acc",
        "loss",
        "f1_word_max",
        "f1_word_min",
        "f1_word_median",
        "f1_word_mean",
        "f1_word_mode",
        "f1_word_zero_count",
        "per_word_count",
        "report_path",
        "model_path",
    ]
    save_csv(summary_csv, summary_rows, summary_fields)

    md_summary = output_dir / "dataset3_500w_summary.md"
    md_columns = [
        "model",
        "accuracy",
        "precision",
        "f1",
        "weighted_f1",
        "top1_acc",
        "top3_acc",
        "top5_acc",
        "top10_acc",
        "f1_word_max",
        "f1_word_min",
        "f1_word_median",
        "f1_word_mean",
        "f1_word_mode",
        "f1_word_zero_count",
    ]
    rounded_rows = []
    for row in summary_rows:
        rounded = dict(row)
        for col in [
            "accuracy",
            "precision",
            "f1",
            "weighted_f1",
            "top1_acc",
            "top3_acc",
            "top5_acc",
            "top10_acc",
            "f1_word_max",
            "f1_word_min",
            "f1_word_median",
            "f1_word_mean",
            "f1_word_mode",
            "f1_word_zero_count",
        ]:
            val = rounded.get(col)
            rounded[col] = "" if val is None else f"{val:.4f}"
        rounded_rows.append(rounded)
    save_markdown_table(md_summary, rounded_rows, md_columns)

    models = [r["model"] for r in summary_rows]
    # plot_grouped_barh(
    plot_grouped_bar(
        output_dir / "f1_precision_accuracy_comparison.png",
        models,
        {
            "F1": [r["f1"] for r in summary_rows],
            "Precision": [r["precision"] for r in summary_rows],
            "Accuracy": [r["accuracy"] for r in summary_rows],
        },
        "Dataset3.0 All Words: F1 vs Precision vs Accuracy",
    )
    # plot_grouped_barh(
    plot_grouped_bar(
        output_dir / "topk_comparison.png",
        models,
        {
            "Top-1": [r["top1_acc"] for r in summary_rows],
            "Top-3": [r["top3_acc"] for r in summary_rows],
            "Top-5": [r["top5_acc"] for r in summary_rows],
            "Top-10": [r["top10_acc"] for r in summary_rows],
        },
        "Dataset3.0 All Words: Top-K Accuracy",
    )
    # plot_grouped_barh(
    plot_grouped_bar(
        output_dir / "f1_distribution_stats_comparison.png",
        models,
        {
            "F1 Min": [r["f1_word_min"] for r in summary_rows],
            "F1 Mean": [r["f1_word_mean"] for r in summary_rows],
            "F1 Median": [r["f1_word_median"] for r in summary_rows],
            "F1 Mode": [r["f1_word_mode"] for r in summary_rows],
            "F1 Max": [r["f1_word_max"] for r in summary_rows],
        },
        "Dataset3.0 All Words: Per-Word F1 Distribution Stats",
    )

    print("\nSaved outputs:")
    print(f"- {json_path}")
    print(f"- {summary_csv}")
    print(f"- {md_summary}")
    print(f"- {output_dir / 'f1_precision_accuracy_comparison.png'}")
    print(f"- {output_dir / 'topk_comparison.png'}")
    print(f"- {output_dir / 'f1_distribution_stats_comparison.png'}")


if __name__ == "__main__":
    main()