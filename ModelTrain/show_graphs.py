import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================================================
# CONFIG
# =========================================================

DATASETS = [
    {
        "title": "Dataset3.0 • 500 Words",
        "json": "/home/aryan/projects_linux/WLASL_USING_MEDIAPIPE_ND_ML/Results/dataset3_500words_analysis/dataset3_500w_model_metrics.json",
    },
    {
        "title": "Dataset3.0 • 2731 Words",
        "json": "/home/aryan/projects_linux/WLASL_USING_MEDIAPIPE_ND_ML/Results/dataset3_allwords_analysis/dataset3_allw_model_metrics.json",
    },
    {
        "title": "Dataset4.0 • 500 Words",
        "json": "/home/aryan/projects_linux/WLASL_USING_MEDIAPIPE_ND_ML/Results/dataset4_500words_analysis/dataset4_500w_model_metrics.json",
    },
    {
        "title": "Dataset4.0 • 2731 Words",
        "json": "/home/aryan/projects_linux/WLASL_USING_MEDIAPIPE_ND_ML/Results/dataset4_allwords_analysis/dataset4_allw_model_metrics.json",
    },
]


# =========================================================
# GRAPH BUILDERS
# =========================================================

def plot_metric_comparison(ax, rows, dataset_title):
    models = [r["model"] for r in rows]

    f1 = [r["f1"] for r in rows]
    precision = [r["precision"] for r in rows]
    accuracy = [r["accuracy"] for r in rows]

    x = np.arange(len(models))
    width = 0.25

    ax.bar(x - width, f1, width, label="F1 Score")
    ax.bar(x, precision, width, label="Precision")
    ax.bar(x + width, accuracy, width, label="Accuracy")

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(
        f"{dataset_title}\nF1 vs Precision vs Accuracy",
        fontsize=18,
        pad=20,
        weight="bold"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")

    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=11)


def plot_topk(ax, rows, dataset_title):
    models = [r["model"] for r in rows]

    top1 = [r["top1_acc"] for r in rows]
    top3 = [r["top3_acc"] for r in rows]
    top5 = [r["top5_acc"] for r in rows]
    top10 = [r["top10_acc"] for r in rows]

    x = np.arange(len(models))
    width = 0.18

    ax.bar(x - 1.5 * width, top1, width, label="Top-1")
    ax.bar(x - 0.5 * width, top3, width, label="Top-3")
    ax.bar(x + 0.5 * width, top5, width, label="Top-5")
    ax.bar(x + 1.5 * width, top10, width, label="Top-10")

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")

    ax.set_title(
        f"{dataset_title}\nTop-K Accuracy Comparison",
        fontsize=18,
        pad=20,
        weight="bold"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")

    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=11)


def plot_f1_distribution(ax, rows, dataset_title):
    models = [r["model"] for r in rows]

    f1_min = [r["f1_word_min"] for r in rows]
    f1_mean = [r["f1_word_mean"] for r in rows]
    f1_median = [r["f1_word_median"] for r in rows]
    f1_max = [r["f1_word_max"] for r in rows]

    x = np.arange(len(models))
    width = 0.2

    ax.bar(x - 1.5 * width, f1_min, width, label="Min")
    ax.bar(x - 0.5 * width, f1_mean, width, label="Mean")
    ax.bar(x + 0.5 * width, f1_median, width, label="Median")
    ax.bar(x + 1.5 * width, f1_max, width, label="Max")

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 Score")

    ax.set_title(
        f"{dataset_title}\nPer-Word F1 Distribution",
        fontsize=18,
        pad=20,
        weight="bold"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")

    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=11)


# =========================================================
# BUILD SLIDES
# =========================================================

slides = []

for dataset in DATASETS:
    json_path = Path(dataset["json"])

    if not json_path.exists():
        print(f"[MISSING] {json_path}")
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    rows = [r["summary"] for r in results]

    rows.sort(
        key=lambda x: x["f1"] if x["f1"] is not None else -1,
        reverse=True
    )

    slides.append(
        (
            dataset["title"],
            "F1 vs Precision vs Accuracy",
            lambda ax, r=rows, t=dataset["title"]:
                plot_metric_comparison(ax, r, t)
        )
    )

    slides.append(
        (
            dataset["title"],
            "Top-K Accuracy",
            lambda ax, r=rows, t=dataset["title"]:
                plot_topk(ax, r, t)
        )
    )

    slides.append(
        (
            dataset["title"],
            "Per-Word F1 Distribution",
            lambda ax, r=rows, t=dataset["title"]:
                plot_f1_distribution(ax, r, t)
        )
    )


# =========================================================
# PRESENTATION VIEWER
# =========================================================

current_index = 0

fig, ax = plt.subplots(figsize=(16, 9))

try:
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
except:
    pass


def draw_slide():
    ax.clear()

    # remove old footer texts
    for txt in fig.texts:
        txt.remove()

    dataset_title, graph_title, graph_func = slides[current_index]

    graph_func(ax)

    footer = (
        f"Slide {current_index + 1}/{len(slides)}    "
        f"[→] Next    [←] Previous    [Q / ESC] Quit"
    )

    fig.text(
        0.5,
        0.01,
        footer,
        ha="center",
        fontsize=11,
        alpha=0.75
    )

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.96])
    fig.canvas.draw_idle()

def on_key(event):
    global current_index

    if event.key == "right":
        current_index = min(current_index + 1, len(slides) - 1)
        draw_slide()

    elif event.key == "left":
        current_index = max(current_index - 1, 0)
        draw_slide()

    elif event.key in ["q", "escape"]:
        plt.close(fig)


fig.canvas.mpl_connect("key_press_event", on_key)

draw_slide()

plt.show()
