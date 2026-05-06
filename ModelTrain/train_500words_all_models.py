"""
Training script for ASL recognition — dataset 3.0.

Config is set directly below. Change MODEL_TYPE to try different architectures.

Available models:
    "original"   — Your proven BiGRU (80% baseline)
    "bigru_v2"   — BiGRU + LayerNorm + wider head
    "conv_bigru" — Conv1D → BiGRU hybrid
    "tcn"        — Temporal Convolutional Network (dilated convolutions)
    "conv1d"     — Pure Conv1D (fastest, lightweight)
"""

import datetime
import numpy as np
import tensorflow as tf
# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')
import matplotlib.pyplot as plt
import os
from data_pipeline import ASLDataPipeline

from model import build_model, get_callbacks, MODEL_REGISTRY
import matplotlib
matplotlib.use('Agg') # Headless mode for stability


class LabelSmoothingSparseCCE(tf.keras.losses.Loss):
    """SparseCategoricalCrossentropy with label smoothing.

    Keras' SparseCategoricalCrossentropy doesn't support label_smoothing,
    so we convert to one-hot and use CategoricalCrossentropy internally.

    smoothing=0.1 spreads 10% of probability mass uniformly across all classes,
    preventing the model from becoming overconfident on training examples.
    """
    def __init__(self, smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.smoothing = smoothing

    def call(self, y_true, y_pred):
        n = tf.cast(tf.shape(y_pred)[-1], tf.float32)
        y_oh = tf.one_hot(tf.cast(y_true, tf.int32), tf.shape(y_pred)[-1])
        y_smooth = y_oh * (1.0 - self.smoothing) + (self.smoothing / n)
        return tf.keras.losses.categorical_crossentropy(y_smooth, y_pred)

    def get_config(self):
        return {**super().get_config(), "smoothing": self.smoothing}


# Global GPU config — must only run once
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


# ══════════════════════════════════════════════════════════════════════
# CONFIG — edit these directly
# ══════════════════════════════════════════════════════════════════════

"""
Model	Params	How it works	Speed
original	1.34M	Your proven BiGRU (80% baseline)	Medium
bigru_v2	1.41M	Same + LayerNorm + wider head	Medium
conv_bigru	0.41M	Conv1D extracts local motion → GRU reads globally	Medium
tcn	0.52M	Dilated convolutions cover full 128 frames	Fast
conv1d	0.31M	Pure convolutions, simplest deep model	Fastest

"""



# ══════════════════════════════════════════════════════════════════════


def plot_training_history(history, save_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history["accuracy"], label="Train", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Val", linewidth=2)
    if "top5_acc" in history.history:
        ax1.plot(
            history.history["top5_acc"],
            label="Train Top-5",
            linewidth=1.5,
            linestyle="--",
        )
        ax1.plot(
            history.history["val_top5_acc"],
            label="Val Top-5",
            linewidth=1.5,
            linestyle="--",
        )
    ax1.set_title("Accuracy", fontsize=14)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["loss"], label="Train", linewidth=2, color="red")
    ax2.plot(history.history["val_loss"], label="Val", linewidth=2, color="orange")
    ax2.set_title("Loss", fontsize=14)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved to {save_path}")
    plt.close()


def main(
    DATASET_DIR,
    NUM_WORDS,
    BATCH_SIZE,
    EPOCHS,
    VAL_SPLIT,
    TEST_SPLIT,
    SEED,
    LEARNING_RATE,
    PATIENCE,
    USE_TFRECORD,
    USE_CLASS_WEIGHTS,
    USE_AUGMENTATION,
    MODEL_TYPE,
    RESUME_MODEL_PATH=None
):
    # GPU config is handled at global scope

    # ── Data pipeline ──
    output_dir = f"./dataset4.0/{NUM_WORDS or 'all'}words"
    pipeline = ASLDataPipeline(
        dataset_dir=DATASET_DIR,
        num_words=NUM_WORDS,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        seed=SEED,
        output_dir=output_dir,
    )

    train_ds, val_ds, test_ds = pipeline.get_datasets(
        use_tfrecord=USE_TFRECORD, augment=USE_AUGMENTATION
    )
    train_steps, val_steps = pipeline.get_steps_per_epoch()
    num_classes = pipeline.num_classes

    print(f"\n{'=' * 60}")
    print(f"Training config:")
    print(f"  Words:         {num_classes}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Epochs:        {EPOCHS}")
    print(f"  LR:            {LEARNING_RATE}")
    print(f"  Model:         {MODEL_TYPE}")
    print(f"  Augmentation:  {USE_AUGMENTATION}")
    print(f"  Train steps:   {train_steps}")
    print(f"  Val steps:     {val_steps}")
    print(f"{'=' * 60}\n")

    # ── Build model ──
    model = build_model(MODEL_TYPE, num_classes=num_classes)

    if RESUME_MODEL_PATH and os.path.exists(RESUME_MODEL_PATH):
        print(f"\n[INFO] Resuming training from weights: {RESUME_MODEL_PATH}\n")
        model.load_weights(RESUME_MODEL_PATH)
    elif RESUME_MODEL_PATH:
        print(f"\n[WARNING] Could not find resume path: {RESUME_MODEL_PATH}. Starting from scratch.\n")

    # ── Compile ──
    model.compile(
        # AdamW: decouples weight decay from adaptive LR update (mathematically correct
        # regularization). weight_decay=1e-4 targets the ~5× train/val loss gap.
        optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-4, clipnorm=5.0),
        # Label smoothing (ε=0.1): prevents overconfident train predictions.
        # Complements AdamW — AdamW controls weight magnitude, this controls output confidence.
        loss=LabelSmoothingSparseCCE(smoothing=0.1),
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc"),
        ],
    )

    model.summary()

    # ── Callbacks ──
    timestamp = datetime.datetime.now().strftime("%d-%m-%y__%H-%M")
    base_model_name = f"{MODEL_TYPE}_aug" if USE_AUGMENTATION else MODEL_TYPE
    model_name = f"{base_model_name}_{NUM_WORDS or 'all'}w_{timestamp}"
    model_dir = f"{output_dir}/{base_model_name}"
    callbacks = get_callbacks(
        model_dir=model_dir,
        model_name=model_name,
        patience=PATIENCE,
    )

    # ── Class weights ──
    class_weights = None
    if USE_CLASS_WEIGHTS:
        class_weights = pipeline.get_class_weights()

    # ── Train ──
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    # ── Plot ──
    plot_training_history(history, f"{model_dir}/{model_name}_curves.png")

    # ── Evaluate on test set ──
    if test_ds is None:
        test_ds = val_ds

    print("\n" + "=" * 60)
    print("Test set evaluation:")
    results = model.evaluate(test_ds, steps=val_steps)

    for name, val in zip(model.metrics_names, results):
        print(f"  {name}: {val:.4f}")

    # ── Classification report (F1, precision, recall) ──
    print("\n" + "=" * 60)
    print("Generating classification report...")

    from sklearn.metrics import classification_report
    import json

    # Collect predictions from test set
    y_true = []
    y_pred = []
    for batch_inputs, batch_labels in test_ds:
        preds = model.predict(batch_inputs, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(batch_labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Load word names for the report
    target_names = [pipeline.ind_to_word[i] for i in range(num_classes)]

    report_str = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    print(report_str)

    # Save report to file

    print(f"Model: {MODEL_TYPE}\n")
    print(f"Words: {NUM_WORDS or 'all'}\n")
    print(f"Timestamp: {timestamp}\n")
    print(f"\n{'=' * 60}\n")
    # for name, val in zip(model.metrics_names, results):
    #     print(f"{name}: {val:.4f}\n")
    print(f"\n{'=' * 60}\n")
    print(report_str)
    report_path = f"{model_dir}/{model_name}_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Model: {MODEL_TYPE}\n")
        f.write(f"Words: {NUM_WORDS or 'all'}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"\n{'=' * 60}\n")
        for name, val in zip(model.metrics_names, results):
            f.write(f"{name}: {val:.4f}\n")
        f.write(f"\n{'=' * 60}\n")
        f.write(str(report_str))
    print(f"Report saved to: {report_path}")

    print(f"\nModel saved to: {model_dir}")
    print(f"Label mapping:  {output_dir}/word_to_ind_{NUM_WORDS or 'all'}.json")


def get_report(
    DATASET_DIR,
    NUM_WORDS,
    BATCH_SIZE,
    EPOCHS,
    VAL_SPLIT,
    TEST_SPLIT,
    SEED,
    LEARNING_RATE,
    USE_TFRECORD,
    USE_AUGMENTATION,
    MODEL_TYPE,
    MODEL_PATH,
    SHOW_ALL_WORDS =False
):
    # GPU config is handled at global scope

    # ── Data pipeline ──
    output_dir = f"./dataset4.0/{NUM_WORDS or 'all'}words"
    pipeline = ASLDataPipeline(
        dataset_dir=DATASET_DIR,
        num_words=NUM_WORDS,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        seed=SEED,
        output_dir=output_dir,
    )

    train_ds, val_ds, test_ds = pipeline.get_datasets(
        use_tfrecord=USE_TFRECORD, augment=USE_AUGMENTATION
    )
    train_steps, val_steps = pipeline.get_steps_per_epoch()
    num_classes = pipeline.num_classes

    print(f"\n{'=' * 60}")
    print(f"Training config:")
    print(f"  Words:         {num_classes}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  Epochs:        {EPOCHS}")
    print(f"  LR:            {LEARNING_RATE}")
    print(f"  Model:         {MODEL_TYPE}")
    print(f"  Augmentation:  {USE_AUGMENTATION}")
    print(f"  Train steps:   {train_steps}")
    print(f"  Val steps:     {val_steps}")
    print(f"{'=' * 60}\n")

    # ── Build model ──
    
    model = tf.keras.models.load_model(MODEL_PATH)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=5.0),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=7, name="top7_acc"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name="top10_acc"),
        ],
    )
    model.summary()
    

    # ── Callbacks ──
    timestamp = datetime.datetime.now().strftime("%d-%m-%y__%H-%M")
    base_model_name = f"{MODEL_TYPE}_aug" if USE_AUGMENTATION else MODEL_TYPE
    model_dir = f"{output_dir}/{base_model_name}"
    # ── Evaluate on test set ──
    if test_ds is None:
        test_ds = val_ds

    print("\n" + "=" * 60)
    print("Test set evaluation:")
    results = model.evaluate(test_ds, steps=val_steps)

    for name, val in zip(model.metrics_names, results):
        print(f"  {name}: {val:.4f}")

    # ── Classification report (F1, precision, recall) ──
    print("\n" + "=" * 60)
    print("Generating classification report...")

    from sklearn.metrics import classification_report

    # Collect predictions from test set
    y_true = []
    y_pred = []
    for batch_inputs, batch_labels in test_ds:
        preds = model.predict(batch_inputs, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(batch_labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Load word names for the report
    target_names = [pipeline.ind_to_word[i] for i in range(num_classes)]

    report_str = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0
    )
    # print(report_str)

    # Save report to file

    print(f"Model: {MODEL_TYPE}\n")
    print(f"Words: {NUM_WORDS or 'all'}\n")
    print(f"Timestamp: {timestamp}\n")
    print(f"\n{'=' * 60}\n")
    # for name, val in zip(model.metrics_names, results):
    #     print(f"{name}: {val:.4f}\n")
    print(f"\n{'=' * 60}\n")
    print(report_str)
    print(f"Label mapping:  {output_dir}/word_to_ind_{NUM_WORDS or 'all'}.json")



if __name__ == '__main______':
    # MODEL_PATH = "./dataset4.0/allwords/asl_bigru_bigger_v1_aug_allw_02-05-26__03-11_alltimebest.keras"
    DATASET_DIR = "../ExtractLandmarks/dataset4.0/landmarks_npz"
    NUM_WORDS = 500
    BATCH_SIZE = 256  # 64 for bigru v2  .... 32 for dualpath (reduced from 128 to avoid CudnnRNNV3 memory issues)
    EPOCHS = 1000
    VAL_SPLIT = 0.20
    TEST_SPLIT = 0.00
    SEED = 1234
    LEARNING_RATE = 1e-4  # 1e-4 for bigru_bigger v1  ..... 1e-3 for rest
    PATIENCE = 10
    USE_TFRECORD = True
    USE_CLASS_WEIGHTS = True
    USE_AUGMENTATION = True
    # ── Pick your model ──
    # "original" | "bigru_v2" | "bigru_v3" | "bigru_bigger_v1" | "bigru_bigger_v2" | "conv_bigru" | "tcn" | "conv1d" | "dualpath" | "dualpath_v2"
    MODEL_TYPE = "bigru_bigger_v1"
    # MODEL_TYPE = "conv_bigru"
    # MODEL_TYPE = "dualpath"
    # MODEL_TYPE = "dualpath_v2"

    get_report(
        DATASET_DIR,
        NUM_WORDS,
        BATCH_SIZE,
        EPOCHS,
        VAL_SPLIT,
        TEST_SPLIT,
        SEED,
        LEARNING_RATE,
        USE_TFRECORD,
        USE_AUGMENTATION,
        MODEL_TYPE,
        MODEL_PATH)
        # RESUME_MODEL_PATH
 



# MODEL_REGISTRY is imported from model.py






if __name__ == "__main__":
    DATASET_DIR = "../ExtractLandmarks/dataset4.0/landmarks_npz"
    NUM_WORDS = 500
    EPOCHS = 1000
    VAL_SPLIT = 0.20
    TEST_SPLIT = 0.00
    SEED = 1234
    LEARNING_RATE = 1e-4
    PATIENCE = 10
    USE_TFRECORD = True
    USE_CLASS_WEIGHTS = True
    USE_AUGMENTATION = True

    # List of models we want to train in this batch
    models_to_train = [
        # "original",
        # "bigru_v2",
        # "bigru_flash",
        # "bigru_angular_v1",
        # "bigru_bigger_v1",
        # "bigru_bigger_v2",
        # "bigru_bigger_angular_v1",
        # "bigru_bigger_angular_flash_v1",
        # "conv_bigru",
        # "conv_only_v2", #skipping due to gpu not being strong
        # "conv_bigru_v3",
        # "tcn",
        # "conv1d",
        # "dualpath",
        # "dualpath_v2"
    ]

    for model_type in models_to_train:
        print(f"\n\n{'='*60}")
        print(f" TRAINING MODEL: {model_type}")
        print(f"{'='*60}\n")

        # Set suitable batch size (DualPath is memory intensive)
        batch_size = 64

        if "bigger" in model_type:
            batch_size = 128
        elif "biggest" in model_type:
            batch_size=64
        else:
            batch_size=256
        if "dualpath" in model_type:
            batch_size = 32
        if "conv" in model_type:
            batch_size=64
        if "tcn" in model_type:
            batch_size=2
        
        try:
            main(
                DATASET_DIR,
                NUM_WORDS,
                batch_size,
                EPOCHS,
                VAL_SPLIT,
                TEST_SPLIT,
                SEED,
                LEARNING_RATE,
                PATIENCE,
                USE_TFRECORD,
                USE_CLASS_WEIGHTS,
                USE_AUGMENTATION,
                model_type
            )
        except Exception as e:
            print(f"CRITICAL ERROR training {model_type}: {e}")
        finally:
            # IMPORTANT: Release GPU memory and reset state for the next model
            print(f"Cleaning up session for {model_type}...")
            tf.keras.backend.clear_session()
            import gc
            gc.collect()

    print("\n\nAll training runs complete!")


    DATASET_DIR = "../ExtractLandmarks/dataset4.0/landmarks_npz"
    NUM_WORDS = None
    EPOCHS = 1000
    VAL_SPLIT = 0.20
    TEST_SPLIT = 0.00
    SEED = 1234
    LEARNING_RATE = 1e-4
    PATIENCE = 10
    USE_TFRECORD = True
    USE_CLASS_WEIGHTS = True
    USE_AUGMENTATION = True

    # List of models we want to train in this batch
    models_to_train = [
        # "original",
        # "bigru_v2",  ## done
        # "bigru_flash",
        # "bigru_angular_v1",
        "bigru_bigger_v1",
        "bigru_bigger_v2",
        "bigru_bigger_angular_v1",
        # "bigru_bigger_angular_flash_v1",
        # "conv_bigru",
        # "conv_only_v2",
        # "conv_bigru_v3",
        # "tcn",
        # "conv1d",
        # "dualpath",
        # "dualpath_v2",
        "bigru_biggest_v1"
    ]

    for model_type in models_to_train:
        print(f"\n\n{'='*60}")
        print(f" TRAINING MODEL: {model_type}")
        print(f"{'='*60}\n")

        # Set suitable batch size (DualPath is memory intensive)
        batch_size = 256

        if "biggest" in model_type:
            batch_size=32
        if "angular" in model_type:
            batch_size=64
        try:
            main(
                DATASET_DIR,
                NUM_WORDS,
                batch_size,
                EPOCHS,
                VAL_SPLIT,
                TEST_SPLIT,
                SEED,
                LEARNING_RATE,
                PATIENCE,
                USE_TFRECORD,
                USE_CLASS_WEIGHTS,
                USE_AUGMENTATION,
                model_type
            )
        except Exception as e:
            print(f"CRITICAL ERROR training {model_type}: {e}")
        finally:
            # IMPORTANT: Release GPU memory and reset state for the next model
            print(f"Cleaning up session for {model_type}...")
            tf.keras.backend.clear_session()
            import gc
            gc.collect()

    print("\n\nAll training runs complete!")

    # main(
    #     DATASET_DIR,
    #     NUM_WORDS,
    #     BATCH_SIZE,
    #     EPOCHS,
    #     VAL_SPLIT,
    #     TEST_SPLIT,
    #     SEED,
    #     LEARNING_RATE,
    #     PATIENCE,
    #     USE_TFRECORD,
    #     USE_CLASS_WEIGHTS,
    #     USE_AUGMENTATION,
    #     MODEL_TYPE
    # )
    # """broooo 85% ke around achieve hogyi h bigru bigger se on 500words"""
