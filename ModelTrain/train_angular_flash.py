
import datetime
import numpy as np
import tensorflow as tf
# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')
import matplotlib.pyplot as plt
import os
from data_pipeline import ASLDataPipeline
from model import build_model, get_callbacks




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
    # ── GPU ──
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # ── Data pipeline ──
    output_dir = f"./dataset3.0/{NUM_WORDS or 'all'}words"
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=5.0),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
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
    results = model.evaluate(test_ds)
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
    # ── GPU ──
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # ── Data pipeline ──
    output_dir = f"./dataset3.0/{NUM_WORDS or 'all'}words"
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
    results = model.evaluate(test_ds)
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



if __name__ == "__main__":

    DATASET_DIR = "../ExtractLandmarks/dataset3.0/landmarks_npz"
    NUM_WORDS = 500
    BATCH_SIZE = 64  # 64 for bigru v2  .... 32 for dualpath (reduced from 128 to avoid CudnnRNNV3 memory issues)
    EPOCHS = 1000
    VAL_SPLIT = 0.20
    TEST_SPLIT = 0.00
    SEED = 1234
    LEARNING_RATE = 1e-4  # 1e-4 for bigru_bigger v1  ..... 1e-3 for rest
    PATIENCE = 25
    USE_TFRECORD = True
    USE_CLASS_WEIGHTS = True
    USE_AUGMENTATION = True
    
    # ── [1] bigru_velocity_v1 on 500 words ──
    MODEL_TYPE = "bigru_velocity_v1"
    BATCH_SIZE = 64
    
    main(
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
        # RESUME_MODEL_PATH
    )
    
    # Free GPU/RAM between runs to prevent OOM crash
    tf.keras.backend.clear_session()
    import gc; gc.collect()

    # ── [2] bigru_velocity_biggest_v1 on ALL 2731 words ──
    NUM_WORDS = None
    MODEL_TYPE = "bigru_velocity_biggest_v1"
    BATCH_SIZE = 64
    
    main(
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
        # RESUME_MODEL_PATH
    )
    
    # Free GPU/RAM between runs to prevent OOM crash
    tf.keras.backend.clear_session()
    import gc; gc.collect()

    # ── [3] Resume bigru_biggest_angular_flash_v1 on ALL 2731 words ──
    # NUM_WORDS = None
    # MODEL_TYPE = "bigru_biggest_angular_flash_v1"
    # BATCH_SIZE = 64
    # RESUME_MODEL_PATH = "dataset3.0/allwords/bigru_biggest_angular_flash_v1_aug/asl_bigru_biggest_angular_flash_v1_aug_allw_04-05-26__07-07_best.keras"
    # 
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
    #     MODEL_TYPE,
    #     RESUME_MODEL_PATH
    # )