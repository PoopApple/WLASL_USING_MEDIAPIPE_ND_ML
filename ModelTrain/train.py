import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import datetime
import matplotlib.pyplot as plt

# print(tf.config.list_physical_devices("GPU"))


def getmodel(num_classes):
    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")

    feature_per_frame = 64 * 4
    x = tf.keras.layers.Reshape((MAX_FRAMES, feature_per_frame))(input_data)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(
        x, mask=input_mask
    )

    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128))(x, mask=input_mask)

    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=[input_data, input_mask], outputs=outputs)
    return model


MAX_FRAMES = 128
EPOCHS = 50
BATCH_SIZE = 32

random_num = 1234
if __name__ == "__main__":
    dataset = np.load("./dataset2.0/dataset2-0.npz")

    X = dataset["features"]
    masks = dataset["masks"]
    y = dataset["labels"]

    num_classes = len(np.unique(y))
    print(f"Loaded {len(y)} samples across {num_classes} classes.")

    X_train, X_val, mask_train, mask_val, y_train, y_val = train_test_split(
        X, masks, y, test_size=0.2, random_state=random_num, stratify=y
    )

    model_name = datetime.datetime.now().strftime("%d-%m-%y__%H-%M")

    input_data = tf.keras.Input(shape=(MAX_FRAMES, 64, 4), name="input_data")
    input_mask = tf.keras.Input(shape=(MAX_FRAMES,), dtype=tf.bool, name="input_mask")

    feature_per_frame = 64 * 4
    x = tf.keras.layers.Reshape((MAX_FRAMES, feature_per_frame))(input_data)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True))(
        x, mask=input_mask
    )

    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128))(x, mask=input_mask)

    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=[input_data, input_mask], outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    tf.keras.utils.plot_model(
        model,
        to_file="asl_model_architecture.png",
        show_shapes=True,
        show_layer_names=True,
        expand_nested=True,
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            f"./dataset2.0/models/asl_bigru_{model_name}_best.keras",
            save_best_only=True,
            monitor="val_accuracy",
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True, monitor="val_accuracy"
        ),
        tf.keras.callbacks.CSVLogger(
            f"./dataset2.0/models/asl_bigru_{model_name}_training_log.csv",
            separator=",",
            append=False,
        ),
    ]

    history = model.fit(
        x=[X_train, mask_train],
        y=y_train,
        validation_data=([X_val, mask_val], y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    print("Generating training statistic graphs...")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(
        history.history["accuracy"], label="Train Accuracy", color="blue", linewidth=2
    )
    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy",
        color="orange",
        linewidth=2,
    )
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss", color="red", linewidth=2)
    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
        color="orange",
        linewidth=2,
    )
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("./dataset2.0/training_stats_BiGRU.png")
    print("Training curves saved as 'training_stats.png'!")
