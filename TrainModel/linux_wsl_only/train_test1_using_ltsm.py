"""activate venv =>


source /home/aryan/opensource_lab_proj/venv/bin/activate
"""

"""
installed using this guide
https://www.tensorflow.org/install/pip#windows-wsl2_1
https://developer.nvidia.com/cuda-12-3-2-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
"""


import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model  # type: ignore
from tensorflow.keras.layers import Softmax, LSTM, Dense, Dropout, Bidirectional, GRU, LayerNormalization, Input, Reshape, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Multiply, Activation  # type: ignore
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard  # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import optimizers
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.utils.class_weight import compute_class_weight  




modelpath = "../models"
landmarkpath = "../gte9_landmarks"


modelnum = 0
with open("../models/max_model_num.txt","r") as f:
    modelnum = int(f.read().strip())

    
print(modelnum)
# GPU CONFIG
gpus = tf.config.list_physical_devices("GPU")
print(gpus)

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ GPU detected: {gpus[0].name}")
        
        # ENABLE MIXED PRECISION for faster GPU training (RTX 4050 supports FP16)
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"✅ Mixed precision enabled: {policy.name}")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ No GPU detected, running on CPU.")

SEQUENCE_LENGTH = 70

"""
BHAIIIII 57 ko 63 krna hi bhool gaya
sochra hu ye 1% kyu aari h aur loss itta saara
crazy
"""
FEATURE_DIM = 63 * 4  # (x, y, z, vis) for 57 landmarks
EPOCHS = 200
BATCH_SIZE = 16  # INCREASED from 16 → GPU needs larger batches for efficiency (still safe for 5000 samples)
TRAIN_SPLIT = 0.25  # Use 85% for training, 15% for validation (with small data, need more training)  

def turn_data_into_readable_for_ltsm(
    dataset_landmark_path=landmarkpath,
):
    X, y = [], []

    for word in os.listdir(dataset_landmark_path):

        word_folder = os.path.join(dataset_landmark_path, word)

        if os.path.isdir(word_folder):
            for npfile in os.listdir(word_folder):
                if npfile.endswith(".npy"):
                    # shape  =  70, 57, 4
                    arr = np.load(os.path.join(word_folder, npfile))

                    # new shape = 70,57*4  = 70,228
                    arr = arr.reshape(arr.shape[0], -1)
                    print(arr.shape)
                    X.append(arr)
                    y.append(word)
    return np.array(X, dtype=np.float32), np.array(y)


if __name__ == "__main__":
    label_encoder = LabelEncoder()

    if os.path.exists(landmarkpath+"/x.npy") and os.path.exists(
        landmarkpath+"/y_onehot.npy"
    ):
        x = np.load(landmarkpath+"/x.npy")
        y = np.load(landmarkpath+"/y.npy")
        y_encoded = np.load(landmarkpath+"/y_encoded.npy")
        y_onehot = np.load(landmarkpath+"/y_onehot.npy")
        label_encoder.fit(y)

    else:
        x, y = turn_data_into_readable_for_ltsm()
        print(y)
        print(x)

        y_encoded = label_encoder.fit_transform(y)
        y_onehot = to_categorical(y_encoded)

        print(y.shape)
        print(y[0].shape)
        np.save(landmarkpath+"/x.npy", x)
        np.save(landmarkpath+"/y.npy", y)
        np.save(landmarkpath+"/y_encoded.npy", y_encoded)
        np.save(landmarkpath+"/y_onehot.npy", y_onehot)

    X_train, X_test, y_train, y_test = train_test_split(
        x, y_onehot, test_size=TRAIN_SPLIT, random_state=42, stratify=y_encoded
    )
    

    
    # # ==================== DATA AUGMENTATION ====================
    # print("\n🔄 Applying data augmentation (for small dataset)...")
    # X_train_augmented = [X_train]
    # y_train_augmented = [y_train]
    
    # # Temporal jittering: add small random noise to simulate slight timing variations
    # for _ in range(2):  # 2x augmentation
    #     noise = np.random.normal(0, 0.02, X_train.shape)  # Small Gaussian noise
    #     X_train_augmented.append(X_train + noise)
    #     y_train_augmented.append(y_train)
    
    # X_train = np.vstack(X_train_augmented)
    # y_train = np.vstack(y_train_augmented)
    
    # print(f"✅ Data augmented: {X_train.shape[0]} samples (3x original)")
    # ===========================================================
    # model = Sequential(
    #     [
    #         LSTM(
    #             128, return_sequences = True,input_shape=(SEQUENCE_LENGTH, FEATURE_DIM)
    #         ),
    #         Dropout(0.3),
    #         LSTM(64),
    #         Dropout(0.3),

    #         Dense(64, activation="relu"),
    #         Dense(len(set(y)), activation="softmax"),
    #     ]
    # )

    def LSTM128to64():
        m = Sequential(
            [
                LSTM(
                    128,
                    return_sequences=True,
                    input_shape=(SEQUENCE_LENGTH, FEATURE_DIM),
                ),
                Dropout(0.4),
                LSTM(64),
                Dropout(0.3),
                Dense(64, activation="relu"),
                Dense(len(label_encoder.classes_), activation="softmax"),
            ]
        )
        return m
    
    def GRUmodel():
        model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM)),

        Bidirectional(GRU(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)),
        LayerNormalization(),
        Dropout(0.15),

        Bidirectional(GRU(64, dropout=0.1, recurrent_dropout=0.1)),
        LayerNormalization(),
        Dropout(0.15),

        Dense(128, activation="relu"),
        Dropout(0.1),

        Dense(len(label_encoder.classes_), activation="softmax"),
        ])
        return model

    def SmallGRU():
        return Sequential([
            Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM)),
            GRU(128, return_sequences=True),
            LayerNormalization(),
            Dropout(0.3),

            GRU(64),
            LayerNormalization(),
            Dropout(0.3),

            Dense(128, activation="relu"),
            Dropout(0.2),

            Dense(len(label_encoder.classes_), activation="softmax")
        ])
        
    def AttentionBlock(inputs):
        
        score = Dense(1)(inputs)                 # (batch, timesteps, 1)
        weights = Softmax(axis=1)(score)         # softmax over time dimension
        out = Multiply()([inputs, weights])      # weighted sequence
        return out

    def SpecialModel():

        inp = Input(shape=(70, 252))

        # reshape for small spatial conv
        x = Reshape((70, 63, 4))(inp)

        # conv encoder
        x = Conv2D(32, (3,3), padding="same", activation="relu")(x)
        x = MaxPooling2D((1,2))(x)

        x = Conv2D(64, (3,3), padding="same", activation="relu")(x)
        x = MaxPooling2D((1,2))(x)

        x = Conv2D(128, (3,3), padding="same", activation="relu")(x)
        x = GlobalAveragePooling2D()(x)
        x = LayerNormalization()(x)

        # temporal modelling
        y = GRU(128, return_sequences=True)(inp)
        y = LayerNormalization()(y)
        y = Dropout(0.3)(y)

        y = GRU(64, return_sequences=True)(y)
        y = LayerNormalization()(y)
        y = Dropout(0.3)(y)

        # attention
        y = AttentionBlock(y)
        y = GRU(64)(y)

        # fuse
        fused = tf.keras.layers.Concatenate()([x, y])

        fused = Dense(128, activation="relu")(fused)
        fused = Dropout(0.3)(fused)

        out = Dense(len(label_encoder.classes_), activation="softmax")(fused)

        return Model(inputs=inp, outputs=out)
    
    print(len(label_encoder.classes_))
    # exit()
    model = LSTM128to64()
    
    optimizer = optimizers.AdamW(
        learning_rate=1e-4,   # VERY LOW: Slower learning for small dataset
        weight_decay=5e-3     # VERY HIGH: Strong L2 regularization
    )

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    

    early_stop = EarlyStopping(
        monitor="val_loss", 
        patience=25,
        restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", 
        factor=0.6,           
        patience=7,          
        min_lr=1e-7
    )
    
    checkpoint = ModelCheckpoint(
        modelpath+"/curr_best_pose_model.keras",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False
    )
    
    def warmup_scheduler(epoch, lr):
        warmup_epochs = 5
        target_lr = 1e-4
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs * target_lr
        return lr

    warmup_callback = tf.keras.callbacks.LearningRateScheduler(warmup_scheduler)
        
    model.summary()
    
    # ==================== COMPUTE CLASS WEIGHTS ====================
    y_train_labels = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train_labels),
        y=y_train_labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"✅ Class weights computed (min={min(class_weights):.3f}, max={max(class_weights):.3f})")
    # ===========================================================

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stop, reduce_lr,checkpoint,warmup_callback],
        class_weight=class_weight_dict,  # ADD THIS
        verbose=1,
    )

    # -------------------- EVALUATE --------------------
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    stopped_epoch = early_stop.stopped_epoch + 1
    print(f"\n✅ Test Accuracy: {acc*100:.2f}%")

    # -------------------- PLOT TRAINING HISTORY --------------------
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss")
    plt.savefig("training_curves.png")
    plt.close()

    # -------------------- CONFUSION MATRIX --------------------
    class_names = list(label_encoder.classes_)

    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true_labels, y_pred_labels)
    print("\nConfusion Matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation="vertical", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # -------------------- SAVE MODEL --------------------
    from datetime import datetime

    timestamp = datetime.now().strftime("%H-%M_%d-%m-%y")

    model_name = modelpath + f"/{modelnum}_LSTM_{timestamp}_{EPOCHS}_{acc:.4f}_{TRAIN_SPLIT}_signlang_model.keras"

    model.save(model_name)
    with open("../models/max_model_num.txt","w") as f:
        f.write(str(modelnum+1))
    print("Saved:", model_name)
