"""activate venv =>


source /home/aryan/opensource_lab_proj/venv/bin/activate
"""

"""

1️⃣ Subtract midpoint of shoulders as origin
2️⃣ Divide by shoulder distance
3️⃣ Optionally, rotate so shoulders are level


add poses of hands




"""


"""

1️⃣ Subtract midpoint of shoulders as origin
2️⃣ Divide by shoulder distance
3️⃣ Optionally, rotate so shoulders are level


add poses of hands




"""



"""
installed using this guide
https://www.tensorflow.org/install/pip#windows-wsl2_1
https://developer.nvidia.com/cuda-12-3-2-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
"""


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


SEQUENCE_LENGTH = 70                        # frames per video
FEATURE_DIM = 57 * 4                        # (x, y, z, vis) for 57 landmarks
EPOCHS = 100    
BATCH_SIZE = 16
TRAIN_SPLIT=0.3


# GPU CONFIG
gpus = tf.config.list_physical_devices('GPU')
print(gpus)

if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ GPU detected: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ No GPU detected, running on CPU.")


def turn_data_into_readable_for_ltsm(dataset_landmark_path="./smaller_dataset_landmarks"):
    X, y = [], []
    
    
    for word in os.listdir(dataset_landmark_path):
        
        word_folder = os.path.join(dataset_landmark_path, word)
        
        if os.path.isdir(word_folder):
            for npfile in os.listdir(word_folder):
                if npfile.endswith('.npy'):
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
    
    if os.path.exists("./smaller_dataset_landmarks/x.npy")  and os.path.exists("./smaller_dataset_landmarks/y_onehot.npy") :
        x = np.load("./smaller_dataset_landmarks/x.npy")
        y = np.load("./smaller_dataset_landmarks/y.npy")
        y_encoded = np.load("./smaller_dataset_landmarks/y_encoded.npy")
        y_onehot = np.load("./smaller_dataset_landmarks/y_onehot.npy")
        label_encoder.fit(y)

    else:
        x,y = turn_data_into_readable_for_ltsm()
        print(y)
        print(x)

        
        y_encoded = label_encoder.fit_transform(y)
        y_onehot = to_categorical(y_encoded)

        print(y.shape)
        print(y[0].shape)
        np.save("./smaller_dataset_landmarks/x.npy",x)
        np.save("./smaller_dataset_landmarks/y.npy",y)
        np.save("./smaller_dataset_landmarks/y_encoded.npy",y_encoded)
        np.save("./smaller_dataset_landmarks/y_onehot.npy",y_onehot)
        
    X_train, X_test, y_train, y_test = train_test_split(
            x, y_onehot, test_size=TRAIN_SPLIT, random_state=42, stratify=y_onehot
        )
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
    
    model = Sequential(
        [
            LSTM(
                256,input_shape=(SEQUENCE_LENGTH, FEATURE_DIM)
            ),
            Dropout(0.3),

            
            Dense(64, activation="relu"),
            Dense(len(set(y)), activation="softmax"),
        ]
    )
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    

    early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    
    
    history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    # callbacks=[early_stop]
    )
    
    # -------------------- EVALUATE --------------------
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    stopped_epoch = early_stop.stopped_epoch+1
    print(f"\n✅ Test Accuracy: {acc*100:.2f}% at epoch {stopped_epoch}")

    # -------------------- PLOT TRAINING HISTORY --------------------
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend(); plt.title('Accuracy')

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend(); plt.title('Loss')
    plt.savefig("training_curves.png")
    plt.close()

    # -------------------- CONFUSION MATRIX --------------------
    class_names = list(label_encoder.classes_)

    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    cm = confusion_matrix(y_true_labels, y_pred_labels)
    print("CONFU MATR")
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation='vertical', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # -------------------- SAVE MODEL --------------------
    model.save(f"./models/{EPOCHS}_{acc:.4f}_{TRAIN_SPLIT}_128lstm_signlang_lstm_model.h5")
    print(f"./models/{EPOCHS}_{acc:.4f}_{TRAIN_SPLIT}_128lstm_signlang_lstm_model.h5")
