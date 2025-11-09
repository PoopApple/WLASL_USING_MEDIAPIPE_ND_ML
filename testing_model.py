import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import LabelEncoder



# model = load_model("200signlang_lstm_model.h5")
# model = load_model("300_0.07500000298023224_signlang_lstm_model.h5")


modelu = "./models/180_0.0500_0.3_128lstm_signlang_lstm_model.h5"



modelu = "./models/100_0.0199_0.3_128lstm_signlang_lstm_model.h5"

model = load_model(modelu)


"""
normalise the landmarks even more:-

add more poses for hand since hand is not always visible
try changing visbility confidence for hands

then normalise based on shoulders and wingspan or whatever

using middle of shoulder as ceter or nose


also normalise height



"""

# assuming you have the same LabelEncoder used before
y = np.load("./smaller_dataset_landmarks/y.npy")
label_encoder = LabelEncoder()
label_encoder.fit(y)

inputs = ["./test_vid/addomen.npy","./test_vid/adult.npy","./test_vid/about.npy","./test_vid/00414.npy","./test_vid/00415.npy"]

for ip in inputs:
    arr = np.load(ip)
    arr = arr.reshape(70, -1)                     # shape (70, 228)
    arr = np.expand_dims(arr, axis=0) 


    pred = model.predict(arr)
    pred_class = np.argmax(pred, axis=1)[0]
    word = label_encoder.inverse_transform([pred_class])[0]

    print(f"Predicted sign: {word}")
    print(ip)
    top3 = np.argsort(pred[0])[-10:][::-1]
    for i in top3:
        print(f"{label_encoder.inverse_transform([i])[0]}: {pred[0][i]*100:.2f}%")