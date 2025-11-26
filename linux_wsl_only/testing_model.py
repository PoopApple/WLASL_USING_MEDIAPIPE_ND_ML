import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import LabelEncoder



# model = load_model("200signlang_lstm_model.h5")
# model = load_model("300_0.07500000298023224_signlang_lstm_model.h5")


modelu = "./models/180_0.0500_0.3_128lstm_signlang_lstm_model.h5"





"""
normalise the landmarks even more:-

add more poses for hand since hand is not always visible
try changing visbility confidence for hands

then normalise based on shoulders and wingspan or whatever

using middle of shoulder as ceter or nose


also normalise height



"""

# assuming you have the same LabelEncoder used before
y = np.load("../gte9_landmarks/y.npy")
label_encoder = LabelEncoder()
label_encoder.fit(y)


modelu = "../models/18-34_09-11-25_100_0.1250_0.3_128lstm_signlang_lstm_model.h5"
modelu = "../models/18-47_09-11-25_120_0.1490_0.3_128lstm_signlang_lstm_model.keras"
modelu = "../models/18-59_09-11-25_120_0.2260_0.3_GRU_signlang_model.keras"
modelu = "../models/19-27_09-11-25_120_0.1683_0.3_GRU_signlang_model.keras"
modelu = "../models/23-57_13-11-25_120_0.0858_0.3_GRU_signlang_model.keras"
model="../models/18-32_13-11-25_45_43.61_0.3_GRU_signlang_model.keras"
model="../models/StrongerGRU_18-52_13-11-25_120_0.0987_0.3_signlang_model.keras"
model = load_model(modelu)

inputs = ["../gte9_test/ball.npy","../gte9_test/bed.npy","../gte9_test/cat.npy","../gte9_test/country.npy"]

for ip in inputs:
    arr = np.load(ip)
    arr = arr.reshape(70, -1)                     # shape (70, 228)
    arr = np.expand_dims(arr, axis=0) 


    pred = model.predict(arr)
    pred_class = np.argmax(pred, axis=1)[0]
    word = label_encoder.inverse_transform([pred_class])[0]

    print(f"Predicted sign: {word}")
    print(ip)
    top3 = np.argsort(pred[0])[-3:][::-1]
    for i in top3:
        print(f"{label_encoder.inverse_transform([i])[0]}: {pred[0][i]*100:.2f}%")