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
# modelu = "../models/23-57_13-11-25_120_0.0858_0.3_GRU_signlang_model.keras" #2/4
# model="../models/18-32_13-11-25_45_43.61_0.3_GRU_signlang_model.keras" #2/4
# model="../models/StrongerGRU_18-52_13-11-25_120_0.0987_0.3_signlang_model.keras" #2/4
# modelu="../models/SimpleGRU__19-53_13-11-25_120_0.1942_0.2_GRU_signlang_model.keras" #0/4
modelu="../models/02-14_14-11-25_200_0.2913_0.15_GRU_signlang_model.keras" #3/4
modelu="../models/02-42_14-11-25_200_0.1767_0.25_GRU_signlang_model.keras"
modelu="../models/18___03-03_14-11-25_200_0.1165_0.25_GRU_signlang_model.keras" #3/4
modelu = "../models/5___13-50_14-11-25_200_0.1184_0.25_GRU_signlang_model.keras"
modelu="../models/6_SMALLGRU_14-02_14-11-25_200_0.2272_0.25_signlang_model.keras"
modelu="../modelscurr_best_pose_model.keras"
modelu="../models/7_LSTM_14-09_14-11-25_200_0.1146_0.25_signlang_model.keras"
modelu="../testing/model_comparison_results/BiGRU_Balanced_Regularization_best.keras"
# modelu="../testing/model_comparison_results/Lightweight_BiLSTM_Balanced_Regularization_best.keras"



modelus = ["../testing/best_bigru/BiGRU_Balanced_Regularization_latest.keras" , "../testing/model_comparison_results - Copy/Lightweight_BiLSTM_Balanced_Regularization_best.keras"]
# modelu="../testing/best_bigru/BiGRU_Balanced_Regularization_latest.keras"

for modelu in modelus:
    print(modelu)
    model = load_model(modelu)


    print(model.summary())
    inputs = ["../gte9_test/bed.npy","../gte9_test/dirty.npy","../gte9_test/doctor.npy","../gte9_test/ball2.npy","../gte9_test/ball.npy"]

    for ip in inputs:
        arr = np.load(ip)
        arr = arr.reshape(70, -1)                     # shape (70, 228)
        arr = np.expand_dims(arr, axis=0) 


        pred = model.predict(arr)
        pred_class = np.argmax(pred, axis=1)[0]
        word = label_encoder.inverse_transform([pred_class])[0]

        # print(f"Predicted sign: {word}")
        print(ip)
        top= np.argsort(pred[0])[-5:][::-1]
        for i in top:
            print(f"{label_encoder.inverse_transform([i])[0]}: {pred[0][i]*100:.2f}%")