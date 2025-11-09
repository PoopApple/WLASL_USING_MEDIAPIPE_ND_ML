import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import LabelEncoder



model = load_model("000signlang_lstm_model.h5")

import pickle

# assuming you have the same LabelEncoder used before
y = np.load("./smaller_dataset_landmarks/y.npy")
label_encoder = LabelEncoder()
label_encoder.fit(y)


arr = np.load("./test_vid/00415.npy")
arr = arr.reshape(70, -1)                     # shape (70, 228)
arr = np.expand_dims(arr, axis=0) 


pred = model.predict(arr)
pred_class = np.argmax(pred, axis=1)[0]
word = label_encoder.inverse_transform([pred_class])[0]

print(f"Predicted sign: {word}")
top3 = np.argsort(pred[0])[-10:][::-1]
for i in top3:
    print(f"{label_encoder.inverse_transform([i])[0]}: {pred[0][i]*100:.2f}%")