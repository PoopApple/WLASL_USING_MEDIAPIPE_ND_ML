import sys
import json
import numpy as np
import tensorflow as tf

MODEL_PATH = "./dataset1.0/asl_bigru_16-03-26__19-50_best.keras"
LABEL_MAP_PATH = "./dataset1.0/word_to_ind.json"

TARGET_FILE = "./test_actor.npz"


model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)
    index_to_word = {v: k for k, v in label_map.items()}

try:
    data = np.load(TARGET_FILE)

    X_sample = data["data"].astype(np.float32)
    mask_sample = data["mask"].astype(bool)

except Exception as e:
    print(f"Failed to load .npz file: {e}")
    exit()
X_batch = np.expand_dims(X_sample, axis=0)

mask_batch = np.expand_dims(mask_sample, axis=0)

predictions = model.predict([X_batch, mask_batch], verbose=0)

predicted_idx = np.argmax(predictions[0])
confidence = predictions[0][predicted_idx]
predicted_word = index_to_word[predicted_idx]

print(f"PREDICTION:  {predicted_word.upper()}")
print(f"CONFIDENCE:  {confidence * 100:.2f}%")

print("\nTop 5 Guesses:")
top_5_indices = np.argsort(predictions[0])[-5:][::-1]
for i in top_5_indices:
    print(f"- {index_to_word[i]}: {predictions[0][i] * 100:.2f}%")
