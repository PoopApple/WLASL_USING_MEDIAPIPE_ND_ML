"""activate venv =>


source /home/aryan/opensource_lab_proj/venv/bin/activate
"""


"""
installed using this guide
https://www.tensorflow.org/install/pip#windows-wsl2_1
https://developer.nvidia.com/cuda-12-3-2-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
"""





import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split

# --- GPU CONFIG ---
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
    ""
    # x,y = turn_data_into_readable_for_ltsm()
    # print(y)
    # print(x)
    
    # print(y.shape)
    # print(y[0].shape)
    # np.save("./smaller_dataset_landmarks/x.npy",x)
    # np.save("./smaller_dataset_landmarks/y.npy",y)
    
    
    
