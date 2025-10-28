"""activate venv =>


source /home/aryan/opensource_lab_proj/venv/bin/activate
"""







import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoderp
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

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
                    X.append(arr)
                    y.append(word)
    return np.array(X, dtype=np.float32), np.array(y)


if __name__ == "__main__":
    x,y = turn_data_into_readable_for_ltsm()
    print(y)
    print(x)
    
    
    
