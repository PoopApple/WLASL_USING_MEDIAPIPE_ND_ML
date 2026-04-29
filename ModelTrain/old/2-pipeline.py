import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from train import getmodel
import time
import os

from evaluate_model import eval_model, compare_models
from train_bilstm import get_bilstm_model


bigruascii = r"""
              ##                                  
 ######:      ##       :####:  ######:   ##    ## 
 #######      ##       ######  #######   ##    ## 
 ##   :##            :##:  .#  ##   :##  ##    ## 
 ##    ##   ####     ##:       ##    ##  ##    ## 
 ##   :##   ####     ##.       ##   :##  ##    ## 
 #######.     ##     ##        #######:  ##    ## 
 #######.     ##     ##  ####  ######    ##    ## 
 ##   :##     ##     ##. ####  ##   ##.  ##    ## 
 ##    ##     ##     ##:   ##  ##   ##   ##    ## 
 ##   :##     ##     :##:  ##  ##   :##  ##    ## 
 ########  ########   #######  ##    ##: :######: 
 ######    ########    :####.  ##    ###  :####:  
"""

bilstmascii = r"""
              ##                                            
 ######:      ##     ##         :####:   ########  ###  ### 
 #######      ##     ##        :######   ########  ###  ### 
 ##   :##            ##        ##:  :#      ##     ###::### 
 ##    ##   ####     ##        ##           ##     ###  ### 
 ##   :##   ####     ##        ###:         ##     ## ## ## 
 #######.     ##     ##        :#####:      ##     ##:##:## 
 #######.     ##     ##         .#####:     ##     ##.##.## 
 ##   :##     ##     ##            :###     ##     ## ## ## 
 ##    ##     ##     ##              ##     ##     ##    ## 
 ##   :##     ##     ##        #:.  :##     ##     ##    ## 
 ########  ########  ########  #######:     ##     ##    ## 
 ######    ########  ########  .#####:      ##     ##    ## 
"""


def printascii(lines):
    for line in lines.splitlines():
        print(line.center(terminal_width))


if __name__ == "__main__":
    terminal_width = os.get_terminal_size().columns

    dataset = np.load("./dataset1.0/dataset1-0.npz")

    X = dataset["features"]
    masks = dataset["masks"]
    y = dataset["labels"]

    num_classes = len(np.unique(y))

    # bigru
    model = getmodel(num_classes)
    for _ in range(10):
        time.sleep(0.02)
        print("\n")
    printascii(bigruascii)
    print()
    model.summary()
    for _ in range(10):
        time.sleep(0.02)
        print("\n")

    bigru_model_path = "./dataset1.0/asl_bigru_16-03-26__19-50_best.keras"
    map_path = "./dataset1.0/word_to_ind.json"
    dataset_path = "./dataset1.0/dataset1-0.npz"

    eval_model(dataset_path, map_path, bigru_model_path)
    for _ in range(10):
        time.sleep(0.02)
        print("\n")

    # bilstm
    model = get_bilstm_model(num_classes)
    for _ in range(10):
        time.sleep(0.02)
        print("\n")
    printascii(bilstmascii)
    print()
    model.summary()
    for _ in range(10):
        time.sleep(0.02)
        print("\n")

    bilstm_model_path = "./dataset1.0/BiLSTM/asl_bilstm_16-03-26__22-38_best.keras"

    eval_model(dataset_path, map_path, bilstm_model_path)
    for _ in range(10):
        time.sleep(0.02)
        print("\n")

    compare_models(
        dataset_path, map_path, bigru_model_path, bilstm_model_path, "BiGRU", "BiLSTM"
    )
