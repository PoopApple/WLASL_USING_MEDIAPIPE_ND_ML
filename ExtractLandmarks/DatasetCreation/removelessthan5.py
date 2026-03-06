import os

import shutil

dspath = "./dataset"


removeddspath = "./dataset_removed"

with open("./stats/lessthan5list.txt","r") as f:
    while word_ := f.readline():
        
        word = word_.strip()
        
        word_path = os.path.join(dspath , word)
        rem_path = os.path.join(removeddspath,word)
        if os.path.exists(word_path):
            print(word_path)
            try:
                shutil.move(word_path,rem_path)
            except:
                break
        