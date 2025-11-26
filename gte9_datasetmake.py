import os

import shutil

dspath = "./dataset"

gte9_dataset = "./gte9_dataset"


os.makedirs(gte9_dataset,exist_ok=True)

with open("./stats/gte9_list.txt","r") as f:
    while word_ := f.readline():
        
        word = word_.strip()
        
        word_path = os.path.join(dspath , word)
        gte9_word_path = os.path.join(gte9_dataset,word)
        
        
        if os.path.exists(word_path):
            print(word_path)
            try:
                shutil.copytree(word_path,gte9_word_path)
                # shutil.copy(word_path,gte9_word_path)
            except Exception as e:
                print(e)
                break
        