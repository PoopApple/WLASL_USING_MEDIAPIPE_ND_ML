import json
import os

video_folder = "videos/"
output_fold = "dataset/"
with open("word_with_instance.json","r") as jsonf:
    word_instance = json.load(jsonf)
    for word,datas in word_instance.items():
        vid_list = datas["video_name"]
        os.makedirs(output_fold+word)
        for video in vid_list:
            if os.path.exists(video_folder+word+"_"+video):
                
                os.rename(video_folder+word+"_"+video,output_fold+word+"/"+video)
                print("yes")
        else:
            print("no")
            
    
    
    