import json
with open("metadata.txt","a") as metaf:
    
    num_words = 0
    num_vids = 0
    
    
    with open("WLASL_V0.3.JSON","r") as f:
        fullfile = json.load(f)
        
        
        word_videoids = {}
        for word_dict in fullfile:
            word = word_dict["gloss"]
            # print(word)
            num_words+=1
            vidnamelist = []
            
            instance_list = word_dict["instances"]
            for instances in instance_list:
                vidnamelist.append(instances["video_id"]+".mp4")
            num_vids+=len(vidnamelist)
            word_videoids[word] = {"count":len(vidnamelist) , "video_name":vidnamelist}
            
            
        with open("output.json","w") as ff:
            json.dump(word_videoids,ff)
            
            
    print(f"num of words = {num_words}")
    print(f"num of vids = {num_vids}")
    print(f"avg = {num_vids/num_words}")
            
                