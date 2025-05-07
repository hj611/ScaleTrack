# import sys
# import json

# test_random_file='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/common/datasets/agent/GUI-Odyssey/datasets--OpenGVLab--GUI-Odyssey/snapshots/2298cb628895d1c6248b8ead10c71429a76ce943/splits/random_split.json'
# anno_dir='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/common/datasets/agent/GUI-Odyssey/datasets--OpenGVLab--GUI-Odyssey/snapshots/2298cb628895d1c6248b8ead10c71429a76ce943/annotations/'
# out_dir='./data/annotations/'
# with open(test_random_file,'r',encoding='utf-8') as file:
#     data=json.load(file)


# for line in data['test']:
    
#     with open(anno_dir+line,'r',encoding='utf-8') as file:
#         item=json.load(file)
        
#     with open(out_dir+line,'w',encoding='utf-8') as file:
#         json.dump(item,file,indent=2)