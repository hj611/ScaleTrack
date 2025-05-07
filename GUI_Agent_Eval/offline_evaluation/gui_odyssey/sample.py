import json
import random
import sys


annotation_path='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/data/scaletrack/odyssey/annotations/'

with open('data/test_random.json','r',encoding='utf-8') as file:
    data=json.load(file)
     
limited_data=[]
for item in data['test']:
    with open(annotation_path+item,'r',encoding='utf-8') as file:
        sub=json.load(file)
        if sub["step_length"]>15:
            continue
        limited_data.append(sub)


random_num=500
if len(limited_data) >= random_num:
    output_data = random.sample(limited_data, random_num)
else:
    print("error")
    
output_file='./data/test.json'

with open(output_file,'w',encoding='utf-8') as file:
    json.dump(output_data,file,indent=2)
