input_file='android_control_ab.json'
output_file='android_control.json'
import json
from copy import deepcopy
with open(input_file,'r',encoding='utf-8') as file:
    data=json.load(file)


new_data=[]

for line in data:
    width=line['width']
    height=line['height']
    actions_re=[]
    for action in line['actions']:
        if action['action_type']=='click' or action['action_type']=='long_press':
            new_action=deepcopy(action)
            new_action['x']=action['x']*1.0/width
            new_action['y']=action['y']*1.0/height
            actions_re.append(new_action)
        else:
            actions_re.append(action)
    line['actions_re']=actions_re
    
    new_data.append(line)
    
with open(output_file,'w',encoding='utf-8') as file:
    json.dump(new_data,file,ensure_ascii=False,indent=2)
            
