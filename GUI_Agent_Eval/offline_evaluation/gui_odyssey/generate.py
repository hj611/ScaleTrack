
import numpy as np
import json
from collections import defaultdict
import argparse
import re
from tqdm import tqdm
import math
import os
import sys
import logging
from transformers import AutoModel, AutoProcessor,Qwen2_5_VLForConditionalGeneration,Qwen2VLForConditionalGeneration,Qwen2VLProcessor
from template import get_register_template
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from qwen_vl_utils import process_vision_info
until = ["<|diff_marker|>"]
recipient_text = "<|im_start|>assistant<|recipient|>all\nThought: "


def load_model(model_path):
    # TODO AutoModel AutoProcessor
    model=Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
    # processor = AutoProcessor.from_pretrained(model_path)
    processor=Qwen2VLProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    return model, processor, tokenizer


def generate_response(messages,model,processor,tokenizer,chat_template,temperature=0.7,max_new_tokens=512):
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, chat_template=chat_template
    )
    logging.info("=====text: "+str(text))
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt",)
    
    device = next(model.parameters()).device 
    inputs = inputs.to(device)

    cont = model.generate(**inputs, temperature=temperature,max_new_tokens=max_new_tokens,do_sample=False)
    cont_toks = cont.tolist()[0][len(inputs.input_ids[0]) :]
    text_outputs = tokenizer.decode(cont_toks, skip_special_tokens=True).strip()
    return text_outputs

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='AGUVIS', help="model type")
    
    parser.add_argument('--model_path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/huangjing/code/mllm/UI/aguvis/model/models--xlangai--Aguvis-7B-720P/snapshots/6dd54127b5b84b9ee89172a5065ab6be576f0db9', help="transformer registerd model path")
    parser.add_argument("--input_file", type=str, default='./data/dev.json', help="Path to sample JSON file")
    parser.add_argument("--output_file", type=str,default='./debug/aguvis.json', help="Path to ans JSON file")
    
    parser.add_argument("--screenshot_dir", type=str,default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/hanwenkang/data/scaletrack/odyssey/image', help="Directory for screenshot images")
    args = parser.parse_args()
    
    output_file_path = args.output_file
    input_file_path = args.input_file
    screenshot_dir = args.screenshot_dir
    model,processor,tokenizer=load_model(args.model_path)
    
    try:
        with open(args.input_file, "r") as infile:
            data = json.load(infile)
    except FileNotFoundError:
        print(f"Input file {args.input_file} not found.")
        exit(1)

    system_pro,user_prompt,chat_template=get_register_template(args.model_type)
    system_prompt={'role':"system","content":system_pro}
    
    output={
        'model_type':args.model_type,
        'model_path':args.model_path,
        'output_file':args.output_file,
        'template':chat_template,
        'system_prompt':system_pro,
        'user_prompt':user_prompt
    }
    
    print(output)
    
    
    result=[]
    
    for item in tqdm(data):
        his=[]
        his_str='[]'

        for idx in range(len(item['steps'])):
            image=screenshot_dir+item['steps'][idx]['screenshot']
            user_message = {"role": "user","content": [
                    {"type": "image","image": image,"max_pixels":60*112*28*28,},
                    {"type": "text","text":user_prompt.format(
                        overall_goal=item['task_info']['task'],
                        previous_actions=his_str)},
                    ]}
            message=[system_prompt,user_message]
            ans=generate_response(message,model,processor,tokenizer,chat_template)
            
            # Odyssey action space
            action_type=item['steps'][idx]['action']
            params=dict()
            if item['steps'][idx]['ps']=='' and 'KEY' in item['steps'][idx]['info']: # key_home,key_recect,key_back
                action_type=item['steps'][idx]['info']
            elif 'COMPLETE' in action_type: # COMPLETE INCOMPLETE
                action_type=action_type
            elif 'TEXT' in action_type: # TEXT
                params['content']=item['steps'][idx]['info']
            elif action_type=='LONG_PRESS' or action_type =='CLICK':
                params['x']=item['steps'][idx]['info'][0][0]*1.0/1000
                params['y']=item['steps'][idx]['info'][0][1]*1.0/1000
            else: # SCROLL
                params['from_coord']=[item['steps'][idx]['info'][0][0]*1.0/1000,item['steps'][idx]['info'][0][1]*1.0/1000]
                params['to_coord']=[item['steps'][idx]['info'][1][0]*1.0/1000,item['steps'][idx]['info'][1][1]*1.0/1000]
                
            now_action={'action_type':action_type,'params':params}
            his.append(now_action)
            his_str=json.dumps(his)    
        
            result.append({"gt":now_action,"pred":ans})
    
    
    output['result']=result
    
    try:
        with open(output_file_path, "w") as outfile:
            json.dump(output,outfile,indent=2)
    except FileNotFoundError:
        print(f"Output file {output_file_path} not found.")
        exit(1)


