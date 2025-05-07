import time
import json
import argparse
from tqdm import tqdm
import sys
import logging
from transformers import AutoModel, AutoProcessor,Qwen2_5_VLForConditionalGeneration,Qwen2VLForConditionalGeneration,Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO AutoModel AutoProcessor
def load_model(model_path):
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
    
    # do_sample : greedy or not
    cont = model.generate(**inputs, temperature=temperature,max_new_tokens=max_new_tokens,do_sample=True)

    cont_toks = cont.tolist()[0][len(inputs.input_ids[0]) :]
    text_outputs = tokenizer.decode(cont_toks, skip_special_tokens=True).strip()
    return text_outputs

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='AGUVIS', help="model type")
    
    parser.add_argument('--model_path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/huangjing/code/mllm/UI/aguvis/model/models--xlangai--Aguvis-7B-720P/snapshots/6dd54127b5b84b9ee89172a5065ab6be576f0db9', help="transformer registerd model path")
    parser.add_argument("--input_file", type=str, default='./data/dev.json', help="Path to sample JSON file")
    parser.add_argument("--output_file", type=str,default='./debug/aguvis_ours_high.json', help="Path to ans JSON file")
    
    parser.add_argument("--screenshot_dir", type=str,default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/common/datasets/agent/AndroidControl/', help="Directory for screenshot images")
    parser.add_argument("--level", default='high', type=str, choices=['high', 'low'], help="Task level in AndroidControl")  # task level in AndroidControl
    args = parser.parse_args()
    
    output_file_path = args.output_file
    input_file_path = args.input_file
    screenshot_dir = args.screenshot_dir
    model,processor,tokenizer=load_model(args.model_path)

    if args.level=='high':
        from template_high import get_register_template
    elif args.level=='low':
        from template_low import get_register_template
    else:
        sys.exit(f"error setting of level {args.level}")
    
    try:
        with open(args.input_file, "r") as infile:
            data = json.load(infile)
    except FileNotFoundError:
        print(f"Input file {args.input_file} not found.")
        exit(1)

    system_pro,user_prompt,chat_template=get_register_template(args.model_type)
    system_prompt={'role':"system","content":system_pro}
    
    
    output={'model_type':args.model_type,
        'model_path':args.model_path,
        'output_file':args.output_file,
        'level':args.level,
        'template':chat_template,
        'system_prompt':system_pro,
        'user_prompt':user_prompt
    }
    
    print(output)
    time.sleep(10)
    
    result=[]
    if args.level=='high':
        for item in tqdm(data):
            his=[]
            his_str='[]'
            for idx in range(len(item['screenshots_path'])-1):
                image=screenshot_dir+item['screenshots_path'][idx][2:]
                user_message = {"role": "user","content": [
                    {"type": "image","image": image,"min_pixels":39*86*28*28,"max_pixels":52*112*28*28,},
                    {"type": "text","text":user_prompt.format(overall_goal=item['goal'],previous_actions=his_str)},]}
                
                message=[system_prompt,user_message]
                if item['actions_re'][idx]['action_type']=='click' or item['actions_re'][idx]['action_type']=='long_press':
                    item['actions_re'][idx]['x']=round(item['actions_re'][idx]['x'],3)
                    item['actions_re'][idx]['y']=round(item['actions_re'][idx]['y'],3)
                
                his.append(item['step_instructions'][idx])
                # his.append(item['actions_re'][idx])
                his_str=json.dumps(his)            
                ans=generate_response(message,model,processor,tokenizer,chat_template)
                result.append({"gt":item['actions_re'][idx],"pred":ans,'width':item['width'],'height':item['height'],
                            'candidate_bbox':item['candidate_bbox'][idx]})
            # sys.exit()
             
    elif args.level=='low': 
        for item in tqdm(data): 

            for idx in range(len(item['screenshots_path'])-1):
                image=screenshot_dir+item['screenshots_path'][idx][2:]
                user_message = {"role": "user","content": [
                    {"type": "image","image": image,"min_pixels":39*86*28*28,"max_pixels":52*112*28*28,},
                    {"type": "text","text":user_prompt.format(
                        overall_goal=item['goal'],
                        low_level_instruction=item['step_instructions'][idx],
                        previous_actions='[]')},]}
                
                message=[system_prompt,user_message]
                ans=generate_response(message,model,processor,tokenizer,chat_template)
                result.append({"gt":item['actions_re'][idx],"pred":ans,'width':item['width'],'height':item['height'],
                            'candidate_bbox':item['candidate_bbox'][idx]})
    else:
        print("no such setting")
        sys.exit()
        
    output['result']=result 
        
    try:
        with open(output_file_path, "w") as outfile:
            json.dump(output,outfile,indent=2)
    except FileNotFoundError:
        print(f"Output file {output_file_path} not found.")
        exit(1)


