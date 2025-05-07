import random
import re
import json
import argparse
import sys
from qwen_vl_utils import smart_resize
import logging
from evaluate_odyssey import evaluate_odyssey_action
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

action_types = [
    'CLICK',
    'LONG_PRESS',
    'TEXT',
    'SCROLL',
    'KEY_HOME',
    'KEY_BACK',
    'KEY_APPSELECT',
    'COMPLETE',
    'INCOMPLETE'
]

aguvis_action_types=[
    'pyautogui.click',
    'pyautogui.write',
    'pyautogui.scroll',
    'pyautogui.hscroll',
    'swipe',
    'long_press',
    'home',
    'back',
    'recent',
    'terminate',
    '.click',
    '.scroll',
    'click',
    'scroll',
]


def extract_action_type_and_params(text, action_type):
    action_pattern = rf"\b{action_type}\s*\(([^)]*)\)"
    matches = re.findall(action_pattern, text)
    
    if not matches:
        return None

    param_dict = {}
    for params in matches:
        if not params:
            continue
    
        if action_type in ['click', 'long_press']:
            start_box_match = re.search(r"(\d+\.?\d*)\s*(?:,|\s+)\s*(\d+\.?\d*)", params)
            if start_box_match:
                param_dict['x'] = float(start_box_match.group(1))
                param_dict['y'] = float(start_box_match.group(2))
            continue

        if action_type == 'swipe':
            from_coord_match = re.search(r"from_coord\s*=\s*\[(.*?)\]", params)
            to_coord_match = re.search(r"to_coord\s*=\s*\[(.*?)\]", params)
            
            if from_coord_match:
                from_coords = from_coord_match.group(1).split(',')
                param_dict['from_coord']=[]
                param_dict['from_coord'].append(float(from_coords[0].strip()))
                param_dict['from_coord'].append(float(from_coords[1].strip()))
            
            if to_coord_match:
                param_dict['to_coord']=[]
                param_dict['to_coord'].append(float(from_coords[0].strip()))
                param_dict['to_coord'].append(float(from_coords[1].strip()))
            continue   
        param_pairs = re.findall(r"(\w+)\s*=\s*(?:['\"](.*?)(?<!\\)['\"]|([^'\",]+))", params)
        for key, value_str, value_num in param_pairs:
            if value_num:
                value = value_num
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            else:
                value = value_str
            param_dict[key] = value
    
    return {'action_type': action_type, 'params': param_dict}

def aguvis2norm(mid_result):
    result=dict()
    result['params']=dict()
    if 'click' in mid_result['action_type']:
        result['action_type']='CLICK'
        result['params']=mid_result['params']
        
    elif 'hscroll'in mid_result['action_type']:
        result['action_type']='SCROLL'
        if mid_result['params']['page']<0:
            result['params']['direction']='left'
        else:
            result['params']['direction']='right'        
        
    elif 'scroll' in mid_result['action_type']:
        result['action_type']='SCROLL'
        if mid_result['params']['page']<0:
            result['params']['direction']='down'
        else:
            result['params']['direction']='up'
  
    elif 'write'in mid_result['action_type']:
        result['action_type']='TEXT'
        if 'message' in mid_result['params']:
            result['params']['content']=mid_result['params']['message']
        else:
            result['params']['content']=''
    elif 'long_press'in mid_result['action_type']:
        result['action_type']='LONG_PRESS'
        result['params']=mid_result['params']
    elif 'swipe'in mid_result['action_type']:
        try:
            dx=mid_result['params']['to_coord'][0]-mid_result['params']['from_coord'][0]
            dy=mid_result['params']['to_coord'][1]-mid_result['params']['from_coord'][1]
            if abs(dx) > abs(dy):
                direction = 'left' if dx < 0 else 'right'
            else:
                direction = 'down' if dy < 0 else 'up'
            result['action_type']='SCROLL'
            result['params']['direction']=direction
        except KeyError:
            result['action_type']='SCROLL'
            result['params']['direction']=random.choice(['left','right','down','up'])
    else:
        if 'home' in mid_result['action_type']:
            result['action_type'] = 'KEY_HOME'
        elif 'back' in mid_result['action_type']:
            result['action_type'] = 'KEY_BACK'
        elif 'recent' in mid_result['action_type']:
            result['action_type'] = 'KEY_APPSELECT'
        elif 'terminate' in mid_result['action_type']:
            result['action_type'] = 'TERMINATE'
    return result
        

def get_metrics(data,model_type,log_path):
    
    Type_match_num = 0
    Extact_match_num = 0
    click_match_num = 0
    all_click_num = 0
    error_num = 0
    
    error_format=0
    for line in data:
        gt_action=line['gt']
        pred_action=None
        if model_type=='AGUVIS' or model_type=='QWEN2VL_Llama':
            for action_type in aguvis_action_types:
                mid_result = extract_action_type_and_params(line['pred'], action_type)
                result=None
                if mid_result!=None:
                    result=aguvis2norm(mid_result)
                if result:
                    pred_action=result
                    break
                
            if pred_action==None:
                error_format+=1
                print("--------error_format------",line['pred'])
                continue
        elif model_type=='QWEN2VL_Llama_Format':
            for action_type in aguvis_action_types:
                tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
                tool_call_matches = re.findall(tool_call_pattern, line['pred'])
                clean_text = tool_call_matches[-1] if len(tool_call_matches)>0 else line['pred']
                mid_result = extract_action_type_and_params(clean_text, action_type)
                result=None
                if mid_result:
                    result=aguvis2norm(mid_result)
                if result:
                    pred_action=result
                    break
                
            if pred_action==None:
                error_format+=1
                print("--------error_format------",line['pred'])
                continue
        elif model_type=='UI-TARS':
            pass
        else: 
            for action_type in action_types:
                result = extract_action_type_and_params(line['pred'], action_type)
                if result:
                    pred_action=result
                    break
                
            if pred_action==None:
                error_format+=1
                continue
        try:   
            type_match, extact_match = evaluate_odyssey_action(pred_action, gt_action)
            if type_match:
                Type_match_num += 1
            
            if extact_match:
                Extact_match_num += 1

            if extact_match and (pred_action['action_type'] == 'CLICK' or pred_action['action_type'] == 'LONG_PRESS'):
                click_match_num += 1
            
            if gt_action['action_type'] == 'CLICK' or gt_action['action_type']=='LONG_PRESS':
                all_click_num += 1
                
        except:
            import traceback
            traceback.print_exc()
            error_num += 1
            continue
        
    res = {
        'type_match_acc': Type_match_num/(len(data)-error_format)*100,
        'extact_match_acc': Extact_match_num/(len(data)-error_format)*100,
        'click_match_acc': click_match_num/all_click_num*100,
        'error_num': error_num,
        'error_format': error_format,
    }
    
    print(json.dumps(res, indent=' '))

    try:
        with open(log_path, "w") as outfile:
            json.dump(res,outfile,indent=2)
    except FileNotFoundError:
        print(f"Output file {log_path} not found.")
        exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--response_path', type=str, default='./debug/aguvis.json')
    parser.add_argument('--log_path', type=str, default='./logs/metrics_aguvis.log')
    parser.add_argument('--model_type', type=str, default='AGUVIS', help="model type")
    args = parser.parse_args()

    response_path=args.response_path
    model_type=args.model_type
    log_path=args.log_path
    
    try:
        with open(response_path, "r") as infile:
            data = json.load(infile)
    except FileNotFoundError:
        print(f"Input file {response_path} not found.")
        exit(1)
        
    get_metrics(data['result'],model_type,log_path)
    
