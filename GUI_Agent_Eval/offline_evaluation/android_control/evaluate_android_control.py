import sys
import numpy as np
import copy
import math
from collections import Counter

def check_text(text_pred, text_gt):
    text_pred = text_pred.lower().strip()
    text_gt = text_gt.lower().strip()
    return (text_pred in text_gt) or (text_gt in text_pred)


def calculate_f1(predicted_text, ground_truth_text, token_level='char'):
    if token_level == 'word':
        predicted_tokens = predicted_text.split()
        ground_truth_tokens = ground_truth_text.split()
    elif token_level == 'char':
        predicted_tokens = list(predicted_text)
        ground_truth_tokens = list(ground_truth_text)
    else:
        raise ValueError("token_level 参数必须是 'word' 或 'char'")

    predicted_counter = Counter(predicted_tokens)
    ground_truth_counter = Counter(ground_truth_tokens)

    tp = sum((predicted_counter & ground_truth_counter).values())  
    fp = sum((predicted_counter - ground_truth_counter).values())  
    fn = sum((ground_truth_counter - predicted_counter).values())  
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1 > 0.5



def check_click(pred_action,gt_action):
    pred_x=pred_action['params']['x']
    pred_y=pred_action['params']['y']
    gt_x=gt_action['x']
    gt_y=gt_action['y']
    
    if math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2) <=0.14:
        return True
    return False

def evaluate_android_control_action(pred_action, gt_action,candidate_bbox, width, height, resized_width, resized_height, pred_type ='abs_resized', gt_type='original_resized'):

    if gt_action['action_type'] == 'wait' or gt_action['action_type'] == 'navigate_home' or gt_action['action_type'] == 'navigate_back':
        if pred_action['action_type'] == gt_action['action_type']:
            return True, True
        return False, False
    elif gt_action['action_type'] == 'input_text':
        if pred_action['action_type'] == 'input_text':
            return True, calculate_f1(pred_action['params']['content'], gt_action['text'])
        else:
            return False, False
    elif gt_action['action_type'] == 'open_app':
        if pred_action['action_type'] == 'open_app':
            return True, calculate_f1(pred_action['params']['app_name'], gt_action['app_name'])
        else:
            return False, False

    elif gt_action['action_type'] == 'scroll':
        if pred_action['action_type'] == 'scroll':
            return True, pred_action['params']['direction'] == gt_action['direction']
        else:
            return False, False
    elif gt_action['action_type'] =='click':
        if pred_action['action_type'] in ['long_press','click']:
            return True, check_click(pred_action,gt_action)
        else:
            return False, False    
    elif gt_action['action_type'] =='long_press':
        if pred_action['action_type'] in ['long_press']:
            return True, check_click(pred_action,gt_action)
        else:
            return False, False
    raise NotImplementedError