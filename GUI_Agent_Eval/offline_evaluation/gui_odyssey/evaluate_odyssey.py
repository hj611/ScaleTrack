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
    gt_x=gt_action['params']['x']
    gt_y=gt_action['params']['y']
    
    if math.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2) <=0.14:
        return True
    return False

def evaluate_odyssey_action(pred_action, gt_action):
    
    if gt_action['action_type'] == 'KEY_HOME' or gt_action['action_type'] == 'KEY_BACK' or gt_action['action_type'] == 'KEY_APPSELECT':
        if pred_action['action_type'] == gt_action['action_type']:
            return True, True
        return False, False
    
    if gt_action['action_type']=='COMPLETE' or gt_action['action_type']=='INCOMPLETE':
        if pred_action['action_type'] == gt_action['action_type'] or pred_action['action_type']=='TERMINATE':
            return True,True
        return False,False
    elif gt_action['action_type'] == 'TEXT':
        if pred_action['action_type'] == 'TEXT':
            return True, calculate_f1(pred_action['params']['content'], gt_action['params']['content'])
        else:
            return False, False
    elif gt_action['action_type'] == 'SCROLL':
        dx=gt_action['params']['to_coord'][0]-gt_action['params']['from_coord'][0]
        dy=gt_action['params']['to_coord'][1]-gt_action['params']['from_coord'][1]
        if abs(dx) > abs(dy):
            direction = 'left' if dx < 0 else 'right'
        else:
            direction = 'down' if dy > 0 else 'up'

        if pred_action['action_type'] == 'SCROLL':
            if direction==pred_action['params']['direction']:
                return True,True
            return True,False
        else:
            return False, False

    elif gt_action['action_type'] =='CLICK':
        if pred_action['action_type'] in ['LONG_PRESS', 'CLICK']:
            return True, check_click(pred_action,gt_action)
        else:
            return False, False    
    elif gt_action['action_type'] =='LONG_PRESS':
        if pred_action['action_type'] in ['LONG_PRESS']:
            return True, check_click(pred_action,gt_action)
        else:
            return False, False

    raise NotImplementedError