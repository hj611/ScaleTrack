from typing import Dict
import sys
import json
swipe_func = {
    "name": "mobile.swipe",
    "description": "Swipe on the screen",
    "parameters": {
        "type": "object",
        "properties": {
            "from_coord": {
                "type": "array",
                "items": {"type": "number"},
                "description": "The starting coordinates of the swipe",
            },
            "to_coord": {
                "type": "array",
                "items": {"type": "number"},
                "description": "The ending coordinates of the swipe",
            },
        },
        "required": ["from_coord", "to_coord"],
    },
}

home_func = {"name": "mobile.home", "description": "Press the home button"}

back_func = {"name": "mobile.back", "description": "Press the back button"}

wait_func = {
    "name": "mobile.wait",
    "description": "wait for the change to happen",
    "parameters": {
        "type": "object",
        "properties": {
            "seconds": {
                "type": "number",
                "description": "The seconds to wait",
            },
        },
        "required": ["seconds"],
    },
}

long_press_func = {
    "name": "mobile.long_press",
    "description": "Long press on the screen",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {
                "type": "number",
                "description": "The x coordinate of the long press",
            },
            "y": {
                "type": "number",
                "description": "The y coordinate of the long press",
            },
        },
        "required": ["x", "y"],
    },
}

open_app_func = {
    "name": "mobile.open_app",
    "description": "Open an app on the device",
    "parameters": {
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": "The name of the app to open",
            },
        },
        "required": ["app_name"],
    },
}


TEMPLATES: Dict[str, str] = {}

def get_register_template(model_name):
    if model_name not in TEMPLATES:
        sys.exit(f"not model named {model_name}")
    return TEMPLATES[model_name]


### QWEN2VL
QWEN2VL_SYS="""You are a helpful assistant.
"""
QWEN2VL_USER="""
You are a GUI agent. You need to perform the next action to complete the task. \n\n## Output Format\n\nThought: ...\nAction: ...\n\n\n## Action Space \nclick(coordinate='(relative_x,relative_y)')\nlong_press(coordinate='(relative_x,relative_y)')\ninput_text(content='')\nscroll(direction='down or up or right or left')\nopen_app(app_name='')\nnavigate_back()\nnavigate_home()\nwait() # Submit the task regardless of whether it succeeds or fails.\n\n## Note\n- Use English in Thought part.\n\n- Use coordinates in relative terms (from 0 to 1)\n\n- Summarize your next action (with its target element) in one sentence in Thought part.\n
Please generate the next action according to the instruction, previous actions and screenshot image. 
## Instruction: {overall_goal} 
## Previous actions: {previous_actions}
"""
qwen2vl_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
TEMPLATES['QWEN2VL']=[QWEN2VL_SYS,QWEN2VL_USER,qwen2vl_template]

### AGUVIS
AGUVIS_SYS=f"""You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.

You have access to the following functions:
- {json.dumps(swipe_func)}
- {json.dumps(home_func)}
- {json.dumps(back_func)}
- {json.dumps(wait_func)}
- {json.dumps(long_press_func)}
- {json.dumps(open_app_func)}
"""
aguvis_template="{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant<|recipient|>all\nThought:{% endif %}"

AGUVIS_USER="""
Please generate the next move according to the ui screenshot, instruction and previous actions.

Instruction: {overall_goal} 

Previous actions: {previous_actions}
"""
TEMPLATES['AGUVIS']=[AGUVIS_SYS,AGUVIS_USER,aguvis_template]



### QWEN2VL_Llama & QWEN2VL_Llama_short
SHORT_SYS=f"""You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.
"""
LLAMA_USER="""
Please generate the next move according to the ui screenshot, instruction and previous actions. 

Instruction: {overall_goal} 

Previous actions: {previous_actions}
"""

TEMPLATES['QWEN2VL_Llama']=[AGUVIS_SYS,LLAMA_USER,qwen2vl_template]
TEMPLATES['QWEN2VL_Llama_short']=[SHORT_SYS,LLAMA_USER,qwen2vl_template]

### QWEN2VL_Llama_prompt_l1
LLAMA_Prompt_SYS_l1=''''
You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.\n\nYou have access to the following functions:\n- {\"name\": \"mobile.swipe\", \"description\": \"Swipe on the screen\", \"parameters\": {\"type\": \"object\", \"properties\": {\"from_coord\": {\"type\": \"array\", \"items\": {\"type\": \"number\"}, \"description\": \"The starting coordinates of the swipe\"}, \"to_coord\": {\"type\": \"array\", \"items\": {\"type\": \"number\"}, \"description\": \"The ending coordinates of the swipe\"}}, \"required\": [\"from_coord\", \"to_coord\"]}}\n- {\"name\": \"mobile.home\", \"description\": \"Press the home button\"}\n- {\"name\": \"mobile.back\", \"description\": \"Press the back button\"}\n- {\"name\": \"mobile.wait\", \"description\": \"wait for the change to happen\", \"parameters\": {\"type\": \"object\", \"properties\": {\"seconds\": {\"type\": \"number\", \"description\": \"The seconds to wait\"}}, \"required\": [\"seconds\"]}}\n- {\"name\": \"mobile.long_press\", \"description\": \"Long press on the screen\", \"parameters\": {\"type\": \"object\", \"properties\": {\"x\": {\"type\": \"number\", \"description\": \"The x coordinate of the long press\"}, \"y\": {\"type\": \"number\", \"description\": \"The y coordinate of the long press\"}}, \"required\": [\"x\", \"y\"]}}\n- {\"name\": \"mobile.open_app\", \"description\": \"Open an app on the device\", \"parameters\": {\"type\": \"object\", \"properties\": {\"app_name\": {\"type\": \"string\", \"description\": \"The name of the app to open\"}}, \"required\": [\"app_name\"]}}\n
'''
LLAMA_Prompt_USER_l1="""
Please generate the next move according to the ui screenshot, instruction and previous actions. 

Instruction: {overall_goal} 

Previous actions: {previous_actions}

Please generate the natural language description of the current action in 'Action: ' and the specific execution action in 'Operation: .
"""
TEMPLATES['QWEN2VL_Llama_prompt_l1']=[AGUVIS_SYS,LLAMA_Prompt_USER_l1,qwen2vl_template]




### sys2user
sys2user_sys='You are a helpful assistant.'
sys2user_user="""
You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.

You have access to the following functions:
- {{"name": "mobile.swipe", "description": "Swipe on the screen", "parameters": {{"type": "object", "properties": {{"from_coord": {{"type": "array", "items": {{"type": "number"}}, "description": "The starting coordinates of the swipe"}}, "to_coord": {{"type": "array", "items": {{"type": "number"}}, "description": "The ending coordinates of the swipe"}}}}, "required": ["from_coord", "to_coord"]}}}}
- {{"name": "mobile.home", "description": "Press the home button"}}
- {{"name": "mobile.back", "description": "Press the back button"}}
- {{"name": "mobile.wait", "description": "wait for the change to happen", "parameters": {{"type": "object", "properties": {{"seconds": {{"type": "number", "description": "The seconds to wait"}}}}, "required": ["seconds"]}}}}
- {{"name": "mobile.long_press", "description": "Long press on the screen", "parameters": {{"type": "object", "properties": {{"x": {{"type": "number", "description": "The x coordinate of the long press"}}, "y": {{"type": "number", "description": "The y coordinate of the long press"}}}}, "required": ["x", "y"]}}}}
- {{"name": "mobile.open_app", "description": "Open an app on the device", "parameters": {{"type": "object", "properties": {{"app_name": {{"type": "string", "description": "The name of the app to open"}}}}, "required": ["app_name"]}}}}

Please generate the next move according to the UI screenshot, instruction and previous actions. 

Instruction: {overall_goal}

Previous actions: {previous_actions}

Please generate the observation of the current screen in 'Observation: ', the thinking of the current action in 'Thought: ' and the natural language description of the current action in 'Action: '. Finally, the specific execution action is generated in 'Operation: 
"""
TEMPLATES['sys2user']=[sys2user_sys,sys2user_user,qwen2vl_template]


### QWEN2VL_Llama_Format
QWEN2VL_Llama_Format_SYS=AGUVIS_SYS
QWEN2VL_Llama_Format_USER="""
Please generate the next move according to the ui screenshot, instruction and previous actions.

Instruction: {overall_goal} 

Previous actions: {previous_actions}
Please generate the observation of the current screen in <observation></observation>, the thinking of the current action in <thinking></thinking>, and the natural language description of the current action in <action></action>. Finally, the specific execution action is generated in <tool_call></tool_call>
"""
TEMPLATES['QWEN2VL_Llama_Format']=[QWEN2VL_Llama_Format_SYS,QWEN2VL_Llama_Format_USER,qwen2vl_template]
