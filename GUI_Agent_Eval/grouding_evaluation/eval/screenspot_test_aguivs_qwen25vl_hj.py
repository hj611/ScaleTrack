import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import json
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm
from qwen_vl_utils import process_vision_info
# from aguvis.constants import agent_system_message, chat_template, grounding_system_message, until, user_instruction
from constants import agent_system_message, chat_template, grounding_system_message, until, user_instruction


from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from process_utils import pred_2_point, extract_bbox

logging.basicConfig(level=logging.INFO)
torch.manual_seed(1234)

def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_pretrained_model(model_path):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    return model, processor, tokenizer

def generate_response(model, processor, tokenizer, image, instruction, previous_actions=None, low_level_instruction=None, mode="self-plan", temperature=0.7, max_new_tokens=1024):
    system_message = {
        "role": "system",
        "content": grounding_system_message if mode == "grounding" else agent_system_message,
    }
    # system_message = {
    #     "role": "system",
    #     "content": "You are a helpful assistent.",
    # }

    if isinstance(previous_actions, list):
        previous_actions = "\n".join(previous_actions)
    if not previous_actions:
        previous_actions = "None"
    user_message = {
        "role": "user",
        "content": [
        {
            "type": "image",
            "image": image,
        },
        {
            "type": "text",
            "text":instruction
        }
        # {
        #     "type": "text",
        #     "text":    "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task." + user_instruction.format(
        #         overall_goal=instruction,
        #         previous_actions=previous_actions,
        #         # low_level_instruction=instruction,
        #     )},
        #  {
        #     "type": "text",
        #     "text": 'Output only the coordinate of element box in your response. What element matches the following instruction or description: ' + instruction},
        ]
    }
    messages = [system_message, user_message]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, chat_template=chat_template
    )
    # print("====text: ",text)
    logging.info("====text: "+str(text))
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    # inputs = inputs.to(model.device)
    inputs = inputs.to("cuda:0")
    # inputs = inputs.to("cuda")
    
    cont = model.generate(**inputs, temperature=temperature, max_new_tokens=max_new_tokens)

    cont_toks = cont.tolist()[0][len(inputs.input_ids[0]) :]
    text_outputs = tokenizer.decode(cont_toks, skip_special_tokens=True).strip()
    for term in until:
        if len(term) > 0:
            text_outputs = text_outputs.split(term)[0]
    return text_outputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--screenspot_imgs', type=str, required=True)
    parser.add_argument('--screenspot_test', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()

    model, processor, tokenizer = load_pretrained_model(args.model_path)
    # model.to(args.device)
    model.tie_weights()
    logging.info("---------------model_path: "+str(args.model_path))

    if args.task == "all":
        tasks = ["mobile", "desktop", "web"]
    else:
        tasks = [args.task]
    
    tasks_result = []
    result = []

    for task in tasks:
        dataset = "screenspot_" + task + ".json"
        screenspot_data = json.load(open(os.path.join(args.screenspot_test, dataset), 'r'))
        print("Num of sample: " + str(len(screenspot_data)))
        num_action = 0
        corr_action = 0
        text_correct = []
        icon_correct = []
        num_wrong_format = 0
        for j, item in tqdm(enumerate(screenspot_data)):
            num_action += 1
            filename = item["img_filename"]
            img_path = os.path.join(args.screenspot_imgs, filename)
            print("====image: ",img_path)
            if not os.path.exists(img_path):
                print("img not found")
                continue
            image = load_image(img_path)
            instruction = item["instruction"]
            print("=====instruction: ",instruction)
            bbox = item["bbox"]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            img_size = image.size
            bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]
            # print("=====bbox: ",bbox)
            logging.info("====bbox: "+str(bbox))

            response = generate_response(
                model,
                processor,
                tokenizer,
                image,
                instruction,
                # mode="self-plan",
                mode="grounding",
                # temperature=0.7,
                temperature=0.1,
                max_new_tokens=1024
            )
            # print("===response: ",response)
            logging.info("====response: "+str(response))

            try:
                click_point = pred_2_point(response)
                if(click_point[0]>1):
                    click_point = [x / 1000 for x in click_point]
                # print("===click_point: ",click_point)
                logging.info("====click_point: "+str(click_point))
                if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
                    corr_action += 1
                    if item["data_type"] == 'text':
                        text_correct.append(1)
                    else:
                        icon_correct.append(1)
                    logging.info("match " + str(corr_action / num_action))
                else:
                    if item["data_type"] == 'text':
                        text_correct.append(0)
                    else:
                        icon_correct.append(0)
                    logging.info("unmatch " + str(corr_action / num_action))
                result.append({"img_path": img_path, "text": instruction, "bbox": bbox, "pred": click_point,
                               "type": item["data_type"], "source": item["data_source"]})
            except:
                num_wrong_format += 1
                if item["data_type"] == 'text':
                    text_correct.append(0)
                else:
                    icon_correct.append(0)
                logging.info("Step: " + str(j) + " wrong format")

        logging.info("Action Acc: " + str(corr_action / num_action))
        logging.info("Total num: " + str(num_action))
        logging.info("Wrong format num: " + str(num_wrong_format))
        logging.info("Text Acc: " + str(sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0))
        logging.info("Icon Acc: " + str(sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0))

        text_acc = sum(text_correct) / len(text_correct) if len(text_correct) != 0 else 0
        icon_acc = sum(icon_correct) / len(icon_correct) if len(icon_correct) != 0 else 0
        tasks_result.append([text_acc, icon_acc])

    logging.info(tasks_result)

if __name__ == "__main__":
    main()
