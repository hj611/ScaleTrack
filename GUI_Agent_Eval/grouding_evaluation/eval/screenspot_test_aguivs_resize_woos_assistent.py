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
from qwen_vl_utils import smart_resize


from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from process_utils import pred_2_point, extract_bbox

logging.basicConfig(level=logging.INFO)
torch.manual_seed(1234)
from PIL import Image, ImageDraw, ImageColor
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28

def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def draw_shapes(image: Image.Image, point: list, bbox: list, img_path, point_color=None, rect_color='red', rect_width=3):
    if isinstance(point_color, str):
        try:
            point_color = ImageColor.getrgb(point_color)
            point_color = point_color + (128,)
        except ValueError:
            point_color = (255, 0, 0, 128)
    else:
        point_color = (255, 0, 0, 128)

    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    img_width, img_height = image.size
    x = int(point[0] * img_width)
    y = int(point[1] * img_height)
    radius = min(img_width, img_height) * 0.01

    overlay_draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        fill=point_color
    )

    absolute_bbox = [int(bbox[0] * img_width), int(bbox[1] * img_height),
                     int(bbox[2] * img_width), int(bbox[3] * img_height)]

    for i in range(rect_width):
        overlay_draw.rectangle(
            [absolute_bbox[0] + i, absolute_bbox[1] + i, absolute_bbox[2] - i, absolute_bbox[3] - i],
            outline=rect_color
        )

    image = image.convert('RGBA')
    combined = Image.alpha_composite(image, overlay)
    result_image = combined.convert('RGB')

    out_folder = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/huangjing/code/mllm/UI/aguvis/eval/logs_qwen2vl/'
    output_path = out_folder + os.path.basename(img_path)
    result_image.save(output_path, format='PNG')

    return result_image

def load_pretrained_model(model_path):
     
    model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    return model, processor, tokenizer

def generate_response(model, processor, tokenizer, image, instruction, previous_actions=None, low_level_instruction=None, mode="self-plan", temperature=0.1, max_new_tokens=512):

    from template_high import get_register_template,until
    system_prompt,user_prompt,chat_template=get_register_template(mode)
    system_message={'role':"system","content":system_prompt}

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
            "text":    user_prompt.format(
                overall_goal=instruction,
                previous_actions=previous_actions,
                # low_level_instruction=instruction,
            )},
        ]
    }
    
    messages = [system_message, user_message]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, chat_template=chat_template,
    )
 
    logging.info("=====text: "+str(text))
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    
    inputs = inputs.to("cuda:0")
  
    
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
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--screenspot_imgs', type=str, required=True)
    parser.add_argument('--screenspot_test', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()

    model, processor, tokenizer = load_pretrained_model(args.model_path)
    logging.info("=====model: "+str(args.model_path))
    model.tie_weights()

    if args.task == "all":
        tasks = ["mobile", "desktop", "web"]
    else:
        tasks = [args.task]
    
    tasks_result = []
    result = []

    total_num_action = 0
    total_corr_action = 0

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
            total_num_action += 1
            filename = item["img_filename"]
            img_path = os.path.join(args.screenspot_imgs, filename)
            logging.info("====image: "+str(img_path))
            if not os.path.exists(img_path):
                print("img not found")
                continue
            image = load_image(img_path)
            dummy_image = image
            instruction = item["instruction"]
            logging.info("=====instruction: "+str(instruction))
            bbox = item["bbox"]
            init_img_size = image.size
            resized_height, resized_width  = smart_resize(dummy_image.height,
                dummy_image.width,
                factor=28,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS,)
            resize_image = dummy_image.resize((resized_width, resized_height))
            resize_size = resize_image.size

            width_scale = resize_size[0] / init_img_size[0]
            height_scale = resize_size[1] / init_img_size[1]
            resized_bbox = [
                bbox[0] * width_scale,
                bbox[1] * height_scale,
                (bbox[0] + bbox[2]) * width_scale,
                (bbox[1] + bbox[3]) * height_scale
            ]
            
            bbox = [resized_bbox[0] / resize_size[0], resized_bbox[1] / resize_size[1], resized_bbox[2] / resize_size[0], resized_bbox[3] / resize_size[1]]

            logging.info("=====resize bbox: "+str(bbox))

            response = generate_response(
                model,
                processor,
                tokenizer,
                resize_image,
                instruction,
                mode=args.mode,
                temperature=0.1,
                max_new_tokens=128,
            )
            logging.info("===response: "+response)

            try:
                click_point = pred_2_point(response)
                if(click_point[0]>1):
                    click_point = [x / 1000 for x in click_point]
                logging.info("===click_point: "+str(click_point))
                if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
                    corr_action += 1
                    total_corr_action += 1
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
                num_action -= 1
                total_num_action -= 1
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

    # Calculate and log MICRO scores
    if total_num_action > 0:
        micro_score = total_corr_action / total_num_action
    else:
        micro_score = 0.0

    logging.info("MICRO Scores: " + str(micro_score))
    logging.info(tasks_result)

if __name__ == "__main__":
    main()
