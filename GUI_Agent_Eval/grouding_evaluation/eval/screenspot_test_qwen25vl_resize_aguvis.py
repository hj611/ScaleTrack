import torch
 
import json
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm
from constants import agent_system_message, chat_template, grounding_system_message, until, user_instruction
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)
from qwen_vl_utils import smart_resize
import json
from PIL import Image
from agent_function_call_hj import MobileUse
import io
# from IPython.display import display
from process_utils import pred_2_point, extract_bbox,pred_2_point_qwen25vl

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import base64
from openai import OpenAI
import os.path as osp
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageColor
import numpy as np
# screenshot = "./assets/agent_function_call/mobile_en_example.png"
model_id="Qwen2.5-VL-72B-Instruct"
model_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/cache/QwenVL25/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/c6adf136ef57b4f6a6d7897c11bd3863bda237d7'

# The operation history can be orgnized by Step x: [action]; Step x+1: [action]...
user_query = 'The user query:  Open the file manager app and view the au_uu_SzH3yR2.mp3 file in MUSIC Folder\nTask progress (You have done the following operation on the current device): Step 1: {"name": "mobile_use", "arguments": {"action": "open", "text": "File Manager"}}; '

def show_image(img):
    
    plt.imshow(img)  # 显示图像
    plt.axis('off')  # 不显示坐标轴
    plt.show()  # 打开图像窗口

from PIL import Image, ImageDraw, ImageColor

def draw_shapes(image: Image.Image, point: list, bbox: list, point_color=None, rect_color='red', rect_width=3):
    # 绘制点
    if isinstance(point_color, str):
        try:
            point_color = ImageColor.getrgb(point_color)
            point_color = point_color + (128,)  # 添加透明度
        except ValueError:
            point_color = (255, 0, 0, 128)  # 默认红色，50% 透明度
    else:
        point_color = (255, 0, 0, 128)  # 默认红色，50% 透明度

    # 创建透明图层用于绘制
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # 将归一化点坐标转换为绝对坐标
    img_width, img_height = image.size
    x = int(point[0] * img_width)
    y = int(point[1] * img_height)
    radius = min(img_width, img_height) * 0.05

    overlay_draw.ellipse(
        [(x - radius, y - radius), (x + radius, y + radius)],
        fill=point_color  # 点的颜色
    )

    # 将归一化边界框坐标转换为绝对坐标
    absolute_bbox = [int(bbox[0] * img_width), int(bbox[1] * img_height),
                     int(bbox[2] * img_width), int(bbox[3] * img_height)]

    # 绘制矩形框，线条宽度通过多次绘制实现
    for i in range(rect_width):
        overlay_draw.rectangle(
            [absolute_bbox[0] + i, absolute_bbox[1] + i, absolute_bbox[2] - i, absolute_bbox[3] - i],
            outline=rect_color
        )

    # 合成图像，将透明图层与原图合并
    image = image.convert('RGBA')
    combined = Image.alpha_composite(image, overlay)
    result_image = combined.convert('RGB')

    # 保存图像
    output_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-basecv/huangjing/code/mllm/UI/aguvis/eval/1.png'
    result_image.save(output_path, format='PNG')

    return result_image


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def load_pretrained_model(model_path):
    # model = Qwen2VLForConditionalGeneration.from_pretrained(model_path,
    #                                                         torch_dtype=torch.bfloat16,
    #                                                         attn_implementation="flash_attention_2",
    #                                                         device_map="auto",
    #                                                         # device_map=_device_map,
    #                                                         )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,  device_map="auto",torch_dtype=torch.bfloat16,
)
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    return model, processor, tokenizer

logging.basicConfig(level=logging.INFO)
torch.manual_seed(1234)

processor = AutoProcessor.from_pretrained(model_path)

client = OpenAI(
    #If the environment variable is not configured, please replace the following line with the Dashscope API Key: api_key="sk-xxx".
    api_key="abc",
    base_url="http://10.166.152.38:8080/v1",
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--screenspot_imgs', type=str, required=True)
    parser.add_argument('--screenspot_test', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()

    if args.task == "all":
        tasks = ["mobile", "desktop", "web"]
    else:
        tasks = [args.task]
    
    tasks_result = []
    result = []

    model, processor, tokenizer = load_pretrained_model(args.model_path)
            # model.to(args.device)
    logging.info("===model: "+str(args.model_path))
    model.tie_weights()

    
    all_num=1271
    all_correct=0
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
            torch.cuda.empty_cache()
            num_action += 1
            filename = item["img_filename"]
            img_path = os.path.join(args.screenspot_imgs, filename)
            print("====image: ",img_path)
            instruction = item["instruction"]
            logging.info("=====instruction: "+str(instruction))
            if not os.path.exists(img_path):
                print("img not found")
                continue
            # image = load_image(img_path)
            image = Image.open(img_path)
            # base64_image = encode_image(img_path)
            width,height = image.size
            # print("====width: ",width)
            # print("====height: ",height)

            # screenshot = img_path
            dummy_image = image
            init_img_size = image.size
            resized_height, resized_width  = smart_resize(dummy_image.height,
                dummy_image.width,
                factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
                min_pixels=processor.image_processor.min_pixels,
                max_pixels=processor.image_processor.max_pixels,)
            resize_image = dummy_image.resize((resized_width, resized_height))
            # base64_image = encode_image(img_path,(resized_width,resized_height))
            base64_image = encode_image(img_path)
            print("====init_img_size: ",init_img_size)
            print("====resized_height: ",[resized_height, resized_width])
            mobile_use = MobileUse(
                cfg={"display_width_px": resized_width, "display_height_px": resized_height}
            )
             # mobile_use = MobileUse(
            #     cfg={"display_width_px": width, "display_height_px": height}
            # )

            bbox = item["bbox"]
            # bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            img_size = resize_image.size

            ##########copute resize box
            width_scale = img_size[0] / init_img_size[0]
            height_scale = img_size[1] / init_img_size[1]
            resized_bbox = [
                bbox[0] * width_scale,
                bbox[1] * height_scale,
                (bbox[0] + bbox[2]) * width_scale,
                (bbox[1] + bbox[3]) * height_scale
            ]
             
            bbox = [resized_bbox[0] / img_size[0], resized_bbox[1] / img_size[1], resized_bbox[2] / img_size[0], resized_bbox[3] / img_size[1]]

            logging.info("=====resize bbox: "+str(bbox))
            # Build messages
            system_message = {
                "role": "system",
                "content": "You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.",
            }
            messages=[
                system_message,
                {
                    "role": "user",
                    "content": [
                        {
                            "min_pixels": processor.image_processor.min_pixels,
                            "max_pixels": processor.image_processor.max_pixels,
                            # "type": "image_url",
                            # "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            "type": "image",
                            "image": resize_image,
                        },
                        # {"type": "text", "text": f"The user query:  {instruction}  (You have done the following operation on the current device):"},
                        {
                        "type": "text",
                        "text":    user_instruction.format(
                            overall_goal=instruction,
                            previous_actions="None",
                            # low_level_instruction=instruction,
                        )},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # print("=====text: ",text)
            logging.info("===text: "+text)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            logging.info("===response: "+response)

            try:
                # click_point = pred_2_point(coordinate)
                # print("===coordinate: ",coordinate)
                # click_point = pred_2_point_qwen25vl(str(coordinate))
                click_point = pred_2_point(response)

                if(click_point[0]>1):
                    click_point = [click_point[0]/resized_width,click_point[1]/resized_height]
                    # click_point = [x / 1000 for x in click_point]
                logging.info("===click_point: "+str(click_point))
                if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
                    corr_action += 1
                    all_correct+=1
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
            # input()  # 等待用户按下回车键

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
