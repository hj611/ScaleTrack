import re

# is instruction English
def is_english_simple(text):
    try:
        text.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

# bbox -> point (str)
#将边界框（bbox）的坐标转换为中心点坐标的字符串表示
#bbox:左、上、右、下边界 dig:要保留的小数位数 ret:中心点坐标
def bbox_2_point(bbox, dig=2):
    # bbox [left, top, right, bottom]
    point = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    point = [f"{item:.2f}" for item in point]
    point_str = "({},{})".format(point[0], point[1])
    return point_str
def bbox_2_point_qwen2vl(bbox, dig=2):
    # bbox [left, top, right, bottom]
    point = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
    point = [f"{item:.2f}" for item in point]
    point_str = "({},{})".format(point[0], point[1])
    return point_str

# bbox -> bbox (str) 将边界框的坐标转换为字符串表示
def bbox_2_bbox(bbox, dig=2):
    bbox = [f"{item:.2f}" for item in bbox]
    bbox_str = "({},{},{},{})".format(bbox[0], bbox[1], bbox[2], bbox[3])
    return bbox_str

# point (str) -> point 从字符串中提取数值并转换为点坐标。如果提取到两个数，直接作为点坐标；如果提取到四个数，计算中心点坐标。
# def pred_2_point(s):
#     print("==============in  pred_2_point")
#     floats = re.findall(r'-?\d+\.?\d*', s)
#     floats = [float(num) for num in floats]
#     print("====floats: ",floats)
#     if len(floats) == 2:
#         click_point = floats
#     elif len(floats) == 4:
#         click_point = [(floats[0]+floats[2])/2, (floats[1]+floats[3])/2]
#     return click_point
def pred_2_point(s):
    # print("==============in pred_2_point")
    # 使用正则表达式匹配括号内的所有数字
    matches = re.findall(r'\(([^)]+)\)', s)
    click_point = None

    for match in matches:
        # 提取括号内的数字
        floats = re.findall(r'-?\d+\.?\d*', match)
        floats = [float(num) for num in floats]
        if len(floats) == 2:
            click_point = floats
            # break
        elif len(floats) == 4:
            click_point = [(floats[0] + floats[2]) / 2, (floats[1] + floats[3]) / 2]
            # break

    # print("====click_point: ", click_point)
    
    return click_point

# def pred_2_point_qwen25vl(s):
#     # 使用正则表达式匹配括号或方括号内的所有数字
#     matches = re.findall(r'[\[\(]([^\]\)]+)[\]\)]', s)
#     click_point = None

#     for match in matches:
#         # 提取括号或方括号内的数字
#         floats = re.findall(r'-?\d+\.?\d*', match)
#         floats = [float(num) for num in floats]
#         if len(floats) == 2:
#             click_point = floats
#             break
#         elif len(floats) == 4:
#             click_point = [(floats[0] + floats[2]) / 2, (floats[1] + floats[3]) / 2]
#             break

#     print("====click_point: ", click_point)
    
#     return click_point
def pred_2_point_qwen25vl(action):
    # action = json.loads(s.split('<tool_call>\n')[1].split('\n</tool_call>')[0])

    if 'click' in action['arguments']['action']:
        click_point = action['arguments']['coordinate']
    else:
        click_point = [0,0]
    return click_point

# bbox (qwen str) -> bbox
def extract_bbox(s):
    # Regular expression to find the content inside <box> and </box>
    pattern = r"<box>\((\d+,\d+)\),\((\d+,\d+)\)</box>"
    matches = re.findall(pattern, s)
    # Convert the tuples of strings into tuples of integers
    return [(int(x.split(',')[0]), int(x.split(',')[1])) for x in sum(matches, ())]

if __name__ == "__main__":
    # ret = bbox_2_point([1020, 630, 78, 84])
    # ret = extract_bbox("The element</ref><box>(5,8),(994,989)</box> corresponding to .")
    # ret = bbox_2_point_qwen2vl([223, 78, 601, 593])
    ret = pred_2_point_qwen25vl("[520, 349]")
    # (5,8),(994,989)
    print(ret)