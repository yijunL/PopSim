import ujson as json
import time
from sklearn.model_selection import train_test_split
import random
from string import Template
from tqdm import tqdm


train_data = json.load(open("train_all.json",'r'))
img_data = json.load(open("UnderstandIMG_train_results_all.json",'r'))

# INSTRUCT_PROMPT = "You are a social media analyst given data about a user and one of their posts. \
# Analyze how each piece of data might affect the post's popularity score. A higher score means the post is trending. \
# Based on the data, predict the popularity score."
INSTRUCT_PROMPT = ""
INPUT_PROMPT = '''You are a social media analyst. Given user profiles and post information, please predict the popularity score.\n
User ${Uid} has uploaded ${photo_count} photos since ${photo_firstdatetaken}. \
They are currently operating in timezone ID ${timezone_id} and have a status of ${ispro}. The post ${Pid} (${Mediatype} type), titled ${Title}, \
was published on ${Postdate}. It belongs to the ${Category} category with a ${Concept} concept and is tagged with ${Subcategory}. \
It can be accessed publicly under ${Pathalias} and is tagged with [${Alltags}]. The post includes specific location details: \
longitude ${Longitude}, latitude ${Latitude}, and geoaccuracy ${Geoaccuracy}. The image description included in this post is: ${Img} The popularity score of this post is: 
'''


texts = []

contextualize_data = []
for sample in tqdm(train_data,total=len(train_data)):
    ispro = 'pro user' if sample["ispro"]==1.0 else 'free user'
    data = {
            "Uid": sample["Uid"], 
            "photo_count": sample["photo_count"], 
            "Postdate": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(sample["Postdate"]))),   
            "ispro": ispro,
            "timezone_id": sample["timezone_id"],
            "Pid": sample["Pid"],
            "Title": sample["Title"],
            "Category": sample["Category"],
            "Concept": sample["Concept"], 
            "Subcategory": sample["Subcategory"],
            "photo_firstdatetaken": sample["photo_firstdatetaken"],
            "Pathalias": sample["Pathalias"],
            "Alltags": sample["Alltags"],
            "Longitude": sample["Longitude"],
            "Latitude": sample["Latitude"],
            "Geoaccuracy": sample["Geoaccuracy"],
            "Mediatype": sample["Mediatype"],
            "Img": img_data[sample["img_filepaths"].strip()]
           }
    processed_prompt = Template(INPUT_PROMPT).substitute(data)
    contextualize_data.append(
    {
        "uid": str(sample["Uid"]), 
        "pid": str(sample["Pid"]),
        "time": int(sample["Postdate"]),
        "instruction": INSTRUCT_PROMPT,
        "input": processed_prompt,
        "output": str(sample["label"]),
    })
    texts.append(processed_prompt)

contextualize_data.sort(key=lambda x: x["time"])
processed_train_data = contextualize_data[:int(len(contextualize_data)*0.8)]
processed_train_data_mini = random.sample(processed_train_data, 30000)

processed_val_data, processed_test_data = contextualize_data[int(len(contextualize_data)*0.8):int(len(contextualize_data)*0.9)], contextualize_data[int(len(contextualize_data)*0.9):]
with open("SMP_train_mm_time.json", "w", encoding="utf-8") as f:
    json.dump(processed_train_data, f, ensure_ascii=False, indent=4)  # indent 用于美化格式
with open("SMP_train_mini_mm_time.json", "w", encoding="utf-8") as f:
    json.dump(processed_train_data_mini, f, ensure_ascii=False, indent=4)  # indent 用于美化格式
with open("SMP_val_mm_time.json", "w", encoding="utf-8") as f:
    json.dump(processed_val_data, f, ensure_ascii=False, indent=4)  # indent 用于美化格式
with open("SMP_test_mm_time.json", "w", encoding="utf-8") as f:
    json.dump(processed_test_data, f, ensure_ascii=False, indent=4)  # indent 用于美化格式