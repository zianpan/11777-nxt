from PIL import Image
import matplotlib.pyplot as plt
from MMObject.llama import Llama32
import torch
import numpy as np
import pandas as pd
import json
from PIL import Image
import matplotlib.pyplot as plt
import requests
from transformers import MllamaForConditionalGeneration, AutoProcessor
import os
import glob
from tqdm import trange
import re


def prepare_dataset(split):
    assert split in ["val", "test", "train"]

    path = "/home/ubuntu/data/aokvqa/"
    all_files = glob.glob(os.path.join(path, "*.json"))

    # read all files into a list
    # data = []
    # for filename in all_files:
    #     print(filename)
    #     with open(filename, 'r') as f:
    #         data.append(json.load(f))
    aokvqa_split = f"/home/ubuntu/data/aokvqa/aokvqa_v1p0_{split}.json"
    
    with open(aokvqa_split, 'r') as f:
        val_aokvqa = json.load(f)

    # val_aokvqa = data[2]

    path = "/home/ubuntu/data/coco/annotations"
    all_files = glob.glob(os.path.join(path, "*.json"))

    # with open(f"/home/ubuntu/data/coco/annotations/captions_{split}2017.json", 'r') as f:
    #     coco_val_caption = json.load(f)

    # with open(f"/home/ubuntu/data/coco/annotations/instances_{split}2017.json", 'r') as f:
    #     coco_val_instance = json.load(f)

    # coco_id_filename = {}

    # for d in coco_val_instance['images']:
    #     coco_id_filename[d['id']] = d['file_name']
    
    return val_aokvqa

def compare_ans(output, base_ans) -> bool:
    cleaned_text = re.sub(r"<.*>", "", output)
    last_row = cleaned_text.split('\n')[-1]
    match = re.findall(r"\{(.*?)\}", last_row)
    model_ans = match[-1]
    if model_ans == base_ans:
        return True
    else:
        return False

def show_image(img_path) -> None:
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def fill_image_path(image_id, split):
    img_file = str(image_id).zfill(12) + ".jpg"
    base_path = f"/home/ubuntu/data/coco/{split}2017/"
    img_path = base_path + img_file
    return img_path

def demonstrate_example(val_di):
#     {'split': 'val',
#  'image_id': 461751,
#  'question_id': '22jbM6gDxdaMaunuzgrsBB',
#  'question': "What is in the motorcyclist's mouth?",
#  'choices': ['toothpick', 'food', 'popsicle stick', 'cigarette'],
#  'correct_choice_idx': 3,
#  'direct_answers': ['cigarette',
#   'cigarette',
#   'cigarette',
#   'cigarette',
#   'cigarette',
#   'cigarette',
#   'cigarette',
#   'cigarette',
#   'cigarette',
#   'cigarette'],
#  'difficult_direct_answer': False,
#  'rationales': ["He's smoking while riding.",
#   'The motorcyclist has a lit cigarette in his mouth while he rides on the street.',
#   'The man is smoking.']}
    split = val_di['split']
    img_id = val_di['image_id']
    question = val_di['question']
    choices = val_di['choices']
    base_ans = val_di['correct_choice_idx']
    direct_ans = val_di['direct_answers']
    rationale = val_di['rationales']
    # img_file = coco_id_filename[img_id]

    img_file = str(img_id).zfill(12) + ".jpg"
    base_path = f"/home/ubuntu/data/coco/{split}2017/"
    img_path = base_path + img_file
    show_image(img_path)
    print(f"Question: {question}")
    print(f"Choices: 0. {choices[0]} 1. {choices[1]} 2. {choices[2]} 3. {choices[3]}")
    print(f"Correct Answer: {choices[base_ans]}")
    print(f"Direct Answers: {direct_ans}")
    print(f"Rationale: {rationale}")


