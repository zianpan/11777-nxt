from PIL import Image
import matplotlib.pyplot as plt
from llama import Llama32
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


def prepare_dataset():
    path = "/home/ubuntu/data/aokvqa/"
    all_files = glob.glob(os.path.join(path, "*.json"))

    # read all files into a list
    data = []
    for filename in all_files:
        with open(filename, 'r') as f:
            data.append(json.load(f))

    val_aokvqa = data[2]

    path = "/home/ubuntu/data/coco/annotations"
    all_files = glob.glob(os.path.join(path, "*.json"))

    with open("/home/ubuntu/data/coco/annotations/captions_val2017.json", 'r') as f:
        coco_val_caption = json.load(f)

    with open("/home/ubuntu/data/coco/annotations/instances_val2017.json", 'r') as f:
        coco_val_instance = json.load(f)

    coco_id_filename = {}

    for d in coco_val_instance['images']:
        coco_id_filename[d['id']] = d['file_name']
    
    return val_aokvqa, coco_val_caption, coco_id_filename

def compare_ans(output, base_ans) -> bool:
    cleaned_text = re.sub(r"<.*>", "", output)
    last_row = cleaned_text.split('\n')[-1]
    match = re.findall(r"\{(.*?)\}", last_row)
    model_ans = match[-1]
    if model_ans == base_ans:
        return True
    else:
        return False