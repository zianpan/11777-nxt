import numpy as np
import pandas as pd
import json
from PIL import Image
import matplotlib.pyplot as plt
import requests
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
import os
import glob
from tqdm import trange


class Llama32:
    def __init__(self) -> None:
        model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
    
    def predict_one(self,img_path,prompt, extra_config):
        image = Image.open(img_path)
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        # print(input_text)
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)
        for k, v in extra_config.items():
            inputs[k] = v
        
            # output = self.model.generate(**inputs)
    
            # output = self.model.generate(**inputs)
        # self.model.eval()
        raw_output = self.model.generate(**inputs)
        
        del inputs, input_text, image,messages
        return raw_output


if __name__ == "__main__":

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


    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    print(model.device)


    total_num = 300
    correct_num = 0


    for i in trange(total_num):
        base_path = "/home/ubuntu/data/coco/val2017/"
        img_id = val_aokvqa[i]["image_id"]
        img_file = coco_id_filename[img_id]
        img_path = base_path + img_file  
        base_ans = val_aokvqa[i]["correct_choice_idx"] + 1
        rationale =  val_aokvqa[i]['rationales']
        direct_ans = val_aokvqa[i]['direct_answers']

        question = val_aokvqa[i]["question"]
        choices = val_aokvqa[i]["choices"]
        

        prompt = f"Question: {question}\nChoices: 1. {choices[0]} 2. {choices[1]} 3. {choices[2]} 4. {choices[3]}\n please select the correct answer by typing the number of the correct choice."

        image = Image.open(img_path)
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)

        output = model.generate(**inputs, max_new_tokens=30)
        ans = processor.decode(output[0])

        ans_num =  ans.split("\n")[-1][0]

        if ans_num not in ["1", "2", "3", "4"]:
            print('wrong format: ', ans)
            print("ground truth: ", base_ans)


            continue
        else:
            ans_num = int(ans_num)

        if ans_num == int(base_ans):
            correct_num += 1
        else:
            print("wrong!")
            print(base_path)
            print(img_file)
            print(img_path)
            print(question)
            print(choices)
            print(prompt)
            print(rationale)
            print(direct_ans)
            print('ground truth: ', base_ans)
            print("model wrong answer: ", ans_num)
            


        
        if i % 100 == 0:
            print(f"Correct number: {correct_num}")
            print(f"Total number: {i+1}")
            print(f"Accuracy: {correct_num/(i+1)}")
            print("=====================================")
    
    print(f"Final Correct number: {correct_num}")
    print(f"Total number: {total_num}")
    print(f"Final Accuracy: {correct_num/total_num}")
