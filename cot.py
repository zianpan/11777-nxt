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
from utls import *

class PromptGenerator:

    def __init__(self,):
        # self.model = model
        # self.model.eval()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)
        pass


        
    def base_oneshot_generator(self,question, choices, rationale, direct_ans, base_ans):
        prompt = f"Question: {question}\nChoices: A. {choices[0]} B. {choices[1]} C. {choices[2]} D. {choices[3]}\nRationale: {{{''.join(rationale)}}}\nAnswer: {{{toABCD[base_ans]}}}"
        return prompt

    def base_fewshot_generator(self,val_aokvqa,coco_id_filename, num_shots = 3):
        prompt_list = []
        sample_used = set()
        for i in range(num_shots):
            meta_data_one_sample = val_aokvqa[i]
        # meta_data_one_sample
            # TODO modify base_path
            base_path = "/home/ubuntu/data/coco/val2017/"
            img_id = meta_data_one_sample["image_id"]
            sample_used.add(img_id)
            img_file = coco_id_filename[img_id]
            img_path = base_path + img_file  
            base_ans = meta_data_one_sample["correct_choice_idx"]
            rationale =  meta_data_one_sample['rationales']
            direct_ans = meta_data_one_sample['direct_answers']
            toABCD = {0:'A', 1:'B', 2:'C', 3:'D'}

            question = meta_data_one_sample["question"]
            choices = meta_data_one_sample["choices"]
            prompt = self.base_oneshot_generator(question, choices, rationale, direct_ans, base_ans)
            prompt_list.append(prompt)
            
        return '\n'.join(prompt_list), sample_used


    def question_generator(self,question, choices):
        prompt = f"Question: {question}\nChoices: A. {choices[0]} B. {choices[1]} C. {choices[2]} D. {choices[3]}\nRationale: {{FILL IN Rationale}} Answer: {{FILL IN Answer}}"
        return prompt



    
        
if __name__ == "__main__":

    to_load = True
    if to_load:
        model = Llama32()
        to_load = False
    pg = PromptGenerator()
    val_aokvqa, coco_val_caption, coco_id_filename = prepare_dataset()

    prompt_template = """
You task is to select one of the four following options based on the image and the question. Specifically, you need to output {{A }} or {{B}} or {{C}}or {{D}} surrounded by curly braces as well as rationales of why you chose that option.
The rationales should also include in curly braces the answer to the question.

Here are some examples that you can follow:

Question: What is in the motorcyclist's mouth?
Choices: A. toothpick B. food C. popsicle stick D. cigarette
Rationale: {He's smoking while riding.The motorcyclist has a lit cigarette in his mouth while he rides on the street.The man is smoking.}
Answer: {D}

Question: Which number birthday is probably being celebrated?
Choices: A. one B. ten C. nine D. thirty
Rationale: {There is a birthday cake on the table with the number 30 written in icing.The cake says 30.The numerals three and zero are written on the cake, which indicates the person is 30 years of age as of the birthdate.}
Answer: {D}

Question: What best describes the pool of water?
Choices: A. frozen B. fresh C. dirty D. boiling
Rationale: {The pool is dark brown.It it brown and surrounded with mud.The pool is dirty.}
Answer: {C}

Now, it's your turn. Again, remember to put your answer in curly braces. Here is the question you need to answer.
"""
    cnt = 0
    for i in trange(4,1004):
        meta_data_one_sample = val_aokvqa[i]
        base_path = "/home/ubuntu/data/coco/val2017/"

        toABCD = {0:'A', 1:'B', 2:'C', 3:'D'}

        img_id = meta_data_one_sample["image_id"]
        img_file = coco_id_filename[img_id]
        img_path = base_path + img_file  
        base_ans = toABCD[meta_data_one_sample["correct_choice_idx"]]
        rationale =  meta_data_one_sample['rationales']
        direct_ans = meta_data_one_sample['direct_answers']


        question = meta_data_one_sample["question"]
        choices = meta_data_one_sample["choices"]
        mcToAsk = pg.question_generator(question, choices)
        local_prompt_template = prompt_template + mcToAsk
        output = model.predict_one(img_path,local_prompt_template,max_new_tokens=300)

        # isSame  = compare_ans(output,base_ans)
        cleaned_text = re.sub(r"<.*>", "", output)
        last_row = cleaned_text.split('\n')[-1]
        match = re.findall(r"\{(.*?)\}", last_row)
        if len(match) == 0:
            with open("logs/eval_cot.log","a") as f:
                f.write("No match found")
                f.write(output)
                f.write("##############################################################################################################")
            continue
        else:
            model_ans = match[-1]
            if model_ans not in ['A','B','C','D']:
                with open("logs/eval_cot.log","a") as f:
                    f.write("Invalid answer")
                    f.write(output)
                    f.write("##############################################################################################################")

                continue


        if model_ans == base_ans:
            isSame =  True
        else:
            isSame = False
        if isSame:
            # print("Correct")
            cnt += 1
        else:
            with open("logs/eval_cot.log","a") as f:
                s = """
                Image: {}
                Question: {}
                Choices: {}
                Base Answer: {}
                Predicted Answer: {}
                Ratinale: {}
                direct_ans: {}
                """.format(img_file, question, choices, base_ans, output, rationale, direct_ans)
                f.write(s)
                f.write("##############################################################################################################")

        if i % 100 == 0:
            print('current accuracy', cnt/(i-3))
    