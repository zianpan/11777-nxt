# %%
to_load = True

# %%
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
from MMObject.llama import Llama32
from MMObject.utls import *
from MMObject.prompt_generator import *
import torch.nn.functional as F

# %%
if __name__ == "__main__":
    if to_load:
        model = Llama32()
        to_load = False



    prompt_template = """
    You task is to select one of the four following options based on the image and the question. Specifically, you need to output {{0}} or {{1}} or {{2}}or {{3}} surrounded by curly braces as well as rationales of why you chose that option.
    The rationales should also include in curly braces the answer to the question.

    Here are some examples that you can follow:
    {}

    Now, it's your turn. Again, remember to put your answer in curly braces. Here is the question you need to answer.
    """

    # %%
    val_aokvqa = prepare_dataset(split = "val")

    # %%
    pg0123 = PromptGenerator0123(prompt_template = prompt_template)
    prompt_template = pg0123.generate_template(val_aokvqa[:3])
    
    error_logs = {"final_accu":"None", "error_logs":{}}

    with torch.no_grad():
        cnt = 0
        overall_confidence = []
        ind = 1
        for i in trange(len(val_aokvqa)):
            meta_data_one_sample = val_aokvqa[i]
            split = meta_data_one_sample['split']
            base_path = f"/home/ubuntu/data/coco/{split}2017/"
            img_id = meta_data_one_sample["image_id"]
            img_file = str(img_id).zfill(12) + ".jpg"
            img_path = base_path + img_file  
            base_ans = meta_data_one_sample["correct_choice_idx"]
            rationale =  meta_data_one_sample['rationales']
            direct_ans = meta_data_one_sample['direct_answers']

            question = meta_data_one_sample["question"]
            choices = meta_data_one_sample["choices"]
            mcToAsk = pg0123.generate_question(question, choices)

            local_prompt_template = mcToAsk.strip() + "\n Now please output the answer using 0 or 1 or 2 or 3. \n Answer: "


            output = model.predict_one(img_path,local_prompt_template,
                                    extra_config = {"max_new_tokens":200, "temperature":0.5, "min_p" : 0.2})
            
            text_ans = model.processor.decode(output[0])

            isCorrect, error_log = compare_ans(meta_data_one_sample, text_ans, last_num_string = 30)
            if isCorrect:
                cnt += 1
            else:
                error_logs["error_logs"][i] = error_log
          
            torch.cuda.empty_cache()

        with open("/home/ubuntu/project/11777-nxt/logs/dict_logs/llama_direct_ans_full_t0.5.json", "w") as f:
            json.dump(error_logs, f)




    # %%
    # TEST

    # %%
    # img = Image.open('/home/ubuntu/data/coco/val2017/000000147415.jpg')
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()

    # # %%
    # ans = model.predict_one('/home/ubuntu/data/coco/val2017/000000147415.jpg',"What can you see?",
    #                   extra_config = {"max_new_tokens":30, 
    #                                   "output_scores":True,
    #                                   "return_dict_in_generate":True})

    # # %%
    # text_ans = model.processor.decode(ans.sequences[0])

    # %%


    # %%
    # # ans.scores
    # import torch.nn.functional as F
    # logits = ans.scores
    # probabilities = [F.softmax(logit, dim=-1) for logit in logits]
    # local_confi = []
    # token_ids = ans.sequences[0]
    # for i in range(-len(probabilities),-1):
    #     # print(i)
    #     prob_pos = token_ids[i]
    #     prob = probabilities[i]
    #     local_confi.append(probabilities[i].tolist()[0][prob_pos])


    # confi = np.mean(local_confi)


    # %%
    # extra_config = {"max_new_tokens":30, 
    #                                   "output_scores":True,
    #                                   "return_dict_in_generate":True}

    # %%


    # %%
    # image = Image.open('/home/ubuntu/data/coco/val2017/000000147415.jpg')
    # messages = [
    #     {"role": "user", "content": [
    #         {"type": "image"},
    #         {"type": "text", "text": "What can you see?"}
    #     ]}
    # ]
    # input_text = model.processor.apply_chat_template(messages, add_generation_prompt=True)
    # # print(input_text)
    # inputs = model.processor(
    #     image,
    #     input_text,
    #     add_special_tokens=False,
    #     return_tensors="pt"
    # )
    # for k, v in extra_config.items():
    #     inputs[k] = v

    # inputs.keys()

    # %%
    # text_ans = model.processor.decode(ans.sequences[0])
    # extracted_content = re.search(r"<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>", text_ans, re.DOTALL).group(1).strip()

    # extracted_content


