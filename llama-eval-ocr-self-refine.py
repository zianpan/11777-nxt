###### RECENT Implementation with LLAMA32 + OCR + CoT with fewshots
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
if to_load:
    model = Llama32()
    to_load = False

# %%
# path_dict = {"val":"/home/ubuntu/data/coco/val2017/",
#              "test":"/home/ubuntu/data/coco/test2017/",
#              "train":"/home/ubuntu/data/coco/train2017/"}

# %%


split = "val"
# print(f"Loading {split} dataset")
eval_num = 3

# eval_num = min(eval_num, 300)
# %%
# val_aokvqa = prepare_dataset(split=split)
ds_path = "new_dataset/difficult_direct_answer_70.json"
val_aokvqa = load_dataset_path(ds_path)

eval_num = min(eval_num, len(val_aokvqa))

# %%
# open this json file  /home/ubuntu/data/aokvqa/ocr_res_val.json
# with open("/home/ubuntu/data/aokvqa/ocr_res_val.json") as f:
#     ocr_res = json.load(f)

# %%
pg0123 = PromptGenerator0123(prompt_template = "")
prompt_template = pg0123.generate_template(val_aokvqa[:3])

# %%
prompt_template = pg0123.generate_template(val_aokvqa[:3])

prompt_template = """
You task is to select one of the four following options based on the image and the question. Specifically, you should focus on output rationales first.
The rationales should also include in curly braces the answer to the question.

Here are some examples that you can follow:
Question: What is in the motorcyclist's mouth?
Choices: 0. toothpick 1. food 2. popsicle stick 3. cigarette
Rationale: Looking at the picture, we can see a man with white shirt riding a motorcycle. Now as we find the motorcyclist in the question, we can see that he has a cigarette in his mouth.

Question: Which number birthday is probably being celebrated?
Choices: 0. one 1. ten 2. nine 3. thirty
Rationale: From the question, we know that someone is celebrating for the birthday. Looking at the picture, we can see two things on the table top. One is a grey bear which is likely to be a cake, the other is a cake with several candles on it. By looking at the purple cake, we can see it writes number thirty on it.

Question: What best describes the pool of water?
Choices: 0. frozen 1. fresh 2. dirty 3. boiling
Rationale: Looking at the picture, we can see a tree in the middle. Behind the tree, we can see several giraffes. On the bottom of the picture, we can see a pool of water. This refers to the pool mentioned in the question. The pool is dark brown and it is brown and surrounded with mud. So the pool is dirty.
"""

# %%
# print(prompt_template)


# %%
# ocr_tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True)
# ocr_model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=tokenizer.eos_token_id)
# ocr_model = ocr_model.eval().cuda()


# img_path = "/home/ubuntu/project/11777-nxt/002877-R1-008-2A.jpg"


# %%
with torch.no_grad():
    cnt = 0
    overall_confidence = []
    ind = 0
    error_logs = {"final_accuracy":"None", "error_logs":{}}
    for i in trange(eval_num):
        ind += 1
        torch.cuda.empty_cache()
        meta_data_one_sample = val_aokvqa[i]
        split = meta_data_one_sample['split']
        base_path = f"/home/ubuntu/data/coco/{split}2017/"
        img_id = meta_data_one_sample["image_id"]
        img_file = str(img_id).zfill(12) + ".jpg"
        # ocr_res_text = ocr_res[str(img_id)]
        img_path = base_path + img_file  
        base_ans = meta_data_one_sample["correct_choice_idx"]
        rationale =  meta_data_one_sample['rationales']
        direct_ans = meta_data_one_sample['direct_answers']
        question = meta_data_one_sample["question"]
        choices = meta_data_one_sample["choices"]
        mcToAsk = pg0123.generate_question(question, choices)
    #     OCR_prompt = f"""Here is the OCR result of the image. You can refer to this information to answer the question. Remember the OCR result is not always accurate.
    #    \n OCR result: <{ocr_res_text}>\n 
    #     Here is the question and choices you have."""

        # local_prompt_template = prompt_template + mcToAsk
        final_text = """You should output rationale step by step based on the image and the OCR result."""
        

        # output = model.predict_one(img_path,local_prompt_template,
        #                         extra_config = {"max_new_tokens":200, 
        #                             "output_scores":True,
        #                             "return_dict_in_generate":True})
        # text_ans = model.processor.decode(output.sequences[0])
        def extract_response(full_response):
            extracted_content = re.search(r"<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>(.*)", full_response, re.DOTALL).group(1).strip()
            return extracted_content
        
        # output = model.predict_one(img_path,mcToAsk + "\n Can you give me the descriptions of the main subject in the question and the four choices?",
        #                             extra_config = {"max_new_tokens":200})
        # choices_def = model.processor.decode(output[0])
        # del output
        # choices_def = extract_response(choices_def)

        # local_prompt_template = prompt_template + OCR_prompt + mcToAsk + "This is the description of the four choices: "+ choices_def + final_text
        local_prompt_template = prompt_template + "Here is the question and choices you have.\n" +  mcToAsk  + final_text

        text_ans = None

        for _ in range(2):
            local_prompt_template += "Now please output the rationale.\n Rationale:"
            output = model.predict_one(img_path,local_prompt_template,
                                    extra_config = {"max_new_tokens":200})
            text_ans = model.processor.decode(output[0])
            text_ans = re.sub(r'<\|.*\|>', '', text_ans)

            local_prompt_template = text_ans + "\n This is the previous response. Is it possible that other choices might be the correct? Provide feedback based on it. \n Feedback:"
            del output, text_ans
            output = model.predict_one(img_path,local_prompt_template,
                                    extra_config = {"max_new_tokens":200})
            text_ans = model.processor.decode(output[0])
            text_ans = re.sub(r'<\|.*\|>', '', text_ans)
            
            local_prompt_template = text_ans + "\n This is previous response. Can you give me a refined version of the rationale given feedback? \n Refined Rationale:"
            del output, text_ans
            output = model.predict_one(img_path,local_prompt_template,
                                    extra_config = {"max_new_tokens":200})
            text_ans = model.processor.decode(output[0])
            text_ans = re.sub(r'<\|.*\|>', '', text_ans)
        
        final_prompt = text_ans + f"\n Now, this is the question again \n {mcToAsk}\n Please output the answer using 0 or 1 or 2 or 3."

        output = model.predict_one(img_path,final_prompt,
                                    extra_config = {"max_new_tokens":200})
        text_ans = model.processor.decode(output[0])
        extracted_content = extract_response(text_ans)

        isCorrect, error_log = compare_ans(meta_data_one_sample, text_ans, last_num_string = 30)
        error_logs["error_logs"][i] = error_log
        if not isCorrect:
            error_logs["error_logs"][i] = error_log
        else:
            cnt += 1

        if ind % 5 == 0:
            print('current accuracy: ', cnt/ind)
        
        
        del output, text_ans, extracted_content,meta_data_one_sample
        
    error_logs['final_accuracy'] = cnt / eval_num


with open("logs/self-refine/self-refine-diffcult-70-t1.json", "w") as f:
    json.dump(error_logs, f)





            
            # self-refine


        

        
        # try:
        #     extracted_content1 = re.search(r"<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>(.*)", text_ans, re.DOTALL).group(1).strip()

        #     output = model.predict_one(img_path,local_prompt_template + extracted_content1 + "\n Now, please output the answer using 0 or 1 or 2 or 3.",
        #                             extra_config = {"max_new_tokens":200})
            
        #     text_ans2 = model.processor.decode(output[0])
        #     extracted_content2 = re.search(r"<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>(.*)", text_ans2, re.DOTALL).group(1).strip()

            # print(text_ans)


        # except:
        #     print("<ERROR0> Extract Ans Failed")
        #     print(text_ans)
        #     print(meta_data_one_sample)
        #     print("<END OF ERROR>")
        #     del output
        #     del meta_data_one_sample,text_ans
        #     continue



        

        # logits = output.scores

        # probabilities = [F.softmax(logit, dim=-1) for logit in logits]
        # local_confi = []
        # token_ids = output.sequences[0]
        # for i in range(-len(probabilities),-1):
        #     # print(i)
        #     prob_pos = token_ids[i]
        #     prob = probabilities[i]
        #     local_confi.append(probabilities[i].tolist()[0][prob_pos])

        # confi = np.mean(local_confi)
        # overall_confidence.append(confi)
        # del confi, probabilities, logits, token_ids,

        # model_ans = -1
        # for num in [0,1,2,3]:
        #     if str(num) in extracted_content:
        #         model_ans = num
        #         break

        # if model_ans == -1:
        #     print("<ERROR1> ANS NOT FOUND")
        #     print(text_ans)
        #     print(meta_data_one_sample)
        #     print("<END OF ERROR>")
        #     del output
        #     continue

        
        # if model_ans == int(base_ans):
        #     cnt += 1
        # else:
        #     print("<ERROR2> INCORRECT ANS")
        #     print(text_ans)
        #     print(meta_data_one_sample)
        #     print("TRUE ANS: ", base_ans)
        #     print("MODEL ANS: ", model_ans)
        #     print("<END OF ERROR>")


        # if i % 20 == 0:
        #     print('current accuracy: ', cnt/ind)
        
    
# print('final confidence', np.mean(overall_confidence))

# %%
# TEST

# %%
# img = Image.open('/home/ubuntu/data/coco/val2017/000000147415.jpg')
# plt.imshow(img)
# plt.axis('off')
# plt.show()

# %%
# ans = model.predict_one('/home/ubuntu/data/coco/val2017/000000147415.jpg',"What can you see?",
#                   extra_config = {"max_new_tokens":300, 
#                                   "output_scores":True,
# #                                   "return_dict_in_generate":True})

# # %%
# text_ans = model.processor.decode(ans.sequences[0])

# # %%
# text_ans

# # %%
# # # ans.scores
# # import torch.nn.functional as F
# # logits = ans.scores
# # probabilities = [F.softmax(logit, dim=-1) for logit in logits]
# # local_confi = []
# # token_ids = ans.sequences[0]
# # for i in range(-len(probabilities),-1):
# #     # print(i)
# #     prob_pos = token_ids[i]
# #     prob = probabilities[i]
# #     local_confi.append(probabilities[i].tolist()[0][prob_pos])


# # confi = np.mean(local_confi)


# # %%
# # extra_config = {"max_new_tokens":30, 
# #                                   "output_scores":True,
# #                                   "return_dict_in_generate":True}

# # %%


# # %%
# # image = Image.open('/home/ubuntu/data/coco/val2017/000000147415.jpg')
# # messages = [
# #     {"role": "user", "content": [
# #         {"type": "image"},
# #         {"type": "text", "text": "What can you see?"}
# #     ]}
# # ]
# # input_text = model.processor.apply_chat_template(messages, add_generation_prompt=True)
# # # print(input_text)
# # inputs = model.processor(
# #     image,
# #     input_text,
# #     add_special_tokens=False,
# #     return_tensors="pt"
# # )
# # for k, v in extra_config.items():
# #     inputs[k] = v

# # inputs.keys()

# # %%
# # text_ans = model.processor.decode(ans.sequences[0])
# # extracted_content = re.search(r"<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>", text_ans, re.DOTALL).group(1).strip()

# # extracted_content


