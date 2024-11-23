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
from llama import Llama32
from utls import *
from prompt_generator import *
import torch.nn.functional as F

# %%
if to_load:
    model = Llama32()
    to_load = False

# %%
path_dict = {"val":"/home/ubuntu/data/coco/val2017/",
             "test":"/home/ubuntu/data/coco/test2017/",
             "train":"/home/ubuntu/data/coco/train2017/"}

# %%

prompt_template = """
You task is to select one of the four following options based on the image and the question. Specifically, you need to output {{A }} or {{B}} or {{C}}or {{D}} surrounded by curly braces as well as rationales of why you chose that option.
The rationales should also include in curly braces the answer to the question.

Here are some examples that you can follow:
{}

Now, it's your turn. Again, remember to put your answer in curly braces. Here is the question you need to answer.
"""

# %%
val_aokvqa, coco_val_caption, coco_id_filename = prepare_dataset()

# %%
# open this json file  /home/ubuntu/data/aokvqa/ocr_res_val.json
with open("/home/ubuntu/data/aokvqa/ocr_res_val.json") as f:
    ocr_res = json.load(f)

# %%
ocr_res

# %%
pg0123 = PromptGenerator0123(prompt_template = prompt_template)
prompt_template = pg0123.generate_template(val_aokvqa[:3])

# %%
prompt_template = pg0123.generate_template(val_aokvqa[:3])

# %%
print(prompt_template)


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
    for i in trange(4,104):
        ind += 1
        torch.cuda.empty_cache()
        meta_data_one_sample = val_aokvqa[i]
        base_path = "/home/ubuntu/data/coco/val2017/"
        img_id = meta_data_one_sample["image_id"]
        img_file = coco_id_filename[img_id]
        ocr_res_text = ocr_res[str(img_id)]
        img_path = base_path + img_file  
        base_ans = meta_data_one_sample["correct_choice_idx"]
        rationale =  meta_data_one_sample['rationales']
        direct_ans = meta_data_one_sample['direct_answers']
        question = meta_data_one_sample["question"]
        choices = meta_data_one_sample["choices"]
        mcToAsk = pg0123.generate_question(question, choices)
        OCR_prompt = f"""Here is the OCR result of the image. You can refer to this information to answer the question. Remember the OCR result is not always accurate.
        {ocr_res_text}\n Now, here is the question you need to answer.
    """
        # local_prompt_template = prompt_template + mcToAsk
        local_prompt_template = prompt_template + OCR_prompt+mcToAsk

        

        # output = model.predict_one(img_path,local_prompt_template,
        #                         extra_config = {"max_new_tokens":200, 
        #                             "output_scores":True,
        #                             "return_dict_in_generate":True})
        # text_ans = model.processor.decode(output.sequences[0])

        output = model.predict_one(img_path,local_prompt_template,
                                extra_config = {"max_new_tokens":200})
        text_ans = model.processor.decode(output[0])

        
        try:
            extracted_content = re.search(r"<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>", text_ans, re.DOTALL).group(1).strip()
        except:
            print("<ERROR0> ANS NOT FOUND")
            print(text_ans)
            print(meta_data_one_sample)
            print("<END OF ERROR>")
            del output
            del meta_data_one_sample,text_ans
            continue
        extracted_content = text_ans[-20:]

        logits = output.scores
        probabilities = [F.softmax(logit, dim=-1) for logit in logits]
        local_confi = []
        token_ids = output.sequences[0]
        for i in range(-len(probabilities),-1):
            # print(i)
            prob_pos = token_ids[i]
            prob = probabilities[i]
            local_confi.append(probabilities[i].tolist()[0][prob_pos])

        confi = np.mean(local_confi)
        overall_confidence.append(confi)

        model_ans = -1
        for num in [0,1,2,3]:
            if str(num) in extracted_content:
                model_ans = num
                break

        if model_ans == -1:
            print("<ERROR1> ANS NOT FOUND")
            print(extracted_content)
            print(meta_data_one_sample)
            print("<END OF ERROR>")
            del output
            continue

        
        if model_ans == int(base_ans):
            cnt += 1
        else:
            print("<ERROR2> INCORRECT ANS")
            print(extracted_content)
            print(meta_data_one_sample)
            print("TRUE ANS: ", base_ans)
            print("MODEL ANS: ", model_ans)
            print("<END OF ERROR>")


        if i % 20 == 0:
            print('current accuracy: ', cnt/ind)
        
        
        del output, text_ans, extracted_content,meta_data_one_sample
        del confi, probabilities, logits, token_ids,
print('final accuracy', cnt/ind)  
print('final confidence', np.mean(overall_confidence))

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


