# %%
# from pyrsistent import m
from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
from MMObject.utls import *
from tqdm import trange

# %%
from MMObject.prompt_generator import *
pg0123  = PromptGenerator0123()

# %%
instruction = f"""
Your task is to provide a good rationale for the multiple choices question asked by the user based on the image provided. Remember, for the rationale, you should think by step and determine between each choice. The template for the rationale is as follows:

      Question: What is in the motorcyclist's mouth?
      Choices: 0. toothpick 1. food 2. popsicle stick 3. cigarette
      Rationale: 
      Looking at the picture, we can see a man with white shirt riding a motorcycle. Now as we find the motorcyclist in the question, we can see that he has a cigarette in his mouth. 
      Now, let’s analyze the choices:
      0. toothpick: Since the man is riding a bike, there is low chance that there is a toothpick in his mouth.
      1. food: Similary, the man is riding a bike and there is a stick-shape thing in his mouth which probably is not food.
      2. popsicle stick: The color of the object in the man's mouth is not a typical popsicle stick's color. Therefore, popsicle stick might not be the answer.
      3. cigarette: The shape of the object and the color of the object resembles a cigarette. And it's very likely that people will smoke cigarette when motorcyling. Threrfore, cigarette is probably correct.
      Based on the analysis, the most correct answer is cigarette.

      Question: Which number birthday is probably being celebrated?
      Choices: 0. one 1. ten 2. nine 3. thirty
      Rationale: 
      From the question, we know that someone is celebrating a birthday. Observing the picture, we notice two key items on the table: a grey bear-shaped object, likely a decorative cake, and a purple cake adorned with several candles. On closer inspection of the purple cake, the number "thirty" is clearly written on it.
      Now, let’s analyze the choices:
      0. one: There is no indication in the image or on the cake that suggests the number "one" is being celebrated.
      1. ten: Similarly, the number "ten" is not visible or suggested anywhere in the image.
      2. nine: The number "nine" is also not evident on any part of the cake or decorations.
      3. thirty: The number "thirty" is explicitly written on the purple cake, making this the correct answer.
      Based on the analysis, the most correct answer is thirty.

      Question: What best describes the pool of water?
      Choices: 0. frozen 1. fresh 2. dirty 3. boiling
      Rationale: 
      Observing the picture, we notice a tree in the middle, with several giraffes visible behind it. At the bottom of the image, there is a pool of water, which is the pool referred to in the question. The pool appears dark brown in color, muddy, and surrounded by dirt.
      Now, let’s analyze the choices:
      0. frozen: The pool cannot be frozen as the weather is clear and there are no signs of ice on the surface of the water.
      1. fresh: The water does not appear fresh because it is muddy and brown in color.
      2. dirty: The muddy appearance and dark brown color of the water indicate that the pool is dirty.
      3. boiling: The water is not boiling, as there are no signs of steam or bubbling, and the giraffes would not be able to drink boiling water.
      Based on the analysis, the most correct answer is dirty.

      Now, please provide a rationale for the following question based on the image provided:
"""

      # Question: {sample['question']}
      # Choices: {[f"{ind}. {choice}" for ind, choice in enumerate(sample['choices'])]}
      # Correct Answer: {sample['choices'][sample['correct_choice_idx']]}
      # Rationale:

def convert_to_conversation(sample, mode = "train"):
    if mode == "text":
        messages = [
    {"role": "user", 
     "content": [
        {"type": "image"},
        {"type": "text", "text": sample + "\n Now please output the answer using 0, 1, 2, 3"}
    ]}
]       
        return messages


    split = sample['split']
    base_path = f"/home/ubuntu/data/coco/{split}2017/"
    img_id = sample["image_id"]
    img_file = str(img_id).zfill(12) + ".jpg"
    # ocr_res_text = ocr_res[str(img_id)]
    img_path = base_path + img_file  
    base_ans = sample["correct_choice_idx"]
    rationale =  sample['rationales']
    direct_ans = sample['direct_answers']
    question = sample["question"]
    choices = sample["choices"]
    mcToAsk = pg0123.generate_question(question, choices)
    image = Image.open(img_path)
    if mode == "train":
        reference_rationale = rationale
   
        reference_rationale = sample['gpt_ratioanle'].strip()
        if reference_rationale.startswith("Rationale:"):
            reference_rationale = reference_rationale[10:].strip()
        
        prompt = instruction.strip()+"\n" + mcToAsk.strip() + "\n Rationale:"
        
    #     output = {"instruction": prompt, "output": reference_rationale, "image": [img_path]}
    #     return output
        conversation = [
            { "role": "user",
            "content" : [
                {"type" : "text",  "text"  : prompt},
                {"type" : "image", "image" : image} ]
            },
            { "role" : "assistant",
            "content" : [
                {"type" : "text",  "text"  : reference_rationale} ]
            },
        ]
        return { "messages" : conversation }

    elif mode == "val":
        prompt = instruction.strip()+"\n" + mcToAsk.strip() + "\n Rationale:"
        conversation = [
            { "role": "user",
            "content" : [
                {"type" : "image"},
                {"type" : "text",  "text"  : prompt},
                 ]
            }]
        return conversation, img_path
    
    



# %%
val_aokvqa = prepare_dataset("val")
# val_aokvqa = load_dataset_path("new_dataset/difficult_direct_answer_70.json")
val_aokvqa = load_dataset_path("new_dataset/based_model_hard_256.json")
# %%
messages = [convert_to_conversation(sample,mode="val") for sample in val_aokvqa]


# %%
# import json
# with open("logs/dict_logs/llama_direct_ans_full_t0.5_force_ans.json", "r") as f:
#     data = json.load(f)
#     m = []
# for k in data['error_logs'].keys():
#     m.append(val_aokvqa[int(k)])

# %%


# %%


# # %%
# messages = [convert_to_conversation(sample,mode="val") for sample in m]

# %%
# sample1 = {'split': 'val', 'image_id': 4795, 'question_id': '2N5sYXgyFqbDnuUhJFAWr5', 'question': 'What is the descriptive word for this surface?', 'choices': ['barren', 'crowded', 'minimalist', 'empty'], 'correct_choice_idx': 1, 'direct_answers': ['desk', 'desktop', 'glossy', 'computer', 'busy', 'crowded', 'table', 'smooth', 'desk', 'cluttered'], 'difficult_direct_answer': True, 'rationales': ['There are many visible objects in a small space without much unused space visible which is consistent with the definition of answer a.', 'There are many object including a cat on the desk space.', 'The area is really crowded and needs some room.']}
# converted_sample = convert_to_conversation(sample1,mode="val") 

# %%
if True:
    from unsloth import FastVisionModel
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = "zianpan01/llama32-V-CoT-Tuned", # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit = True, # Set to False for 16bit LoRA
    )
    FastVisionModel.for_inference(model) # Enable for inference!


# %%
def eval(messages,eval_num):
    error_logs = {"final_accu":"None", "error_logs":{}}
    eval_num = min(eval_num, len(messages))
    error_log = {}
    cnt = 0
    for i in trange(eval_num):
        message = messages[i][0]
        image = Image.open(messages[i][1])
        input_text = tokenizer.apply_chat_template(message, add_generation_prompt = True)
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens = False,
            return_tensors = "pt",
        ).to("cuda")
        res = []
        output1= model.generate(**inputs, max_new_tokens = 1000,
                        use_cache = True, temperature = 0.5, min_p = 0.2)
        rationale_text = re.search(r"(Now, please provide a rationale .*)", tokenizer.decode(output1[0], skip_special_tokens = True), re.DOTALL)
        extracted_rationale = rationale_text.group(1).strip() if rationale_text else "No rationale found."
        if "I'm sorry" in extracted_rationale:
            output1= model.generate(**inputs, max_new_tokens = 1000,
                        use_cache = True, temperature = 0.5, min_p = 0.2)
            rationale_text = re.search(r"(Now, please provide a rationale .*)", tokenizer.decode(output1[0], skip_special_tokens = True), re.DOTALL)
            extracted_rationale = rationale_text.group(1).strip() if rationale_text else "No rationale found."

        message2 = convert_to_conversation(extracted_rationale, "text")
        input_text = tokenizer.apply_chat_template(message2, add_generation_prompt = True)
        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens = False,
            return_tensors = "pt",
        ).to("cuda")
        res = []
        
        from transformers import TextStreamer
        # text_streamer = TextStreamer(tokenizer, skip_prompt = True)
        output1 = model.generate(**inputs, max_new_tokens = 300,
                        use_cache = True, temperature = 0.5, min_p = 0.2)
        langauge_output = tokenizer.decode(output1[0], skip_special_tokens = True)

        isCorrect, sample_error_log = compare_ans(val_aokvqa[i], langauge_output)
        if isCorrect:
            cnt += 1
        else:
            error_logs["error_logs"][i] = sample_error_log
        print(f"Accuracy: {cnt/(i+1)}")
    error_logs["final_accu"] = cnt/eval_num
    return error_logs

# %%
# error_logs = eval(messages,len(messages))
error_logs = eval(messages,len(messages))


# %%
with open("/home/ubuntu/project/11777-nxt/logs/hard_263/callama_hard_263_val.json", "w") as f:
    json.dump(error_logs, f)

# # %%
# false_sample = [(convert_to_conversation(x[1]['ERROR_MSG']['sample'], mode='val'), x[0]) for x in list(error_log.items())]

# # %%
# false_sample[:1]

# %%
# from transformers import TextStreamer
def eval_one(messages):

    message = messages[0]
    image = Image.open(messages[1])
    input_text = tokenizer.apply_chat_template(message, add_generation_prompt = True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")
    res = []
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    output1= model.generate(**inputs, streamer = text_streamer, max_new_tokens = 1000,
                    use_cache = True, temperature = 0.5, min_p = 0.2)

    rationale_text = re.search(r"(Now, please provide a rationale .*)", tokenizer.decode(output1[0], skip_special_tokens = True), re.DOTALL)
    extracted_rationale = rationale_text.group(1).strip() if rationale_text else "No rationale found."
    message2 = convert_to_conversation(extracted_rationale, "text")
    input_text = tokenizer.apply_chat_template(message2, add_generation_prompt = True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens = False,
        return_tensors = "pt",
    ).to("cuda")
    res = []
    
    # text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    output1 = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 1000,
                    use_cache = True, temperature = 0.5, min_p = 0.2)
    langauge_output = tokenizer.decode(output1[0], skip_special_tokens = True)

    isCorrect, sample_error_log = compare_ans(val_aokvqa[i], langauge_output)
    return isCorrect
    

# # %%
# for i in range(len(false_sample)):
#     ind = false_sample[i][1]
#     messages = false_sample[i][0]
    
#     eval_one(messages)
#     demonstrate_example(val_aokvqa[ind])


# %%


# %%
# demonstrate_example(val_aokvqa[5])

# %%
# for _,item in error_log.items():
#     print(item['ERROR_TYPE'])

# %%
# print(tokenizer.decode(_[0], skip_special_tokens = True))

# %%


# # %%
# rationale_text = re.search(r"(Now, please provide a rationale .*)", tokenizer.decode(_[0], skip_special_tokens = True), re.DOTALL)
# extracted_rationale = rationale_text.group(1).strip() if rationale_text else "No rationale found."

# message2 = convert_to_conversation(extracted_rationale, "text")

# input_text = tokenizer.apply_chat_template(message2, add_generation_prompt = True)
# inputs = tokenizer(
#     image,
#     input_text,
#     add_special_tokens = False,
#     return_tensors = "pt",
# ).to("cuda")
# res = []
# from transformers import TextStreamer
# # text_streamer = TextStreamer(tokenizer, skip_prompt = True)
# _ = model.generate(**inputs, max_new_tokens = 1000,
#                 use_cache = True, temperature = 0.5, min_p = 0.2)

#     # res.append(tokenizer.decode(_[0], skip_special_tokens = True)[-100:])


# # %%



