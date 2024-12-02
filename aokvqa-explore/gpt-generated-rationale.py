# %%
# !export OPENAI_API_KEY="sk-L1XFMvcUKBdPxoV6TH3QDQ"  
from openai import OpenAI
import base64
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import boto3
from MMObject.utls import *
from tqdm import tqdm

# %%

def upload_to_s3(image_path, bucket_name, object_name):
    s3_client = boto3.client(
        service_name='s3',
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )
    try:
        # Add ExtraArgs to set the ACL to public-read
        s3_client.upload_file(
            image_path, 
            bucket_name, 
            object_name, 
            ExtraArgs={'ACL': 'public-read'}
        )
        url = f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
        return url
    except Exception as e:
        print(f"Error uploading to S3: {e}")


def chat(image_url, prompt, api_key = "sk-L1XFMvcUKBdPxoV6TH3QDQ"):
    client = OpenAI(
      api_key = api_key,
      base_url="https://cmu.litellm.ai",
  )

    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
      {
        "role": "user",
        "content": [
          {"type": "text", "text": prompt},
          {
            "type": "image_url",
            "image_url": {
              "url": image_url,
            },
          },
        ],
      }
    ],
    max_tokens=300,
  )

    return response

# %%
train_aokvqa = prepare_dataset(split="train")

# %%
res = []


for i in trange(len(train_aokvqa)):
    sample = train_aokvqa[i]
    prompt = f"""
    Your task is to provide a good rationale for the multiple choices question asked by the user based on the image provided. The correct answer to the question will be provided. The template for the rationale is as follows:

    Question: What is in the motorcyclist's mouth?
    Choices: 0. toothpick 1. food 2. popsicle stick 3. cigarette
    Correct Answer: cigarette
    Rationale: Looking at the picture, we can see a man with white shirt riding a motorcycle. Now as we find the motorcyclist in the question, we can see that he has a cigarette in his mouth.

    Question: Which number birthday is probably being celebrated?
    Choices: 0. one 1. ten 2. nine 3. thirty
    Correct Answer: thirty
    Rationale: From the question, we know that someone is celebrating for the birthday. Looking at the picture, we can see two things on the table top. One is a grey bear which is likely to be a cake, the other is a cake with several candles on it. By looking at the purple cake, we can see it writes number thirty on it.

    Question: What best describes the pool of water?
    Choices: 0. frozen 1. fresh 2. dirty 3. boiling
    Correct Answer: dirty
    Rationale: Looking at the picture, we can see a tree in the middle. Behind the tree, we can see several giraffes. On the bottom of the picture, we can see a pool of water. This refers to the pool mentioned in the question. The pool is dark brown and it is brown and surrounded with mud. So the pool is dirty.

    Now, please provide a rationale for the following question based on the image provided:
    Question: {sample['question']}
    Choices: {[f"{ind}. {choice}" for ind, choice in enumerate(sample['choices'])]}
    Correct Answer: {sample['choices'][sample['correct_choice_idx']]}
    Rationale:
    """

    split = sample["split"]
    base_path = f"/home/ubuntu/data/coco/{split}2017/"
    img_id = sample["image_id"]
    img_file = f"{str(img_id).zfill(12)}.jpg"

    img_path = base_path + img_file

    img_url = upload_to_s3(img_path, 'vimaimage', 'train-aokvqa-img/img_file')
    response = chat(img_url, prompt)
    content  = response.choices[0].message.content
    sample['gpt_ratioanle'] = content
    res.append(sample)



# %%
with open(f"/home/ubuntu/data/aokvqa/aokvqa_{split}_gpt_ratioanle.json", "w") as f:
    json.dump(res, f)

# # %%
# with open(f"/home/ubuntu/data/aokvqa/aokvqa_{split}_gpt_ratioanle.json", "r") as f:
#     data = json.load(f)

# %%


# # %%
# print(response.choices[0].message.content)

# # %%
# show_image(img_path)

# # %%
# demonstrate_example(sample)

# %%
# split = "val"
# aokvqa_split = f"/home/ubuntu/data/aokvqa/aokvqa_v1p0_{split}.json"

# with open(aokvqa_split, 'r') as f:
#     val_aokvqa = json.load(f)

# # %%
# val_aokvqa

# # %%



