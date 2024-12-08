export AOKVQA_DIR=~/data/aokvqa
mkdir -p ${AOKVQA_DIR}

curl -fsSL https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz | tar xvz -C ${AOKVQA_DIR}

export COCO_DIR=~/data/coco
mkdir -p ${COCO_DIR}

for split in val; do
    wget "http://images.cocodataset.org/zips/${split}2017.zip"
    unzip "${split}2017.zip" -d ${COCO_DIR}; rm "${split}2017.zip"
done

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d ${COCO_DIR}; rm annotations_trainval2017.zip


conda upgrade conda
# conda create -n aokvqa python=3.11
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install huggingface_hub
pip install -e .
pip install -r requirement.txt
# pip install unsloth
# pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
# pip install --upgrade --no-cache-dir unsloth unsloth_zoo
