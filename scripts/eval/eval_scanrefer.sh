#!/bin/bash

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT="./ckpt/$1"
ANWSER_FILE="results/scanrefer/$1.jsonl"


CUDA_VISIBLE_DEVICES=0 python3 llava/eval/model_scanrefer.py \
    --model-path $CKPT \
    --lora-path $CKPT \
    --model-base llava_video_qwen \
    --video-folder /data \
    --embodiedscan-folder /data/embodiedscan \
    --n_gpu 1 \
    --question-file data/scanrefer_val_llava_style.json \
    --conv-mode qwen_1_5 \
    --answer-file $ANWSER_FILE \
    --overwrite_cfg true \
    --bbox-type pred \
    > "./eval_log/scanrefer/$1.log" 2>&1

python llava/eval/eval_scanrefer.py --input-file $ANWSER_FILE > "./eval_log/scanrefer/${1}_res.log" 2>&1
