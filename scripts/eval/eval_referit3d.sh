#!/bin/bash

export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false

CKPT="./ckpt/$1"
ANWSER_FILE_1="results/nr3d/${1}.jsonl"

CUDA_VISIBLE_DEVICES=0 python3 llava/eval/model_referit3d.py \
    --model-path $CKPT \
    --lora-path $CKPT \
    --model-base llava_video_qwen \
    --video-folder /data \
    --embodiedscan-folder /data/embodiedscan \
    --n_gpu 1 \
    --question-file /data/nr3d_val_llava_style_viewtype.json \
    --conv-mode qwen_1_5 \
    --answer-file $ANWSER_FILE_1 \
    --overwrite_cfg true \
    --bbox_type pred \
    > "./eval_log/nr3d/${1}.log" 2>&1

python llava/eval/eval_referit3d.py --input-file $ANWSER_FILE >"./eval_log/nr3d/${1}_res.log" 2>&1

ANWSER_FILE_2="results/sr3d/${1}.jsonl"
CUDA_VISIBLE_DEVICES=0 python3 llava/eval/model_referit3d.py \
    --model-path $CKPT \
    --lora-path $CKPT \
    --model-base llava_video_qwen \
    --video-folder /data \
    --embodiedscan-folder /data/embodiedscan \
    --n_gpu 1 \
    --question-file /data/sr3d_val_llava_style_viewtype.json \
    --conv-mode qwen_1_5 \
    --answer-file $ANWSER_FILE_2 \
    --overwrite_cfg true \
    --bbox_type pred \
    > "./eval_log/sr3d/${1}.log" 2>&1

python llava/eval/eval_referit3d.py --input-file $ANWSER_FILE >"./eval_log/sr3d/${1}_res.log" 2>&1
