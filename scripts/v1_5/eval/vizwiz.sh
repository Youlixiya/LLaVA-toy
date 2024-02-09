#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path checkpoints/llava-openclip-convnext-qwen-v1.5-1.8b \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/llava-openclip-convnext-qwen-v1.5-1.8b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/llava-openclip-convnext-qwen-v1.5-1.8b \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/llava-openclip-convnext-qwen-v1.5-1.8b.json
