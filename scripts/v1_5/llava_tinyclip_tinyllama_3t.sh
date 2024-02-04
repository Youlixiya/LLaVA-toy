deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./ckpts/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --version plain \
    --data_path data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder data/LLaVA-Pretrain/images \
    --vision_tower tinyclip \
    --mm_projector_type tinyclip \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-tinyclip-tinyllama-v1.0-1.1b-3t-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
# --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./ckpts/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --version llava_tinyllama \
    --data_path data/llava_v1_5_mix665k.json \
    --image_folder data \
    --vision_tower tinyclip \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-tinyclip-tinyllama-v1.0-1.1b-3t-pretrain/mm_projector.bin \
    --mm_projector_type tinyclip \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-tinyclip-tinyllama-v1.0-1.1b-3t \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/llava-tap-tinyllama-v1.0-1.1b-3t \
    --version llava_tinyllama \
    --data_path data/Flickr30k_train.json+data/coco_train.json+data/refcoco3_train.json \
    --image_folder data/flickr30k-images+data/coco/train2017+data/coco/train2014 \
    --vision_tower tinyclip \
    --mm_projector_type tinyclip \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-tinyclip-tinyllama-v1.0-1.1b-3t-rec \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

# deepspeed llava/train/train_mem.py \
#     --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path ./ckpts/TinyLlama-1.1B-Chat-v1.0 \
#     --version llava_tinyllama \
#     --data_path data/llava_instruct_150k.json+data/Flickr30k_train.json+data/coco_train.json+data/refcoco3_train.json \
#     --image_folder data/coco/train2017+data/flickr30k-images+data/coco/train2017+data/coco/train2014 \
#     --vision_tower sam \
#     --pretrain_mm_mlp_adapter ./checkpoints/llava-sam-tinyllama-v1.0-1.1b-pretrain/mm_projector.bin \
#     --mm_projector_type sam \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir ./checkpoints/llava-sam-tinyllama-v1.0-1.1b-lora \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-4 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb

# python scripts/merge_lora_weights.py --model-path ./checkpoints/llava-sam-tinyllama-v1.0-1.1b-lora \
#                                   --model-base ./ckpts/TinyLlama-1.1B-Chat-v1.0 \
#                                   --save-model-path ./checkpoints/llava-sam-tinyllama-v1.0-1.1b