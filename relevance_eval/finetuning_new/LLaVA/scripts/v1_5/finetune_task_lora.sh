#!/bin/bash

export TRANSFORMERS_OFFLINE=1 # 设置为1时，启用Transformers的离线模式
### deepspeed单机多卡设置显卡，不要使用export CUDA_VISIBLE_DEVICES=2,5，改成deepspeed --include localhost:2,5
# export CUDA_VISIBLE_DEVICES=2,5
include=localhost:0,1,2,3 # 设置显卡id

model_name_or_path=/home/qiangminc/codes/Shuri/relevance_eval/finetuning_new/LLaVA/checkpoints/llava-v1.5-7b # 模型名称
data_path=/home/qiangminc/codes/Shuri/relevance_eval/finetuning_new/finetuning_dataset/finetune_综合2.json # 训练的json
image_folder=/home/qiangminc/codes/Shuri/relevance_eval/finetuning_new/finetuning_dataset/image/综合 # 训练的图像数据
output_dir=/home/qiangminc/codes/Shuri/relevance_eval/finetuning_new/LLaVA/checkpoints/llava-v.15-7b-lora-综合2 # 输出目录

deepspeed --include $include llava/train/train_mem.py \
    --lora_enable True --lora_r 8 --lora_alpha 16 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $model_name_or_path  \
    --version v1 \
    --data_path $data_path \
    --image_folder $image_folder \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
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