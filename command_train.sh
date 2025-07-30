#!/bin/bash
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONWARNINGS="ignore::UserWarning"
# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"

CUDA_VISIBLE_DEVICES=0,1 nohup accelerate launch train.py \
 --pretrained_model_name_or_path="./checkpoints/SVD/stable-video-diffusion-img2vid-xt" \
 --output_dir="./checkpoints/Animation/train_20250730" \
 --data_root_path="./animation_data/fashion_sub7_fps16" \
 --rec_data_path="./animation_data/fashion_sub7_fps16/video_vec_path.txt" \
 --vec_data_path="./animation_data/fashion_sub7_fps16/video_vec_path.txt" \
 --validation_image_folder="./validation/fashion_sub7_fps16_00003/ground_truth" \
 --validation_control_folder="./validation/fashion_sub7_fps16_00003/poses" \
 --validation_image="./validation/fashion_sub7_fps16_00003/reference.png" \
 --num_workers=4 \
 --lr_warmup_steps=500 \
 --sample_n_frames=16 \
 --learning_rate=1e-5 \
 --lr_scheduler="cosine" \
 --per_gpu_batch_size=1 \
 --max_train_steps=6000 \
 --mixed_precision="fp16" \
 --gradient_accumulation_steps=1 \
 --checkpointing_steps=200 \
 --validation_steps=100 \
 --gradient_checkpointing \
 --checkpoints_total_limit=3 \
 --resume_from_checkpoint="latest" \
 > ./logs/command_train_20250730.log 2>&1 &
