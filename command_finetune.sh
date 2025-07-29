#!/bin/bash

# 禁用albumentations版本检查
export ALBUMENTATIONS_DISABLE_ONLINE_VERSION_CHECK=1

# 禁用albumentations更新检查
export NO_ALBUMENTATIONS_UPDATE=1

# 使用单卡训练
CUDA_VISIBLE_DEVICES=1 python train.py \
  --pretrained_model_name_or_path="./checkpoints/SVD/stable-video-diffusion-img2vid-xt" \
  --finetune_mode=True \
  --posenet_model_finetune_path="./checkpoints/Animation/pose_net.pth" \
  --face_encoder_finetune_path="./checkpoints/Animation/face_encoder.pth" \
  --unet_model_finetune_path="./checkpoints/Animation/unet.pth" \
  --output_dir="./checkpoints/Animation/finetune" \
  --data_root_path="./animation_data" \
  --rec_data_path="./animation_data/video_vec_path.txt" \
  --vec_data_path="./animation_data/video_vec_path.txt" \
  --validation_image_folder="./validation/ground_truth" \
  --validation_control_folder="./validation/poses" \
  --validation_image="./validation/reference.png" \
  --num_workers=4 \
  --lr_warmup_steps=100 \
  --sample_n_frames=16 \
  --learning_rate=3e-6 \
  --per_gpu_batch_size=1 \
  --max_train_steps=500 \
  --num_train_epochs=50 \
  --mixed_precision="fp16" \
  --gradient_accumulation_steps=4 \
  --checkpointing_steps=200 \
  --validation_steps=100 \
  --gradient_checkpointing \
  --checkpoints_total_limit=10 \
  --resume_from_checkpoint="latest"
