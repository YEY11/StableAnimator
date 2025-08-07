#!/bin/bash
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONWARNINGS="ignore::UserWarning"

# 1. [新增] 解决分布式训练通信超时的关键环境变量
#    这会强制 PyTorch 使用更稳定但不一定最快的通信方式，能有效避免初始化阶段的死锁问题。
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 因为你的验证数据是 vec (竖屏) 格式, 所以在这里指定正确的宽高
VAL_WIDTH=576
VAL_HEIGHT=1024

# - 解决了端口占用问题 (--main_process_port)
# - 修正了核心训练参数 (lr, steps)
# - 增加了检查点数量和频率，方便观察
CUDA_VISIBLE_DEVICES=6,7 nohup accelerate launch --main_process_port 29501 train.py \
 --pretrained_model_name_or_path="./checkpoints/SVD/stable-video-diffusion-img2vid-xt" \
 --output_dir="./checkpoints/Animation/train_20250807_fps8" \
 --data_root_path="./animation_data/fashion_sub7_fps8" \
 --rec_data_path="./animation_data/fashion_sub7_fps8/video_rec_path.txt" \
 --vec_data_path="./animation_data/fashion_sub7_fps8/video_vec_path.txt" \
 --validation_image_folder="./validation/fashion_sub7_fps8_00003/ground_truth" \
 --validation_control_folder="./validation/fashion_sub7_fps8_00003/poses" \
 --validation_image="./validation/fashion_sub7_fps8_00003/reference.png" \
 --validation_width=${VAL_WIDTH} \
 --validation_height=${VAL_HEIGHT} \
 --num_workers=4 \
 --lr_warmup_steps=500 \
 --sample_n_frames=16 \
 --learning_rate=1e-5 \
 --lr_scheduler="cosine" \
 --max_grad_norm=1.0 \
 --per_gpu_batch_size=1 \
 --max_train_steps=6000 \
 --mixed_precision="fp16" \
 --gradient_accumulation_steps=1 \
 --checkpointing_steps=200 \
 --validation_steps=100 \
 --gradient_checkpointing \
 --checkpoints_total_limit=3 \
 --resume_from_checkpoint="latest" \
 >> ./logs/command_train_20250807_fps8.log 2>&1 &
