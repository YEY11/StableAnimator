# -*- coding: utf-8 -*-

"""
Author: YEY
Date: 2025-07-30

这是一个用于训练 StableAnimator 模型的 Python 脚本。
该脚本使用 Hugging Face 的 Accelerate库进行分布式训练，并集成了 Diffusers 库中的组件。
主要流程包括：
1. 解析命令行参数，配置训练环境。
2. 加载预训练的 Stable Video Diffusion (SVD) 模型组件，包括 VAE, Image Encoder, UNet 等。
3. 初始化自定义模块，如 PoseNet, FusionFaceId 等，并设置特定的注意力处理器 (Attention Processor)。
4. 配置优化器，实现对模型部分参数的选择性训练。
5. 实现健壮的数据加载逻辑，能够处理不同分辨率(rec/vec)的数据集，并能优雅地处理空数据集的情况。
6. 执行主训练循环，包括前向传播、损失计算、反向传播和模型更新。
7. 在训练过程中定期保存检查点 (checkpoint) 和执行验证 (validation)，生成样本视频以监控训练效果。
"""

import argparse
import random
import logging
import math
import os

import cv2
import shutil
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import PIL
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from diffusers.models.attention_processor import XFormersAttnProcessor

from animation.dataset.animation_dataset import LargeScaleAnimationVideos
from animation.modules.attention_processor import AnimationAttnProcessor
from animation.modules.attention_processor_normalized import AnimationIDAttnNormalizedProcessor
from animation.modules.face_model import FaceModel
from animation.modules.id_encoder import FusionFaceId
from animation.modules.pose_net import PoseNet
from animation.modules.unet import UNetSpatioTemporalConditionModel

from animation.pipelines.validation_pipeline_animation import ValidationAnimationPipeline
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange

import datetime
import diffusers
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available
import warnings
import torch.nn as nn
from diffusers.utils.torch_utils import randn_tensor
from torch.utils.data import ConcatDataset # 引入 ConcatDataset 用于合并数据集

# 确保安装了最低版本的 diffusers
check_min_version("0.24.0.dev0")

# 初始化日志记录器
logger = get_logger(__name__, log_level="INFO")

# 工具函数：验证并转换图像为PIL格式，用于后续处理
def validate_and_convert_image(image, target_size=(256, 256)):
    if image is None:
        print("Encountered a None image")
        return None

    if isinstance(image, torch.Tensor):
        if image.ndim == 3 and image.shape[0] in [1, 3]:
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(image)
        else:
            print(f"Invalid image tensor shape: {image.shape}")
            return None
    elif isinstance(image, Image.Image):
        image = image.resize(target_size)
    else:
        print("Image is not a PIL Image or a PyTorch tensor")
        return None
    
    return image

# 工具函数：将多张图片拼接成网格图
def create_image_grid(images, rows, cols, target_size=(256, 256)):
    valid_images = [validate_and_convert_image(img, target_size) for img in images]
    valid_images = [img for img in valid_images if img is not None]

    if not valid_images:
        print("No valid images to create a grid")
        return None

    w, h = target_size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, image in enumerate(valid_images):
        grid.paste(image, box=((i % cols) * w, (i // cols) * h))

    return grid

# 工具函数：保存拼接后的验证帧图像
def save_combined_frames(batch_output, validation_images, validation_control_images,output_folder):
    flattened_batch_output = [img for sublist in batch_output for img in sublist]
    combined_frames = validation_images + validation_control_images + flattened_batch_output
    num_images = len(combined_frames)
    cols = 3
    rows = (num_images + cols - 1) // cols
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"combined_frames_{timestamp}.png"
    grid = create_image_grid(combined_frames, rows, cols)
    output_folder = os.path.join(output_folder, "validation_images")
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"combined_frames_{timestamp}.png"
    output_loc = os.path.join(output_folder, filename)
    
    if grid is not None:
        grid.save(output_loc)
    else:
        print("Failed to create image grid")

# 工具函数：从文件夹加载图片序列
def load_images_from_folder(folder):
    images = []
    files = os.listdir(folder)
    png_files = [f for f in files if f.endswith('.png')]
    # 根据文件名中的数字排序，例如 'frame_0.png', 'frame_1.png'
    png_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    for filename in png_files:
        img = Image.open(os.path.join(folder, filename)).convert('RGB')
        images.append(img)
    return images

# --- 以下是 K-diffusion 中的噪声采样策略相关函数 ---
def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    if group < 0 or group >= groups:
        raise ValueError(f"group must be in [0, {groups})")
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n

def rand_cosine_interpolated(shape, image_d, noise_d_low, noise_d_high, sigma_data=1., min_value=1e-3, max_value=1e3, device='cpu', dtype=torch.float32):
    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(
            t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_uniform(
        shape, group=0, groups=1, dtype=dtype, device=device
    )
    logsnr = logsnr_schedule_cosine_interpolated(
        u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data

def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()

min_value = 0.002
max_value = 700
image_d = 64
noise_d_low = 32
noise_d_high = 64
sigma_data = 0.5

# --- 图像处理相关工具函数 ---
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]
    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1
    input = _gaussian_blur2d(input, ks, sigmas)
    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners)
    return output

def _compute_padding(kernel_size):
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]
    out_padding = 2 * len(kernel_size) * [0]
    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front
        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear
    return out_padding

def _filter2d(input, kernel):
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)
    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)
    height, width = tmp_kernel.shape[-2:]
    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    out = output.view(b, c, h, w)
    return out

def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])
    batch_size = sigma.shape[0]
    x = (torch.arange(window_size, device=sigma.device,
         dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)
    if window_size % 2 == 0:
        x = x + 0.5
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))
    return gauss / gauss.sum(-1, keepdim=True)

def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)
    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])
    return out

# 工具函数：将帧序列导出为视频
def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)

# 工具函数：将帧序列导出为GIF
def export_to_gif(frames, output_gif_path, fps):
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]
    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=1000 // fps,
                       loop=0)

# 工具函数：将图像张量通过VAE编码为潜在表示
def tensor_to_vae_latent(t, vae, scale=True):
    t = t.to(vae.dtype)
    if len(t.shape) == 5: # (B, F, C, H, W)
        video_length = t.shape[1]
        t = rearrange(t, "b f c h w -> (b f) c h w")
        latents = vae.encode(t).latent_dist.sample()
        latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    elif len(t.shape) == 4: # (B, C, H, W)
        latents = vae.encode(t).latent_dist.sample()
    if scale:
        latents = latents * vae.config.scaling_factor
    return latents

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练 StableAnimator 模型的脚本")
    
    # --- 模型与路径参数 ---
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="预训练SVD模型路径或Hugging Face模型ID")
    parser.add_argument("--revision", type=str, default=None, help="预训练模型的特定版本(revision)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="模型检查点和输出的保存目录")
    parser.add_argument("--data_root_path", type=str, default=None, help="训练数据的根目录")
    parser.add_argument("--rec_data_path", type=str, default=None, help="指向rec(矩形, 512x512)数据列表的txt文件路径")
    parser.add_argument("--vec_data_path", type=str, default=None, help="指向vec(竖屏, 576x1024)数据列表的txt文件路径")

    # --- 训练过程参数 ---
    parser.add_argument("--num_frames", type=int, default=14, help="视频帧数")
    parser.add_argument("--sample_n_frames", type=int, default=16, help="训练时采样的帧数")
    parser.add_argument("--per_gpu_batch_size", type=int, default=1, help="每个GPU的批处理大小")
    parser.add_argument("--num_train_epochs", type=int, default=100, help="训练的总轮数")
    parser.add_argument("--max_train_steps", type=int, default=None, help="最大训练步数，如果提供，将覆盖num_train_epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="是否使用梯度检查点以节省显存")

    # --- 优化器与学习率调度器参数 ---
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--scale_lr", action="store_true", default=False, help="是否根据GPU数量/梯度累积/批量大小缩放学习率")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help="学习率调度器类型")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="学习率预热步数")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam优化器的beta1参数")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam优化器的beta2参数")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Adam优化器的权重衰减")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Adam优化器的epsilon值")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="最大梯度范数（用于梯度裁剪）")
    parser.add_argument("--use_8bit_adam", action="store_true", help="是否使用8位Adam优化器")

    # --- 验证参数 ---
    parser.add_argument("--validation_steps", type=int, default=500, help="每隔X步运行一次验证")
    parser.add_argument("--num_validation_images", type=int, default=1, help="验证时生成的视频数量")
    parser.add_argument("--validation_image_folder", type=str, default=None, help="验证用的真实视频帧文件夹路径")
    parser.add_argument("--validation_image", type=str, default=None, help="验证用的参考图像路径")
    parser.add_argument("--validation_control_folder", type=str, default=None, help="验证用的姿态控制序列文件夹路径")
    parser.add_argument("--validation_width", type=int, default=512, help="验证流程的宽度")
    parser.add_argument("--validation_height", type=int, default=512, help="验证流程的高度")
    
    # --- 性能与精度参数 ---
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"], help="混合精度训练类型")
    parser.add_argument("--allow_tf32", action="store_true", help="是否在Ampere GPU上允许TF32以加速训练")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="是否启用xformers以优化内存使用")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载使用的工作进程数")

    # --- 检查点与恢复参数 ---
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="每隔X步保存一个检查点")
    parser.add_argument("--checkpoints_total_limit", type=int, default=1, help="最多保存的检查点数量")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从指定检查点恢复训练，'latest'表示最新的")

    # --- 日志与报告参数 ---
    parser.add_argument("--logging_dir", type=str, default="logs", help="TensorBoard日志目录")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="报告结果的平台 (e.g., 'tensorboard', 'wandb')")
    parser.add_argument("--log_trainable_parameters", action="store_true", help="是否记录可训练的参数")

    # --- 其他参数 ---
    parser.add_argument("--seed", type=int, default=None, help="用于可复现训练的随机种子")
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练的本地排名")
    parser.add_argument("--conditioning_dropout_prob", type=float, default=0.1, help="条件丢弃概率")
    parser.add_argument("--rank", type=int, default=128, help="LoRA更新矩阵的维度")
    
    # (以下为项目中可能存在的其他参数，保留以确保兼容性)
    parser.add_argument("--dataset_type", type=str, default='ubc')
    parser.add_argument("--use_ema", action="store_true", help="是否使用EMA模型")
    parser.add_argument("--non_ema_revision", type=str, default=None, required=False)
    parser.add_argument("--pretrain_unet", type=str, default=None)
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--video_folder", type=str, default=None)
    parser.add_argument("--condition_folder", type=str, default=None)
    parser.add_argument("--motion_folder", type=str, default=None)
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--ref_augment", action="store_true")
    parser.add_argument("--train_stage", type=int, default=2)
    parser.add_argument("--posenet_model_name_or_path", type=str, default=None)
    parser.add_argument("--face_encoder_model_name_or_path", type=str, default=None)
    parser.add_argument("--unet_model_name_or_path", type=str, default=None)
    parser.add_argument("--finetune_mode", type=bool, default=False)
    parser.add_argument("--posenet_model_finetune_path", type=str, default=None)
    parser.add_argument("--face_encoder_finetune_path", type=str, default=None)
    parser.add_argument("--unet_model_finetune_path", type=str, default=None)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

def main():
    """
    主训练函数
    """
    # --- 1. 初始化和环境设置 ---
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    torch.multiprocessing.set_start_method('spawn')

    args = parse_args()

    # 初始化 Accelerator
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    
    # 设置随机种子生成器
    generator = torch.Generator(device=accelerator.device).manual_seed(23123134)

    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    # 创建输出目录
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # --- 2. 加载模型组件 ---
    logger.info(f"从 {args.pretrained_model_name_or_path} 加载模型...")
    feature_extractor = CLIPImageProcessor.from_pretrained(args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision)
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision)
    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant="fp16")
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet,
        subfolder="unet",
        low_cpu_mem_usage=True,
        variant="fp16"
    )
    # 初始化自定义模块
    pose_net = PoseNet(noise_latent_channels=unet.config.block_out_channels[0])
    face_encoder = FusionFaceId(cross_attention_dim=1024, id_embeddings_dim=512, clip_embeddings_dim=1024, num_tokens=4)
    face_model = FaceModel()

    # --- 3. 设置注意力处理器 (Attention Processors) ---
    # 这是模型动画能力的关键，通过替换默认的注意力处理器来实现
    lora_rank = 128
    attn_procs = {}
    unet_svd = unet.state_dict()
    for name in unet.attn_processors.keys():
        if "transformer_blocks" in name and "temporal_transformer_blocks" not in name: # 空间注意力
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            
            if cross_attention_dim is None: # 自注意力
                attn_procs[name] = AnimationAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
            else: # 交叉注意力
                layer_name = name.split(".processor")[0]
                weights = {
                    "id_to_k.weight": unet_svd[layer_name + ".to_k.weight"],
                    "id_to_v.weight": unet_svd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = AnimationIDAttnNormalizedProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
                attn_procs[name].load_state_dict(weights, strict=False)
        elif "temporal_transformer_blocks" in name: # 时间注意力，保持 SVD 原样
            attn_procs[name] = XFormersAttnProcessor()
    unet.set_attn_processor(attn_procs)

    # 如果是微调模式，加载已有的模型权重
    if args.finetune_mode:
        # ... (微调模式加载逻辑) ...
        pass

    # --- 4. 准备训练 ---
    # 冻结不需要训练的模型部分
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    pose_net.requires_grad_(False)
    face_encoder.requires_grad_(False)

    # 设置权重数据类型
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 将冻结的模型移至设备
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # 启用xformers以优化显存
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers 未安装，请先安装。")

    # 启用梯度检查点以节省显存
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # --- 5. 设置优化器 (选择性参数训练) ---
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_cls = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError("请安装 bitsandbytes 以使用8位Adam优化器。")
    else:
        optimizer_cls = torch.optim.AdamW

    # 将需要训练的参数加入列表
    pose_net.requires_grad_(True)
    face_encoder.requires_grad_(True)
    parameters_list = []
    parameters_list.extend([{"params": para, "lr": args.learning_rate} for para in pose_net.parameters()])
    parameters_list.extend([{"params": para, "lr": args.learning_rate} for para in face_encoder.parameters()])
    
    # 只训练U-Net中注意力相关的部分
    for name, para in unet.named_parameters():
        if "attentions" in name:
            para.requires_grad = True
            parameters_list.append({"params": para})
        else:
            para.requires_grad = False

    optimizer = optimizer_cls(
        parameters_list,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # --- 6. 数据加载逻辑 (健壮版本) ---
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes
    root_path = args.data_root_path
    
    active_datasets = []

    # --- 处理 rec (512x512) 数据集 ---
    txt_path_1 = args.rec_data_path
    if txt_path_1 and os.path.exists(txt_path_1) and os.path.getsize(txt_path_1) > 0:
        logger.info(f"正在加载 REC 数据集 (512x512) 从: {txt_path_1}")
        train_dataset_1 = LargeScaleAnimationVideos(
            root_path=root_path, txt_path=txt_path_1, width=512, height=512,
            n_sample_frames=args.sample_n_frames, sample_frame_rate=4,
            app=face_model.app, handler_ante=face_model.handler_ante, face_helper=face_model.face_helper
        )
        if len(train_dataset_1) > 0:
            active_datasets.append(train_dataset_1)
        else:
            logger.warning(f"REC 数据集文件 {txt_path_1} 存在但内部为空，已跳过。")
    else:
        logger.info("未找到或 REC 数据集为空，已跳过。")

    # --- 处理 vec (576x1024) 数据集 ---
    txt_path_2 = args.vec_data_path
    if txt_path_2 and os.path.exists(txt_path_2) and os.path.getsize(txt_path_2) > 0:
        logger.info(f"正在加载 VEC 数据集 (576x1024) 从: {txt_path_2}")
        train_dataset_2 = LargeScaleAnimationVideos(
            root_path=root_path, txt_path=txt_path_2, width=576, height=1024,
            n_sample_frames=args.sample_n_frames, sample_frame_rate=4,
            app=face_model.app, handler_ante=face_model.handler_ante, face_helper=face_model.face_helper
        )
        if len(train_dataset_2) > 0:
            active_datasets.append(train_dataset_2)
        else:
            logger.warning(f"VEC 数据集文件 {txt_path_2} 存在但内部为空，已跳过。")
    else:
        logger.info("未找到或 VEC 数据集为空，已跳过。")
    
    # --- 合并数据集并创建单个 DataLoader ---
    if not active_datasets:
        raise ValueError("错误：所有数据集均为空，没有可用于训练的数据。请检查数据路径。")

    logger.info(f"共加载了 {len(active_datasets)} 个有效数据集。")
    combined_dataset = ConcatDataset(active_datasets)
    
    train_dataloader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )

    # --- 7. 设置学习率调度器和计算总步数 ---
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # 使用 accelerator.prepare 包装所有组件
    unet, pose_net, face_encoder, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        unet, pose_net, face_encoder, optimizer, lr_scheduler, train_dataloader
    )

    # 重新计算训练总步数和轮数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # 初始化日志追踪器
    if accelerator.is_main_process:
        accelerator.init_trackers("StableAnimator", config=vars(args))

    # --- 8. 主训练循环 ---
    total_batch_size = args.per_gpu_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** 开始训练 *****")
    logger.info(f"  总样本数 = {len(combined_dataset)}")
    logger.info(f"  总轮数 = {args.num_train_epochs}")
    logger.info(f"  每台设备批大小 = {args.per_gpu_batch_size}")
    logger.info(f"  总训练批大小 = {total_batch_size}")
    logger.info(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
    logger.info(f"  总优化步数 = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    def encode_image(pixel_values):
        # ... (函数实现保持不变)
        pixel_values = _resize_with_antialiasing(pixel_values, (224, 224))
        pixel_values = (pixel_values + 1.0) / 2.0
        pixel_values = pixel_values.to(torch.float32)
        pixel_values = feature_extractor(
            images=pixel_values, do_normalize=True, do_center_crop=False,
            do_resize=False, do_rescale=False, return_tensors="pt",
        ).pixel_values
        pixel_values = pixel_values.to(device=accelerator.device, dtype=image_encoder.dtype)
        image_embeddings = image_encoder(pixel_values).image_embeds
        image_embeddings= image_embeddings.unsqueeze(1)
        return image_embeddings

    def _get_add_time_ids(fps, motion_bucket_id, noise_aug_strength, dtype, batch_size, unet=None, device=None):
        # ... (函数实现保持不变)
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids

    # 从检查点恢复
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        if path is None:
            accelerator.print(f"检查点 '{args.resume_from_checkpoint}' 不存在。开始新的训练。")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"从检查点 {path} 恢复训练...")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        pose_net.train()
        face_encoder.train()
        unet.train()
        train_loss = 0.0

        # --- 核心简化：直接遍历合并后的 train_dataloader ---
        for step, batch in enumerate(train_dataloader):
            # 如果从检查点恢复，跳过已经训练过的步骤
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(pose_net, face_encoder, unet):
                # --- 前向传播 ---
                with accelerator.autocast():
                    pixel_values = batch["pixel_values"].to(weight_dtype).to(accelerator.device, non_blocking=True)
                    conditional_pixel_values = batch["reference_image"].to(weight_dtype).to(accelerator.device, non_blocking=True)
                    
                    # 编码为 latent
                    latents = tensor_to_vae_latent(pixel_values, vae).to(dtype=weight_dtype)
                    
                    # 获取图像 embedding 作为条件
                    encoder_hidden_states = encode_image(conditional_pixel_values).to(dtype=weight_dtype)
                    image_embed = encoder_hidden_states.clone()

                    train_noise_aug = 0.02
                    conditional_pixel_values = conditional_pixel_values + train_noise_aug * randn_tensor(conditional_pixel_values.shape, generator=generator, device=conditional_pixel_values.device, dtype=conditional_pixel_values.dtype)
                    conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae, scale=False)

                    # 加噪过程
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    sigmas = rand_cosine_interpolated(shape=[bsz,], image_d=image_d, noise_d_low=noise_d_low, noise_d_high=noise_d_high, sigma_data=sigma_data, min_value=min_value, max_value=max_value).to(latents.device, dtype=weight_dtype)
                    sigmas_reshaped = sigmas.clone()
                    while len(sigmas_reshaped.shape) < len(latents.shape):
                        sigmas_reshaped = sigmas_reshaped.unsqueeze(-1)
                    noisy_latents  = latents + noise * sigmas_reshaped
                    timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas]).to(latents.device, dtype=weight_dtype)
                    inp_noisy_latents = noisy_latents  / ((sigmas_reshaped**2 + 1) ** 0.5)
                    
                    added_time_ids = _get_add_time_ids(
                        fps=6, motion_bucket_id=127.0, noise_aug_strength=train_noise_aug,
                        dtype=encoder_hidden_states.dtype, batch_size=bsz, device=latents.device
                    )
                    
                    # 条件丢弃 (classifier-free guidance)
                    if args.conditioning_dropout_prob is not None:
                        # ... (条件丢弃逻辑) ...
                        random_p = torch.rand(bsz, device=latents.device, generator=generator)
                        prompt_mask = (random_p < 2 * args.conditioning_dropout_prob).reshape(bsz, 1, 1)
                        null_conditioning = torch.zeros_like(encoder_hidden_states)
                        encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)
                        image_mask_dtype = conditional_latents.dtype
                        image_mask = 1 - ((random_p >= args.conditioning_dropout_prob).to(image_mask_dtype) * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype))
                        image_mask = image_mask.reshape(bsz, 1, 1, 1)
                        conditional_latents = image_mask * conditional_latents

                    # 准备模型输入
                    conditional_latents = conditional_latents.unsqueeze(1).repeat(1, noisy_latents.shape[1], 1, 1, 1)
                    pose_pixels = batch["pose_pixels"].to(dtype=weight_dtype, device=accelerator.device, non_blocking=True)
                    faceid_embeds = batch["faceid_embeds"].to(dtype=weight_dtype, device=accelerator.device, non_blocking=True)
                    
                    # 获取姿态和人脸ID的 latent
                    pose_latents = pose_net(pose_pixels)
                    faceid_latents = face_encoder(faceid_embeds, image_embed)
                    
                    # 拼接输入
                    inp_noisy_latents = torch.cat([inp_noisy_latents, conditional_latents], dim=2)
                    encoder_hidden_states = torch.cat([encoder_hidden_states, faceid_latents], dim=1)
                    target = latents

                    # 模型预测
                    model_pred = unet(
                        inp_noisy_latents.to(latents.dtype), 
                        timesteps, 
                        encoder_hidden_states.to(latents.dtype),
                        added_time_ids=added_time_ids,
                        pose_latents=pose_latents.to(latents.dtype),
                    ).sample
                    
                    # --- 损失计算 ---
                    sigmas = sigmas_reshaped
                    c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                    c_skip = 1 / (sigmas**2 + 1)
                    denoised_latents = model_pred * c_out + c_skip * noisy_latents
                    weighing = (1 + sigmas ** 2) * (sigmas**-2.0)
                    
                    # 根据人脸区域加权损失
                    tgt_face_masks = batch["tgt_face_masks"].to(dtype=weight_dtype, device=accelerator.device, non_blocking=True)
                    tgt_face_masks = rearrange(tgt_face_masks, "b f c h w -> (b f) c h w")
                    tgt_face_masks = F.interpolate(tgt_face_masks, size=(target.size()[-2], target.size()[-1]), mode='nearest')
                    tgt_face_masks = rearrange(tgt_face_masks, "(b f) c h w -> b f c h w", f=args.sample_n_frames)

                    loss = torch.mean(
                        (weighing.float() * (denoised_latents.float() - target.float()) ** 2 * (1 + tgt_face_masks)).reshape(target.shape[0], -1),
                        dim=1,
                    ).mean()

                # --- 反向传播 ---
                avg_loss = accelerator.gather(loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                accelerator.backward(loss)
                
                # --- 核心修正：在这里添加梯度裁剪 ---
                if accelerator.sync_gradients:
                    # 获取所有需要训练的参数
                    params_to_clip = (
                        list(pose_net.parameters()) +
                        list(face_encoder.parameters()) +
                        list(unet.parameters())
                    )
                    # 对可训练的参数执行梯度裁剪
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # 清理显存
                with torch.cuda.device(latents.device):
                    torch.cuda.empty_cache()

            # --- 检查点和验证 ---
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # 保存检查点
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    if args.checkpoints_total_limit is not None:
                        # ... (管理检查点数量的逻辑) ...
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]
                            for removing_checkpoint in removing_checkpoints:
                                shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))
                    
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    # 单独保存可训练模型的权重
                    unwrap_unet = accelerator.unwrap_model(unet)
                    unwrap_pose_net = accelerator.unwrap_model(pose_net)
                    unwrap_face_encoder = accelerator.unwrap_model(face_encoder)
                    torch.save(unwrap_unet.state_dict(), os.path.join(save_path, f"unet-{global_step}.pth"))
                    torch.save(unwrap_pose_net.state_dict(), os.path.join(save_path, f"pose_net-{global_step}.pth"))
                    torch.save(unwrap_face_encoder.state_dict(), os.path.join(save_path, f"face_encoder-{global_step}.pth"))
                    logger.info(f"已保存检查点到 {save_path}")

                # 执行验证
                if accelerator.is_main_process and global_step % args.validation_steps == 0:
                    logger.info("开始运行验证...")
                    log_validation(
                        vae=vae, image_encoder=image_encoder, unet=unet, pose_net=pose_net, face_encoder=face_encoder,
                        app=face_model.app, face_helper=face_model.face_helper, handler_ante=face_model.handler_ante,
                        scheduler=noise_scheduler, accelerator=accelerator, feature_extractor=feature_extractor,
                        width=args.validation_width, height=args.validation_height, torch_dtype=weight_dtype,
                        validation_image_folder=args.validation_image_folder, validation_image=args.validation_image,
                        validation_control_folder=args.validation_control_folder, output_dir=args.output_dir,
                        generator=generator, global_step=global_step, num_validation_cases=1,
                    )
                    with torch.cuda.device(latents.device):
                        torch.cuda.empty_cache()

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        
        if global_step >= args.max_train_steps:
            break
            
    # 训练结束，保存最终模型
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-last")
        accelerator.save_state(save_path)
        logger.info(f"最终模型状态已保存到 {save_path}")


def log_validation(
        vae, image_encoder, unet, pose_net, face_encoder, app, face_helper, handler_ante,
        scheduler, accelerator, feature_extractor, width, height, torch_dtype,
        validation_image_folder, validation_image, validation_control_folder,
        output_dir, generator, global_step, num_validation_cases=1,
):
    """
    执行验证流程，生成样本视频
    """
    logger.info("正在准备验证 pipeline...")
    # 解包模型以进行推理
    validation_unet = accelerator.unwrap_model(unet)
    validation_image_encoder = accelerator.unwrap_model(image_encoder)
    validation_vae = accelerator.unwrap_model(vae)
    validation_pose_net = accelerator.unwrap_model(pose_net)
    validation_face_encoder = accelerator.unwrap_model(face_encoder)

    # 创建验证 pipeline
    pipeline = ValidationAnimationPipeline(
        vae=validation_vae, image_encoder=validation_image_encoder, unet=validation_unet,
        scheduler=scheduler, feature_extractor=feature_extractor,
        pose_net=validation_pose_net, face_encoder=validation_face_encoder,
    )
    pipeline = pipeline.to(accelerator.device)
    
    # 加载验证数据
    validation_image_path = validation_image
    if validation_image is None:
        validation_images = load_images_from_folder(validation_image_folder)
        validation_image = validation_images[0]
    else:
        validation_image = Image.open(validation_image).convert('RGB')
    validation_control_images = load_images_from_folder(validation_control_folder)

    val_save_dir = os.path.join(output_dir, "validation_images")
    os.makedirs(val_save_dir, exist_ok=True)

    with accelerator.autocast():
        for val_img_idx in range(num_validation_cases):
            num_frames = len(validation_control_images)
            logger.info(f"正在生成第 {val_img_idx + 1} 个验证视频，共 {num_frames} 帧...")

            # 提取参考图的人脸 embedding
            face_helper.clean_all()
            validation_face = cv2.imread(validation_image_path)
            validation_image_face_info = app.get(validation_face)
            if len(validation_image_face_info) > 0:
                validation_image_face_info = sorted(validation_image_face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
                validation_image_id_ante_embedding = validation_image_face_info['embedding']
            else:
                # 如果检测不到，尝试对齐后再提取
                face_helper.read_image(validation_face)
                face_helper.get_face_landmarks_5(only_center_face=True)
                face_helper.align_warp_face()
                if len(face_helper.cropped_faces) == 0:
                    validation_image_id_ante_embedding = np.zeros((512,))
                else:
                    validation_image_align_face = face_helper.cropped_faces[0]
                    validation_image_id_ante_embedding = handler_ante.get_feat(validation_image_align_face)

            # 运行 pipeline 生成视频帧
            video_frames = pipeline(
                image=validation_image, image_pose=validation_control_images,
                height=height, width=width, num_frames=num_frames,
                tile_size=num_frames, tile_overlap=4, decode_chunk_size=4,
                motion_bucket_id=127., fps=7, min_guidance_scale=3, max_guidance_scale=3,
                noise_aug_strength=0.02, num_inference_steps=25, generator=generator,
                output_type="pil", validation_image_id_ante_embedding=validation_image_id_ante_embedding,
            ).frames[0]
            
            # 保存为 GIF
            out_file = os.path.join(val_save_dir, f"step_{global_step}_val_img_{val_img_idx}.mp4")
            video_frames_np = [np.array(img) for img in video_frames]
            export_to_gif(video_frames_np, out_file, 8)
            logger.info(f"验证视频已保存到 {out_file.replace('.mp4', '.gif')}")

    del pipeline
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
