# -*- coding: utf-8 -*-
"""
本脚本定义了 PoseNet 模型，它是 StableAnimator 架构中负责处理姿态信息的关键组件。

核心功能:
PoseNet 的主要任务是将输入的姿态序列图像（由 DWPose 模块生成）编码成一个特征图。
这个特征图在维度上与 Stable Video Diffusion (SVD) 的 U-Net 中间层的噪声潜在表示(noisy latent)完全匹配。
编码后的姿态特征图会直接与噪声潜在表示相加，从而将运动指导信息注入到视频的生成过程中。
这是一种非常直接且有效的条件注入方式，引导 U-Net 在去噪的每一步都遵循给定的姿态序列。

模型结构:
该模型本质上是一个轻量级的卷积神经网络（CNN），其结构可以概括为：
1. 一系列交替的 2D 卷积层 (Conv2d) 和 SiLU 激活函数。
2. 卷积层中，一部分用于提取更深层次的特征（通道数增加），另一部分通过 stride=2 实现下采样，
   逐步将输入姿态图的空间尺寸（H, W）缩小，以匹配 U-Net 中间层的尺寸。
3. 一个最终的 1x1 卷积层 (final_proj)，用于将特征图的通道数精确地映射到 SVD U-Net 
   所期望的通道数（默认为 320）。
4. 一个可学习的缩放因子 `scale`，用于在训练时自动调整姿态特征的注入强度。

权重初始化:
- 卷积层使用何恺明(He)初始化，这是一种在深度网络中被证明有效的初始化方法。
- 最后的投影层`final_proj`的权重和偏置被初始化为零。这是一种常见的实践，
  被称为“零初始化”，它确保在训练初期，这个新加入的控制模块（PoseNet）
  对原始的SVD模型没有影响，让模型可以从一个稳定的状态开始学习如何利用姿态信息。
"""

from pathlib import Path

import einops  # 一个非常方便的张量操作库，用于重排维度
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from diffusers.models.modeling_utils import ModelMixin # 继承自Diffusers库，便于集成和管理

class PoseNet(ModelMixin):
    """
    PoseNet 模型，用于将姿态图像编码为特征图。
    """
    def __init__(self, noise_latent_channels=320):
        """
        初始化 PoseNet 模型。

        Args:
            noise_latent_channels (int): 目标噪声潜在表示的通道数。
                                         对于SVD模型，这个值通常是320。
        """
        super().__init__()
        
        # 定义一系列卷积层，用于特征提取和下采样
        # 输入: (B*F, 3, H, W) - 姿态图
        # 输出: (B*F, 128, H/8, W/8) - 经过编码的姿态特征
        self.conv_layers = nn.Sequential(
            # 初始卷积，保持尺寸
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.SiLU(),
            # 第一次下采样: H -> H/2, W -> W/2。通道数 3 -> 16
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),

            # 保持尺寸
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.SiLU(),
            # 第二次下采样: H/2 -> H/4, W/2 -> W/4。通道数 16 -> 32
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),

            # 保持尺寸
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.SiLU(),
            # 第三次下采样: H/4 -> H/8, W/4 -> W/8。通道数 32 -> 64
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),

            # 保持尺寸
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.SiLU(),
            # 增加特征维度，通道数 64 -> 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.SiLU()
        )

        # 最终的投影层，使用 1x1 卷积将通道数调整为目标通道数
        # 输入: (B*F, 128, H/8, W/8)
        # 输出: (B*F, 320, H/8, W/8)
        self.final_proj = nn.Conv2d(in_channels=128, out_channels=noise_latent_channels, kernel_size=1)

        # 初始化权重
        self._initialize_weights()

        # 定义一个可学习的缩放参数，用于调整姿态特征的注入强度
        # 训练时模型可以自动学习一个最佳的缩放值
        self.scale = nn.Parameter(torch.ones(1) * 2)

    def _initialize_weights(self):
        """
        使用特定的策略初始化网络权重。
        """
        # 遍历所有卷积层进行 He 初始化
        for m in self.conv_layers:
            if isinstance(m, nn.Conv2d):
                # 何恺明(He)正态分布初始化，适用于ReLU及其变体(如SiLU)
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                init.normal_(m.weight, mean=0.0, std=np.sqrt(2. / n))
                if m.bias is not None:
                    init.zeros_(m.bias) # 偏置初始化为0
        
        # 对最终的投影层进行零初始化
        # 确保在训练开始时，PoseNet的输出为0，不干扰原始的SVD模型
        init.zeros_(self.final_proj.weight)
        if self.final_proj.bias is not None:
            init.zeros_(self.final_proj.bias)

    def forward(self, x):
        """
        模型的前向传播过程。

        Args:
            x (torch.Tensor): 输入的姿态图像张量。
                              形状可以是 (B, F, C, H, W) 或 (B*F, C, H, W)。
                              B: batch size, F: 帧数, C: 通道数(3), H: 高, W: 宽。

        Returns:
            torch.Tensor: 编码后的姿态特征图，准备与噪声潜在表示相加。
                          形状为 (B*F, noise_latent_channels, H/8, W/8)。
        """
        # 检查输入张量的维度，如果是5维(B, F, C, H, W)，则将其重排为4维
        # 这是为了让2D卷积层可以处理视频帧序列
        if x.ndim == 5:
            x = einops.rearrange(x, "b f c h w -> (b f) c h w")
        
        # 通过卷积层进行特征提取和下采样
        x = self.conv_layers(x)
        
        # 通过最终投影层调整通道数
        x = self.final_proj(x)

        # 应用可学习的缩放因子
        return x * self.scale

    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        """
        从预训练的权重文件中加载模型。

        Args:
            pretrained_model_path (str): .pth 权重文件的路径。

        Returns:
            PoseNet: 加载了预训练权重的模型实例。
        """
        if not Path(pretrained_model_path).exists():
            print(f"模型文件不存在: {pretrained_model_path}")
        print(f"从 {pretrained_model_path} 加载 PoseNet 预训练权重。")

        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        model = PoseNet(noise_latent_channels=320)

        # 严格模式加载权重，确保模型结构与权重文件完全匹配
        model.load_state_dict(state_dict, strict=True)

        return model
