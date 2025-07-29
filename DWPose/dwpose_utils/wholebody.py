# -*- coding: utf-8 -*-
"""
本脚本定义了 Wholebody 类，是DWPose模型的核心。

它实现了一个两阶段的姿态估计流程:
1. **人体检测**: 使用YOLOX ONNX模型在输入图像中检测出所有的人，并获取他们的边界框(bounding box)。
2. **姿态估计**: 对每一个检测到的边界框区域，使用RTMPose ONNX模型估计出内部人体的
   133个关键点（包括身体、脚、脸、手）。
3. **后处理**: 对关键点进行重组和排序，以匹配OpenPose的格式，并计算出'neck'（脖子）关键点。
"""
import cv2
import numpy as np
import os

import onnxruntime as ort
from .onnxdet import inference_detector
from .onnxpose import inference_pose

class Wholebody:
    """
    整合人体检测和姿态估计，实现完整的全身姿态提取。
    """
    def __init__(self):
        """
        初始化函数，加载所需的人体检测和姿态估计ONNX模型。
        """
        device = 'cuda:0'
        # 根据设备选择ONNX Runtime的执行提供者（CPU或CUDA）
        providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CUDAExecutionProvider']
        
        # 定位ONNX模型文件的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.abspath(os.path.join(current_dir, "../.."))
        onnx_det = os.path.join(project_dir, "checkpoints/DWPose/yolox_l.onnx") # YOLOX检测模型
        onnx_pose = os.path.join(project_dir, "checkpoints/DWPose/dw-ll_ucoco_384.onnx") # RTMPose姿态估计模型

        # 创建ONNX Runtime的推理会话
        self.session_det = ort.InferenceSession(path_or_bytes=onnx_det, providers=providers)
        self.session_pose = ort.InferenceSession(path_or_bytes=onnx_pose, providers=providers)
    
    def __call__(self, oriImg):
        """
        对输入图像执行完整的全身姿态估计。

        Args:
            oriImg (np.array): 输入的原始图像，格式为 (H, W, C) 的RGB图像。

        Returns:
            tuple: 包含两个元素的元组:
                - keypoints (np.array): 检测到的关键点坐标，形状为 (N, 133, 2)，N是人数。
                - scores (np.array): 对应关键点的置信度，形状为 (N, 133)。
        """
        # 第一阶段：人体检测，获取边界框
        det_result = inference_detector(self.session_det, oriImg)
        
        # 第二阶段：姿态估计，根据边界框提取关键点
        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)

        # 后处理步骤
        # 将关键点坐标和置信度分数合并在一起
        keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1)
        
        # 1. 计算'neck'（脖子）关键点，通常是左右肩膀的平均值
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # 计算脖子关键点的置信度，只有当左右肩膀置信度都>0.3时才认为有效
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        
        # 将计算出的脖子关键点插入到第17个位置
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1)
        
        # 2. 关键点重排序，将MMPose的索引顺序转换为OpenPose的索引顺序
        # 这是为了与许多下游应用（如绘图函数）的期望格式保持一致
        mmpose_idx = [
            17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
        ]
        openpose_idx = [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
        ]
        new_keypoints_info[:, openpose_idx] = \
            new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        # 最终分离出关键点坐标和置信度
        keypoints, scores = keypoints_info[
            ..., :2], keypoints_info[..., 2]
        
        return keypoints, scores
