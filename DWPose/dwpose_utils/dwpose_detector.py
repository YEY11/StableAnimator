# -*- coding: utf-8 -*-
"""
本脚本定义了 DWposeDetectorAligned 类，作为整个姿态检测流程的高级封装。

它负责:
1. 实例化核心的 Wholebody 检测器。
2. 调用 Wholebody 检测器获取原始的关键点和置信度。
3. 对原始输出进行后处理，将其组织成一个包含 'bodies', 'hands', 'faces' 
   的结构化字典，方便上层应用使用。
4. 提供一个全局单例 dwpose_detector_aligned 供其他脚本直接导入和使用。
"""
import os
import numpy as np
import torch

from .wholebody import Wholebody

# 设置环境变量，防止某些情况下因重复加载库而报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 自动选择使用CUDA或CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DWposeDetectorAligned:
    """
    DWPose全身姿态检测器的封装类。
    """
    def __init__(self, device='cpu'):
        """
        初始化检测器，加载模型。
        """
        self.pose_estimation = Wholebody()

    def release_memory(self):
        """
        释放模型占用的内存。
        """
        if hasattr(self, 'pose_estimation'):
            del self.pose_estimation
            import gc; gc.collect()

    def __call__(self, oriImg):
        """
        对输入的图像进行姿态检测。

        Args:
            oriImg (np.array): 输入的原始图像，格式为 (H, W, C) 的RGB图像。

        Returns:
            dict: 一个包含检测结果的字典，结构如下:
                  {
                      'bodies': {'candidate': ..., 'subset': ..., 'score': ...},
                      'hands': ...,
                      'hands_score': ...,
                      'faces': ...,
                      'faces_score': ...
                  }
        """
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            # 1. 调用底层的Wholebody检测器，获取原始关键点和置信度
            candidate, score = self.pose_estimation(oriImg)
            nums, _, locs = candidate.shape # nums: 检测到的人数

            # 2. 坐标归一化
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)

            # 3. 重组身体(body)部分的数据
            # candidate[:, :18] 是身体的18个关键点
            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs) # 将多人的身体关键点展平成一个列表
            
            # 创建 'subset'，用于识别人体实例和关节点对应关系
            subset = score[:, :18].copy()
            for i in range(len(subset)): # 遍历每一个人
                for j in range(len(subset[i])): # 遍历每一个关节点
                    if subset[i][j] > 0.3: # 如果置信度足够高
                        # 将subset的值设为该关节点在body列表中的全局索引
                        subset[i][j] = int(18 * i + j)
                    else:
                        # 否则设为-1，表示未检测到
                        subset[i][j] = -1

            # 4. 提取面部(face)和手部(hand)的数据
            faces = candidate[:, 24:92]
            # 左右手的数据是分开的，这里将它们垂直堆叠起来
            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])
            
            # 提取对应的置信度分数
            faces_score = score[:, 24:92]
            hands_score = np.vstack([score[:, 92:113], score[:, 113:]])
            
            # 5. 将所有部分打包成最终的字典格式
            bodies = dict(candidate=body, subset=subset, score=score[:, :18])
            pose = dict(bodies=bodies, hands=hands, hands_score=hands_score, faces=faces, faces_score=faces_score)

            return pose

# 创建一个全局的检测器实例，其他脚本可以直接 from . import dwpose_detector_aligned 来使用
dwpose_detector_aligned = DWposeDetectorAligned(device=device)
