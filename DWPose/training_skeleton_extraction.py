# -*- coding: utf-8 -*-
"""
本脚本用于为StableAnimator项目的训练数据进行预处理，主要功能是从视频帧图像中提取人体姿态信息。

工作流程:
1. 接收命令行参数，指定要处理的数据集路径和范围。
2. 遍历指定的数据集文件夹（如 'animation_data/rec/00001/'）。
3. 对 'images' 子文件夹下的每一张图片（视频帧）进行处理。
4. 使用 DWpose 模型检测图像中的人体姿态，包括身体、手部和面部的关键点。
5. 将检测到的关键点绘制成一张新的、背景为黑色的姿态图。
6. 将生成的姿态图保存到对应的 'poses' 子文件夹下。

命令行使用示例:
python DWPose/training_skeleton_extraction.py --root_path="path/to/animation_data" --name="rec" --start=1 --end=500
"""

import math
import argparse
import cv2
import os
from tqdm import tqdm
import decord
import numpy as np
import matplotlib

# 从dwpose_utils工具包中导入核心的姿态检测器
from dwpose_utils.dwpose_detector import dwpose_detector_aligned

# 定义一个极小值，用于浮点数比较，避免因精度问题导致的错误
eps = 0.01

def alpha_blend_color(color, alpha):
    """
    根据关键点的置信度（alpha）对颜色进行混合，实现半透明效果。
    置信度越高，颜色越不透明。

    Args:
        color (list): RGB颜色值，如 [255, 0, 0]。
        alpha (float): 透明度/置信度，范围在 0.0 到 1.0 之间。

    Returns:
        list: 混合后的RGB颜色值。
    """
    return [int(c * alpha) for c in color]

def draw_bodypose_aligned(canvas, candidate, subset, score):
    """
    在画布上绘制身体姿态的骨骼和关节点。

    Args:
        canvas (np.array): 用于绘制的画布图像。
        candidate (np.array): 检测到的所有身体关键点的坐标列表。坐标是归一化的 (0-1范围)。
        subset (np.array): 用于区分图中不同的人体实例。
        score (np.array): 每个关键点的置信度分数。

    Returns:
        np.array: 绘制好身体姿态的画布。
    """
    H, W, C = canvas.shape  # 获取画布的高、宽、通道数
    candidate = np.array(candidate)
    subset = np.array(subset)
    stickwidth = 4  # 骨骼线条的宽度

    # 定义身体骨骼的连接顺序，每个元素是[起点索引, 终点索引]
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]
    # 为每个骨骼定义不同的颜色
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    
    # 遍历17个身体主要部分（不包括耳朵）来绘制骨骼
    for i in range(17):
        # 遍历图中的每一个人
        for n in range(len(subset)):
            # 获取当前骨骼两端关节点的索引
            index = subset[n][np.array(limbSeq[i]) - 1]
            # 获取对应关节点的置信度
            conf = score[n][np.array(limbSeq[i]) - 1]
            # 如果任意一端的置信度低于0.3，则不绘制该骨骼
            if conf[0] < 0.3 or conf[1] < 0.3:
                continue
            # 将归一化的坐标转换为图像实际坐标
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            # 计算骨骼中点坐标
            mX = np.mean(X)
            mY = np.mean(Y)
            # 计算骨骼长度和角度
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            # 使用一个填充的椭圆来模拟有宽度的骨骼
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            # 绘制填充椭圆，颜色根据置信度混合
            cv2.fillConvexPoly(canvas, polygon, alpha_blend_color(colors[i], conf[0] * conf[1]))

    # 将整个画布的亮度降低，使得关节点更突出
    canvas = (canvas * 0.6).astype(np.uint8)

    # 遍历18个关节点来绘制圆点
    for i in range(18):
        for n in range(len(subset)):
            # 获取关节点索引
            index = int(subset[n][i])
            # 如果索引为-1，表示未检测到该关节点
            if index == -1:
                continue
            # 获取关节点坐标和置信度
            x, y = candidate[index][0:2]
            conf = score[n][i]
            # 转换为图像实际坐标
            x = int(x * W)
            y = int(y * H)
            # 绘制实心圆点，颜色根据置信度混合
            cv2.circle(canvas, (int(x), int(y)), 4, alpha_blend_color(colors[i], conf), thickness=-1)

    return canvas

def draw_handpose_aligned(canvas, all_hand_peaks, all_hand_scores):
    """
    在画布上绘制手部姿态。

    Args:
        canvas (np.array): 用于绘制的画布图像。
        all_hand_peaks (list): 包含所有检测到的手部关键点坐标的列表。
        all_hand_scores (list): 每个手部关键点的置信度分数。

    Returns:
        np.array: 绘制好手部姿态的画布。
    """
    H, W, C = canvas.shape

    # 定义手指骨骼的连接顺序
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    # 遍历检测到的每一只手
    for peaks, scores in zip(all_hand_peaks, all_hand_scores):
        # 绘制手指骨骼连接线
        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1, y1 = int(x1 * W), int(y1 * H)
            x2, y2 = int(x2 * W), int(y2 * H)
            # 结合两端点的置信度作为线条的置信度
            score_val = int(scores[e[0]] * scores[e[1]] * 255)
            # 只有当坐标有效时才绘制
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                # 使用HSV颜色空间生成彩虹色，便于区分不同的手指
                color = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * score_val
                cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=2)
        
        # 绘制手指关节点
        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x, y = int(x * W), int(y * H)
            score_val = int(scores[i] * 255)
            if x > eps and y > eps:
                # 关节点用蓝色系表示，亮度由置信度决定
                cv2.circle(canvas, (x, y), 4, (0, 0, score_val), thickness=-1)
    return canvas

def draw_facepose_aligned(canvas, all_lmks, all_scores):
    """
    在画布上绘制面部关键点。

    Args:
        canvas (np.array): 用于绘制的画布图像。
        all_lmks (list): 包含所有检测到的面部关键点坐标的列表。
        all_scores (list): 每个面部关键点的置信度分数。

    Returns:
        np.array: 绘制好面部关键点的画布。
    """
    H, W, C = canvas.shape
    # 遍历检测到的每一张脸
    for lmks, scores in zip(all_lmks, all_scores):
        # 遍历脸上的每一个关键点
        for lmk, score in zip(lmks, scores):
            x, y = lmk
            # 将归一化坐标转换为图像实际坐标
            x, y = int(x * W), int(y * H)
            # 将置信度转换为0-255范围的整数
            conf = int(score * 255)
            if x > eps and y > eps:
                # 绘制灰度圆点，亮度由置信度决定
                cv2.circle(canvas, (x, y), 3, (conf, conf, conf), thickness=-1)
    return canvas

def draw_pose_aligned(pose, H, W, ref_w=2160):
    """
    将身体、手部和面部姿态统一绘制到一张画布上。

    Args:
        pose (dict): 包含所有姿态信息的字典，由 dwpose_detector 返回。
        H (int): 目标图像的高度。
        W (int): 目标图像的宽度。
        ref_w (int): 参考宽度，用于临时放大画布以获得更好的绘制质量。

    Returns:
        np.array: 最终生成的姿态图，格式为 (C, H, W)。
    """
    # 从pose字典中解包出身体、面部、手部信息
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    
    # 为了绘制更高质量的骨骼，可以先将画布放大，画完再缩放回去
    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1
    canvas = np.zeros(shape=(int(H*sr), int(W*sr), 3), dtype=np.uint8)
    
    # 依次在画布上绘制身体、手部和面部姿态
    canvas = draw_bodypose_aligned(canvas, candidate, subset, score=bodies['score'])
    canvas = draw_handpose_aligned(canvas, hands, pose['hands_score'])
    canvas = draw_facepose_aligned(canvas, faces, pose['faces_score'])

    # 将绘制好的画布缩放回原始尺寸，并进行颜色空间和维度的转换
    resized_canvas = cv2.resize(canvas, (W, H))
    rgb_canvas = cv2.cvtColor(resized_canvas, cv2.COLOR_BGR2RGB)
    
    # 将图像维度从 (H, W, C) 转换为 (C, H, W)，以符合PyTorch等框架的输入要求
    return rgb_canvas.transpose(2, 0, 1)


def get_image_pose(ref_image_path):
    """
    从给定的图像路径中提取姿态信息并生成姿态图。

    Args:
        ref_image_path (str): 输入图像的文件路径。

    Returns:
        np.array: 生成的姿态图数组，格式为 (C, H, W)。
    """
    # 读取图像并转换为RGB格式
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    height, width, _ = ref_image.shape
    
    # 调用核心检测器进行姿态估计
    ref_pose = dwpose_detector_aligned(ref_image)
    
    # 将检测结果绘制成姿态图
    pose_img = draw_pose_aligned(ref_pose, height, width)
    
    return np.array(pose_img)


if __name__ == '__main__':
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser("Training Skeleton Poses Extraction", add_help=True)
    parser.add_argument("--start", type=int, help="指定处理的起始文件夹编号")
    parser.add_argument("--end", type=int, help="指定处理的结束文件夹编号")
    parser.add_argument("--name", type=str, help="指定数据集名称 (如 'rec' 或 'vec')")
    parser.add_argument("--root_path", type=str, help="指定数据集的根目录")
    args = parser.parse_args()

    start = args.start
    end = args.end
    dataset_name = args.name

    # 构建数据集的根路径
    image_root = os.path.join(args.root_path, dataset_name)
    
    # 遍历指定的文件夹编号范围
    for idx in range(start, end + 1):
        # 将数字编号格式化为5位字符串，如 1 -> '00001'
        subfolder = str(idx).zfill(5)
        subfolder_path = os.path.join(image_root, subfolder)
        images_subfolder_path = os.path.join(subfolder_path, "images")
        print(f"正在处理 images 文件夹: {images_subfolder_path}")

        # 构建姿态图的保存路径
        pose_subfolder_path = os.path.join(subfolder_path, "poses")
        # 如果保存路径不存在，则创建
        if not os.path.exists(pose_subfolder_path):
            os.makedirs(pose_subfolder_path)
            print(f"文件夹已创建: {pose_subfolder_path}")
        else:
            print(f"文件夹已存在: {pose_subfolder_path}")
            
        # 遍历 'images' 文件夹下的所有文件
        for root, dirs, files in os.walk(images_subfolder_path):
            for file in files:
                # 只处理.png文件
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    print(file_path)
                    file_name = os.path.splitext(file)[0]
                    image_name = file_name + '.png'
                    image_legal_path = os.path.join(images_subfolder_path, image_name)
                    
                    # 检查对应的姿态图是否已经存在，如果存在则跳过，实现断点续传
                    pose_save_path = os.path.join(pose_subfolder_path, file_name + '.png')
                    if os.path.exists(pose_save_path):
                        print(f"{pose_save_path} 已存在，跳过！")
                        continue
                        
                    # 调用核心函数提取姿态并生成姿态图
                    detected_map = get_image_pose(image_legal_path)
                    
                    # 将图像维度从 (C, H, W) 转回 (H, W, C) 以便cv2.imwrite保存
                    detected_map = np.transpose(detected_map, (1, 2, 0))
                    
                    # 保存生成的姿态图
                    cv2.imwrite(pose_save_path, detected_map)
                    print(f"姿态提取完成: {pose_save_path}")
