# -*- coding: utf-8 -*-
"""
本脚本用于为StableAnimator项目的**推理/评估**数据进行预处理。

工作流程:
1. 接收一个目标图像序列文件夹、一张参考图像和一个输出文件夹作为输入。
2. 对参考图像进行姿态检测，获取其身体比例作为基准。
3. 对目标图像序列中的每一帧进行姿态检测。
4. **核心步骤**: 使用线性回归（polyfit）计算一个仿射变换，将目标序列中所有帧的姿态
   对齐到参考图像的姿态上。这可以校正身体比例，减少姿态抖动。
5. 将对齐后的姿态关键点绘制成姿态图。
6. 将生成的姿态图序列保存到指定的输出文件夹中。
"""
import math
import matplotlib
import cv2
import os
import numpy as np
from dwpose_utils.dwpose_detector import dwpose_detector_aligned
import argparse

eps = 0.01

def alpha_blend_color(color, alpha):
    """根据置信度混合颜色，实现半透明效果。"""
    return [int(c * alpha) for c in color]

def draw_bodypose(canvas, candidate, subset, score):
    """在画布上绘制身体姿态的骨骼和关节点。"""
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            conf = score[n][np.array(limbSeq[i]) - 1]
            if conf[0] < 0.3 or conf[1] < 0.3:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, alpha_blend_color(colors[i], conf[0] * conf[1]))

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            conf = score[n][i]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, alpha_blend_color(colors[i], conf), thickness=-1)
    return canvas

def draw_handpose(canvas, all_hand_peaks, all_hand_scores):
    """在画布上绘制手部姿态。"""
    H, W, C = canvas.shape
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    for peaks, scores in zip(all_hand_peaks, all_hand_scores):
        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1, y1 = int(x1 * W), int(y1 * H)
            x2, y2 = int(x2 * W), int(y2 * H)
            score = int(scores[e[0]] * scores[e[1]] * 255)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                color = matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * score
                cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=2)
        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x, y = int(x * W), int(y * H)
            score = int(scores[i] * 255)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, score), thickness=-1)
    return canvas

def draw_facepose(canvas, all_lmks, all_scores):
    """在画布上绘制面部关键点。"""
    H, W, C = canvas.shape
    for lmks, scores in zip(all_lmks, all_scores):
        for lmk, score in zip(lmks, scores):
            x, y = lmk
            x, y = int(x * W), int(y * H)
            conf = int(score * 255)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (conf, conf, conf), thickness=-1)
    return canvas

def draw_pose(pose, H, W, ref_w=2160):
    """将身体、手部和面部姿态统一绘制到一张画布上。"""
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1
    canvas = np.zeros(shape=(int(H*sr), int(W*sr), 3), dtype=np.uint8)
    canvas = draw_bodypose(canvas, candidate, subset, score=bodies['score'])
    canvas = draw_handpose(canvas, hands, pose['hands_score'])
    canvas = draw_facepose(canvas, faces, pose['faces_score'])
    resized_canvas = cv2.resize(canvas, (W, H))
    rgb_canvas = cv2.cvtColor(resized_canvas, cv2.COLOR_BGR2RGB)
    return rgb_canvas.transpose(2, 0, 1)

def get_video_pose(video_path, ref_image_path, poses_folder_path=None):
    """
    从图像序列中提取姿态，并将其对齐到参考图像的姿态。

    Args:
        video_path (str): 包含目标图像帧的文件夹路径。
        ref_image_path (str): 参考图像的路径。
        poses_folder_path (str, optional): 保存姿态图的文件夹路径。

    Returns:
        np.array: 对齐并绘制好的姿态图序列，形状为 (F, C, H, W)。
    """
    # 1. 读取参考图并提取其姿态作为基准
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    height, width, _ = ref_image.shape
    ref_pose = dwpose_detector_aligned(ref_image)
    
    # 定义用于对齐的身体关键点ID
    ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    # 过滤掉参考图中未检测到的关键点
    ref_keypoint_id = [i for i in ref_keypoint_id \
        if len(ref_pose['bodies']['subset']) > 0 and ref_pose['bodies']['subset'][0][i] >= .0]
    ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]

    os.makedirs(poses_folder_path, exist_ok=True)
    
    # 2. 读取目标图像序列并提取所有帧的姿态
    detected_poses = []
    files = os.listdir(video_path)
    png_files = [f for f in files if f.endswith('.png')]
    # 按文件名中的数字排序，确保帧序正确
    png_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    for sub_name in png_files:
        sub_driven_image_path = os.path.join(video_path, sub_name)
        driven_image = cv2.imread(sub_driven_image_path)
        driven_image = cv2.cvtColor(driven_image, cv2.COLOR_BGR2RGB)
        driven_pose = dwpose_detector_aligned(driven_image)
        detected_poses.append(driven_pose)

    # 3. 姿态对齐
    # 提取所有检测到的身体姿态
    detected_bodies = np.stack(
        [p['bodies']['candidate'] for p in detected_poses if p['bodies']['candidate'].shape[0] == 18])[:,
                      ref_keypoint_id]
    # 使用线性回归（1阶多项式拟合）找到从'检测到的姿态'到'参考姿态'的最佳映射关系
    # ay, by 是y坐标的缩放系数和偏移量
    ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(), np.tile(ref_body[:, 1], len(detected_bodies)), 1)
    fh, fw = height, width
    # ax, bx 是x坐标的缩放系数和偏移量
    ax = ay / (fh / fw / height * width) # 保持长宽比
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
    
    a = np.array([ax, ay]) # 缩放系数 [ax, ay]
    b = np.array([bx, by]) # 偏移量 [bx, by]
    
    # 4. 应用对齐变换并绘制
    output_pose = []
    for detected_pose in detected_poses:
        # 将计算出的缩放和偏移应用到身体、面部和手部的所有关键点上
        detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
        detected_pose['faces'] = detected_pose['faces'] * a + b
        detected_pose['hands'] = detected_pose['hands'] * a + b
        # 绘制对齐后的姿态
        im = draw_pose(detected_pose, height, width)
        output_pose.append(np.array(im))
        
    return np.stack(output_pose)

def get_image_pose(ref_image_path):
    """从单张图片提取姿态并绘制，不进行对齐。"""
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    height, width, _ = ref_image.shape
    ref_pose = dwpose_detector_aligned(ref_image)
    pose_img = draw_pose(ref_pose, height, width)
    return np.array(pose_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从图像中提取骨骼姿态。")
    parser.add_argument('--target_image_folder_path', type=str, required=True, help='包含目标图像的文件夹路径。')
    parser.add_argument('--ref_image_path', type=str, required=True, help='参考图像的路径。')
    parser.add_argument('--poses_folder_path', type=str, required=True, help='保存提取姿态的文件夹路径。')
    args = parser.parse_args()

    video_path = args.target_image_folder_path
    ref_image_path = args.ref_image_path
    poses_folder_path = args.poses_folder_path
    
    # 获取对齐后的姿态图序列
    detected_maps = get_video_pose(video_path, ref_image_path, poses_folder_path=poses_folder_path)
    
    # 逐帧保存姿态图
    for i in range(detected_maps.shape[0]):
        # 将(C,H,W)转为(H,W,C)以供cv2保存
        pose_image = np.transpose(detected_maps[i], (1, 2, 0))
        pose_save_path = os.path.join(poses_folder_path, f"frame_{i}.png")
        cv2.imwrite(pose_save_path, pose_image)
        print(f"已保存姿态图: {pose_save_path}")
