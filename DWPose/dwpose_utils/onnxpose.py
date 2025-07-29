# -*- coding: utf-8 -*-
"""
本脚本封装了使用RTMPose ONNX模型进行姿态估计的完整流程。

这是一个"Top-Down"（自顶向下）的方法，它首先需要一个边界框，然后在框内估计姿态。
主要功能包括:
- preprocess: 对检测到的人体边界框区域进行预处理（仿射变换、归一化）。
- inference: 执行ONNX模型推理。
- postprocess: 对模型输出（SimCC热图）进行解码，得到关键点坐标和置信度。
- inference_pose: 串联以上所有步骤，完成从边界框到最终关键点的推理。
"""
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort

def preprocess(
    img: np.ndarray, out_bbox, input_size: Tuple[int, int] = (192, 256)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对RTMPose模型进行预处理。

    Args:
        img (np.ndarray): 原始输入图像。
        out_bbox (np.ndarray): 从检测器获得的人体边界框列表。
        input_size (tuple): 姿态模型的输入尺寸 (w, h)。

    Returns:
        tuple:
        - out_img (list): 经过仿射变换和归一化后的图像块列表，可直接输入模型。
        - out_center (list): 每个框的中心点列表，用于后处理。
        - out_scale (list): 每个框的尺寸列表，用于后处理。
    """
    # 获取图像尺寸
    img_shape = img.shape[:2]
    out_img, out_center, out_scale = [], [], []
    
    # 如果检测器没有返回任何边界框，则将整个图像视为一个边界框
    if len(out_bbox) == 0:
        out_bbox = [[0, 0, img_shape[1], img_shape[0]]]
        
    # 遍历每个检测到的人体边界框
    for i in range(len(out_bbox)):
        x0 = out_bbox[i][0]
        y0 = out_bbox[i][1]
        x1 = out_bbox[i][2]
        y1 = out_bbox[i][3]
        bbox = np.array([x0, y0, x1, y1])

        # 1. 将(x1, y1, x2, y2)格式的边界框转换为(center, scale)格式，并增加1.25倍的padding以包含上下文信息
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)

        # 2. 对边界框区域进行仿射变换，将其裁剪、缩放并warp到模型所需的固定输入尺寸
        resized_img, scale = top_down_affine(input_size, scale, center, img)

        # 3. 对图像进行归一化（减均值，除以标准差）
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        resized_img = (resized_img - mean) / std

        # 保存处理结果
        out_img.append(resized_img)
        out_center.append(center)
        out_scale.append(scale)

    return out_img, out_center, out_scale


def inference(sess: ort.InferenceSession, img: np.ndarray) -> np.ndarray:
    """
    执行RTMPose模型推理。

    Args:
        sess (ort.InferenceSession): ONNXRuntime的推理会话。
        img (np.ndarray): 经过预处理的图像块列表。

    Returns:
        outputs (np.ndarray): 模型的原始输出列表。
    """
    all_out = []
    # 遍历每个预处理过的人体图像块
    for i in range(len(img)):
        # 构建输入，需要将图像维度从(H, W, C)转换为(C, H, W)
        input_data = [img[i].transpose(2, 0, 1)]

        # 准备ONNX Runtime的输入字典和输出名称列表
        sess_input = {sess.get_inputs()[0].name: input_data}
        sess_output_names = []
        for out in sess.get_outputs():
            sess_output_names.append(out.name)

        # 执行模型推理
        outputs = sess.run(sess_output_names, sess_input)
        all_out.append(outputs)

    return all_out


def postprocess(outputs: List[np.ndarray],
                model_input_size: Tuple[int, int],
                center: Tuple[int, int],
                scale: Tuple[int, int],
                simcc_split_ratio: float = 2.0
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    对RTMPose模型的输出进行后处理。

    Args:
        outputs (np.ndarray): 模型的原始输出列表（SimCC热图）。
        model_input_size (tuple): 模型输入尺寸(w, h)。
        center (tuple): 原始边界框的中心点。
        scale (tuple): 原始边界框的尺寸。
        simcc_split_ratio (float): SimCC解码时的缩放因子。

    Returns:
        tuple:
        - keypoints (np.ndarray): 在原始图像坐标系下的关键点坐标。
        - scores (np.ndarray): 对应关键点的置信度分数。
    """
    all_key = []
    all_score = []
    # 遍历每个人的推理结果
    for i in range(len(outputs)):
        # 1. 解码SimCC：模型输出是x和y轴上的概率分布（热图），解码过程是找到概率最高的位置
        simcc_x, simcc_y = outputs[i]
        keypoints, scores = decode(simcc_x, simcc_y, simcc_split_ratio)

        # 2. 坐标逆变换：将从模型输出空间得到的坐标，通过仿射变换的逆运算，映射回原始图像的坐标系
        # 公式: 原始坐标 = (模型空间坐标 / 模型尺寸) * 原始bbox尺寸 + 原始bbox左上角坐标
        keypoints = keypoints / model_input_size * scale[i] + center[i] - scale[i] / 2
        all_key.append(keypoints[0])
        all_score.append(scores[0])

    return np.array(all_key), np.array(all_score)


def bbox_xyxy2cs(bbox: np.ndarray,
                 padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    """
    将(x1, y1, x2, y2)格式的边界框转换为(center, scale)格式。

    Args:
        bbox (ndarray): 边界框，格式为(左, 上, 右, 下)。
        padding (float): 缩放填充因子，>1会使边界框扩大。

    Returns:
        tuple:
        - center (np.ndarray[float32]): 边界框中心点(x, y)。
        - scale (np.ndarray[float32]): 边界框尺寸(w, h)。
    """
    # 兼容单个边界框和多个边界框的情况
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]

    # 计算中心点和尺寸
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding

    if dim == 1:
        center = center[0]
        scale = scale[0]

    return center, scale


def _fix_aspect_ratio(bbox_scale: np.ndarray,
                      aspect_ratio: float) -> np.ndarray:
    """
    将边界框尺寸(scale)调整为指定的长宽比。

    Args:
        bbox_scale (np.ndarray): 原始的边界框尺寸(w, h)。
        aspect_ratio (float): 目标长宽比 (w/h)。

    Returns:
        np.ndarray: 调整后的边界框尺寸。
    """
    w, h = np.hsplit(bbox_scale, [1])
    # 如果当前w/h > 目标w/h，则以w为基准，调整h
    # 否则以h为基准，调整w
    bbox_scale = np.where(w > h * aspect_ratio,
                          np.hstack([w, w / aspect_ratio]),
                          np.hstack([h * aspect_ratio, h]))
    return bbox_scale


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    使用旋转矩阵旋转一个2D点。

    Args:
        pt (np.ndarray): 2D点坐标(x, y)。
        angle_rad (float): 旋转角度（弧度）。

    Returns:
        np.ndarray: 旋转后的点。
    """
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    计算仿射变换所需的第三个点。
    通过将向量(a-b)绕b点逆时针旋转90度得到。
    """
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c


def get_warp_matrix(center: np.ndarray,
                    scale: np.ndarray,
                    rot: float,
                    output_size: Tuple[int, int],
                    shift: Tuple[float, float] = (0., 0.),
                    inv: bool = False) -> np.ndarray:
    """
    计算将输入图像中的边界框区域映射到输出尺寸的仿射变换矩阵。

    Args:
        center (np.ndarray[2, ]): 边界框中心(x, y)。
        scale (np.ndarray[2, ]): 边界框尺寸(w, h)。
        rot (float): 旋转角度（度）。
        output_size (np.ndarray[2, ]): 目标输出尺寸(w, h)。
        shift (tuple): 平移比例。
        inv (bool): 是否计算逆矩阵。

    Returns:
        np.ndarray: 2x3的仿射变换矩阵。
    """
    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    # 计算变换矩阵
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0., src_w * -0.5]), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    # 获取源矩形的三个角点
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    # 获取目标矩形的三个角点
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    # 使用OpenCV计算仿射矩阵
    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return warp_mat


def top_down_affine(input_size: tuple, bbox_scale: np.ndarray, bbox_center: np.ndarray,
                    img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    通过仿射变换，从原图中裁剪并缩放bbox区域，作为模型输入。

    Args:
        input_size (tuple): 模型输入尺寸(w, h)。
        bbox_scale (np.ndarray): 边界框尺寸。
        bbox_center (np.ndarray): 边界框中心。
        img (np.ndarray): 原始图像。

    Returns:
        tuple:
        - np.ndarray[float32]: 经过仿射变换后的图像。
        - np.ndarray[float32]: 调整长宽比后的边界框尺寸。
    """
    w, h = input_size
    warp_size = (int(w), int(h))

    # 1. 调整bbox尺寸以匹配模型的输入长宽比
    bbox_scale = _fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)

    # 2. 获取仿射变换矩阵
    center = bbox_center
    scale = bbox_scale
    rot = 0
    warp_mat = get_warp_matrix(center, scale, rot, output_size=(w, h))

    # 3. 执行仿射变换
    img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)

    return img, bbox_scale


def get_simcc_maximum(simcc_x: np.ndarray,
                      simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    从SimCC表示中找到响应最大的位置和值。

    Args:
        simcc_x (np.ndarray): x轴上的SimCC概率分布。
        simcc_y (np.ndarray): y轴上的SimCC概率分布。

    Returns:
        tuple:
        - locs (np.ndarray): 响应最大位置的坐标 (N, K, 2)。
        - vals (np.ndarray): 响应的最大值（置信度） (N, K)。
    """
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    # 在x和y轴上分别找到最大值的索引（即坐标）
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    
    # 找到最大值本身作为置信度
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    # 取x和y轴上较小的那个最大值作为最终的置信度
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1  # 置信度为0的点标记为无效

    # Reshape回(N, K, ...)的格式
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)

    return locs, vals


def decode(simcc_x: np.ndarray, simcc_y: np.ndarray,
           simcc_split_ratio) -> Tuple[np.ndarray, np.ndarray]:
    """
    解码SimCC，得到关键点坐标和分数。

    Args:
        simcc_x (np.ndarray): x轴上的SimCC。
        simcc_y (np.ndarray): y轴上的SimCC。
        simcc_split_ratio (int): 缩放因子。

    Returns:
        tuple:
        - keypoints (np.ndarray): 关键点坐标。
        - scores (np.ndarray): 关键点分数。
    """
    # 找到最大响应位置
    keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)
    # 应用缩放因子
    keypoints /= simcc_split_ratio

    return keypoints, scores


def inference_pose(session, out_bbox, oriImg):
    """
    执行完整的姿态估计推理流程。

    Args:
        session (ort.InferenceSession): ONNX Runtime的姿态模型会话。
        out_bbox (np.array): 人体边界框列表。
        oriImg (np.array): 原始输入图像。

    Returns:
        tuple:
        - keypoints (np.array): 关键点坐标。
        - scores (np.array): 关键点置信度。
    """
    # 获取模型的输入尺寸
    h, w = session.get_inputs()[0].shape[2:]
    model_input_size = (w, h)
    
    # 1. 预处理
    resized_img, center, scale = preprocess(oriImg, out_bbox, model_input_size)
    # 2. 推理
    outputs = inference(session, resized_img)
    # 3. 后处理
    keypoints, scores = postprocess(outputs, model_input_size, center, scale)

    return keypoints, scores

