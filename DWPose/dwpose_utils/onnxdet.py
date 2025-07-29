# -*- coding: utf-8 -*-
"""
本脚本封装了使用YOLOX ONNX模型进行目标检测的完整流程。

主要功能包括:
- preprocess: 对输入图像进行预处理（缩放、填充）。
- demo_postprocess: 对模型原始输出进行解码。
- nms/multiclass_nms: 实现非极大值抑制（NMS）以消除重叠的边界框。
- inference_detector: 串联以上所有步骤，完成从输入图像到最终人体边界框的推理。
"""
import cv2
import numpy as np
import onnxruntime

def nms(boxes, scores, nms_thr):
    """单类别非极大值抑制 (NMS) 的Numpy实现。"""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # 按置信度降序排序

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i) # 保留置信度最高的框
        # 计算剩余框与当前框的交并比 (IoU)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留IoU小于阈值的框
        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """多类别非极大值抑制的Numpy实现。"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes): # 对每个类别分别执行NMS
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def demo_postprocess(outputs, img_size, p6=False):
    """对YOLOX模型的原始输出进行后处理，解码成边界框坐标。"""
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    # 将模型输出的相对坐标(dx, dy)和尺寸(dw, dh)转换成绝对坐标
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs

def preprocess(img, input_size, swap=(2, 0, 1)):
    """
    对输入图像进行预处理，以满足YOLOX模型的输入要求。
    - 保持长宽比缩放
    - 填充到指定的input_size
    - 维度转换 (H,W,C) -> (C,H,W)
    """
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def inference_detector(session, oriImg):
    """
    执行完整的人体检测推理流程。

    Args:
        session (ort.InferenceSession): ONNX Runtime的检测模型会话。
        oriImg (np.array): 原始输入图像。

    Returns:
        np.array: 检测到的人体边界框列表，格式为 (N, 4)，N是人数。
    """
    input_shape = (640, 640) # YOLOX模型输入尺寸
    # 1. 图像预处理
    img, ratio = preprocess(oriImg, input_shape)

    # 2. ONNX模型推理
    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    
    # 3. 模型输出后处理
    predictions = demo_postprocess(output[0], input_shape)[0]

    # 4. 解码成(x1, y1, x2, y2)格式的边界框
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio # 将坐标缩放回原始图像尺寸
    
    # 5. NMS 和类别过滤
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        # 过滤条件：置信度 > 0.3 并且 类别为0 ('person')
        isscore = final_scores > 0.3
        iscat = final_cls_inds == 0
        isbbox = [ i and j for (i, j) in zip(isscore, iscat)]
        final_boxes = final_boxes[isbbox]
    else:
        final_boxes = np.array([])

    return final_boxes
