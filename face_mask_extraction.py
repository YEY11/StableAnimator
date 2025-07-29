# -*- coding: utf-8 -*-
"""
本脚本是 StableAnimator 项目的一个关键预处理工具，用于从图像中提取人脸蒙版(mask)。

核心功能:
1. 接收一个包含多张图像的文件夹路径作为输入。
2. 对文件夹中的每一张图像进行人脸检测。
3. 采用双重检测策略以提高成功率：
   - 首先使用强大的 insightface.FaceAnalysis (antelopev2 模型)进行检测。
   - 如果主检测器失败，则使用 facexlib.FaceRestoreHelper (retinaface 模型)作为备用检测器。
4. 如果两种检测器都找不到人脸，则生成一个全白的蒙版。
5. 将检测到的人脸区域（以边界框形式）绘制成一个白色的实心矩形，背景为黑色，生成二值蒙版图。
6. 将生成的蒙版图保存到与输入图像文件夹同级的 "faces" 子文件夹中。

这些蒙版在后续的 HJB-based Face Optimization (基于HJB方程的人脸优化) 步骤中至关重要，
它们告诉模型应该在图像的哪个区域集中进行人脸细节的优化。
"""
import numpy as np
import torch
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from insightface.app import FaceAnalysis
import cv2
import argparse
import os

def get_face_masks(image_path, save_path, app, face_helper, height=904, width=512):
    """
    对单张图片执行人脸检测，并生成和保存对应的蒙版图。

    Args:
        image_path (str): 输入图像的文件路径。
        save_path (str): 生成蒙版的保存路径。
        app (FaceAnalysis): insightface 的人脸分析器实例（主检测器）。
        face_helper (FaceRestoreHelper): facexlib 的人脸辅助工具实例（备用检测器）。
        height (int, optional): 图像高度（注：此参数在函数内部会被实际图像尺寸覆盖）。
        width (int, optional): 图像宽度（注：此参数在函数内部会被实际图像尺寸覆盖）。
    """
    # 1. 读取图像并获取其实际尺寸
    image_1 = cv2.imread(image_path)
    height, width = image_1.shape[:2]
    image_bgr_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2BGR) # insightface 需要 BGR 格式

    # 2. 创建一个与原图等大的全黑画布，用于绘制蒙版
    mask_1 = np.zeros((height, width), dtype=np.uint8)

    # --- 双重检测策略开始 ---
    # 策略一：使用 insightface (antelopev2) 主检测器
    image_info_1 = app.get(image_bgr_1)

    if len(image_info_1) > 0:
        print("检测成功 (使用 FaceAnalysis 主检测器)")
        # 遍历所有检测到的人脸
        for info in image_info_1:
            # 获取人脸边界框坐标
            x_1, y_1, x_2, y_2 = info['bbox']
            # 在黑色画布上，将人脸区域绘制成一个实心白色矩形
            cv2.rectangle(mask_1, (int(x_1), int(y_1)), (int(x_2), int(y_2)), (255), thickness=cv2.FILLED)
        cv2.imwrite(save_path, mask_1)
    else:
        # 策略二：如果主检测器失败，使用 facexlib (retinaface) 备用检测器
        face_helper.clean_all() # 清理上一次的检测结果
        with torch.no_grad():
            bboxes = face_helper.face_det.detect_faces(image_bgr_1, 0.97) # 设定一个较高的置信度阈值
        
        if len(bboxes) > 0:
            print("检测成功 (使用 FaceRestoreHelper 备用检测器)")
            for bbox in bboxes:
                # 同样地，将人脸区域绘制成白色矩形
                cv2.rectangle(mask_1, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255), thickness=cv2.FILLED)
            cv2.imwrite(save_path, mask_1)
        else:
            # 策略三：如果两种检测器都失败，生成一个全白的蒙版
            print("未检测到人脸，生成全白蒙版。")
            mask_1[:] = 255 # 将整个画布填充为白色
            cv2.imwrite(save_path, mask_1)


if __name__ == "__main__":
    # --- 1. 初始化与设置 ---
    parser = argparse.ArgumentParser("Human Face Mask Extraction", add_help=True)
    parser.add_argument("--image_folder", type=str, help="指定一个包含图像的文件夹路径")
    args = parser.parse_args()

    image_folder = args.image_folder

    # 初始化主检测器：InsightFace FaceAnalysis (antelopev2模型)
    # 这是一个非常强大和常用的人脸分析库
    print("正在初始化人脸检测模型...")
    app = FaceAnalysis(
        name='antelopev2', root='.', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    # 初始化备用检测器：FaceXLib FaceRestoreHelper (retinaface模型)
    # FaceXLib 通常用于人脸修复，但其内部也包含了人脸检测模块
    face_helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        device="cuda",
    )
    # 注意: 下面这行代码初始化了人脸解析(parsing)模型，用于分割五官。
    # 但在本脚本的 get_face_masks 函数中并未使用，可能为预留功能或其他模块使用。
    face_helper.face_parse = init_parsing_model(model_name='bisenet', device="cuda")
    print("模型初始化完成。")

    # --- 2. 文件处理循环 ---
    print(f"开始处理图像文件夹: {image_folder}")
    # 构建输出文件夹路径 (e.g., ../your_case/images -> ../your_case/faces)
    face_subfolder_path = os.path.join(os.path.dirname(image_folder), "faces")
    
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(face_subfolder_path):
        os.makedirs(face_subfolder_path)
        print(f"文件夹已创建: {face_subfolder_path}")
    else:
        print(f"文件夹已存在: {face_subfolder_path}")
    
    # 遍历输入文件夹中的所有.png文件
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                print(f"正在处理: {file_path}")
                file_name = os.path.splitext(file)[0]
                
                # 检查对应的蒙版文件是否已存在，如果存在则跳过，实现断点续处理
                face_save_path = os.path.join(face_subfolder_path, file_name + '.png')
                if os.path.exists(face_save_path):
                    print(f"{face_save_path} 已存在，跳过！")
                    continue

                # 调用核心函数进行人脸蒙版提取和保存
                get_face_masks(image_path=file_path, save_path=face_save_path, app=app, face_helper=face_helper)
                print(f"蒙版提取完成: {face_save_path}")
