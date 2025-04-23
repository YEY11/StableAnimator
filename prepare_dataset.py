#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
import glob
import shutil
import time
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import re
from loguru import logger

# 配置 loguru 日志
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add("./logs/prepare_dataset.log", rotation="10 MB", retention="1 week")


def get_video_resolution(video_path):
    """获取视频分辨率"""
    try:
        # 使用 ffprobe 获取视频分辨率
        cmd = [
            "ffprobe", 
            "-v", "error", 
            "-select_streams", "v:0", 
            "-show_entries", "stream=width,height", 
            "-of", "csv=s=x:p=0", 
            video_path
        ]
        result = subprocess.check_output(cmd).decode('utf-8').strip()
        width, height = map(int, result.split('x'))
        return width, height
    except Exception as e:
        # 如果 ffprobe 失败，尝试使用 OpenCV
        try:
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                raise Exception("无法打开视频文件")
                
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video.release()
            
            if width == 0 or height == 0:
                raise Exception("无法获取有效分辨率")
                
            return width, height
        except Exception as cv_error:
            logger.error(f"获取视频 {video_path} 分辨率失败 (ffprobe 和 OpenCV 都失败): {str(e)} | {str(cv_error)}")
            return 0, 0


def extract_frames(video_path, output_dir):
    """从视频中提取帧"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        # 尝试使用 ffmpeg 提取帧
        try:
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-q:v", "1",
                "-start_number", "0",
                f"{output_dir}/frame_%d.png"
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except Exception as ffmpeg_error:
            logger.warning(f"ffmpeg 提取帧失败，尝试使用 OpenCV: {str(ffmpeg_error)}")
            
            # 如果 ffmpeg 失败，尝试使用 OpenCV
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                raise Exception("无法打开视频文件")
            
            frame_count = 0
            while True:
                success, frame = video.read()
                if not success:
                    break
                
                output_path = os.path.join(output_dir, f"frame_{frame_count}.png")
                cv2.imwrite(output_path, frame)
                frame_count += 1
            
            video.release()
            
            if frame_count == 0:
                raise Exception("未能提取任何帧")
                
            return True
    except Exception as e:
        logger.error(f"提取帧失败 {video_path}: {str(e)}")
        return False


def extract_face_masks(stableanimator_dir, image_folder, video_dir):
    """提取人脸遮罩"""
    try:
        current_dir = os.getcwd()
        os.chdir(stableanimator_dir)
        
        # 创建保存人脸遮罩的目录
        faces_dir = os.path.join(video_dir, "faces")
        os.makedirs(faces_dir, exist_ok=True)
        
        cmd = [
            "python", "face_mask_extraction.py",
            f"--image_folder={image_folder}"
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 回到原目录
        os.chdir(current_dir)
        
        # 检查是否生成了人脸遮罩
        mask_count = len(glob.glob(f"{faces_dir}/*.png"))
        if mask_count == 0:
            # 检查多个可能的位置
            video_name = os.path.basename(video_dir)
            video_parent = os.path.basename(os.path.dirname(video_dir))
            
            possible_paths = [
                os.path.join(stableanimator_dir, "inference", video_name, "faces"),
                os.path.join(stableanimator_dir, "inference", video_parent, video_name, "faces"),
                os.path.join(stableanimator_dir, "inference", video_parent, "faces"),
                os.path.join(stableanimator_dir, "inference", "faces")
            ]
            
            found = False
            for faces_src in possible_paths:
                if os.path.exists(faces_src) and len(glob.glob(f"{faces_src}/*.png")) > 0:
                    logger.info(f"找到人脸遮罩目录: {faces_src}")
                    for face_file in glob.glob(f"{faces_src}/*.png"):
                        shutil.copy(face_file, faces_dir)
                    found = True
                    break
            
            if not found:
                logger.warning(f"未能为 {video_dir} 生成人脸遮罩")
                return False
        
        logger.success(f"成功为 {video_dir} 提取人脸遮罩")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"提取人脸遮罩失败 {image_folder}: {e.stderr.decode('utf-8')}")
        return False
    except Exception as e:
        logger.error(f"提取人脸遮罩时出错 {image_folder}: {str(e)}")
        # 确保回到原目录
        os.chdir(current_dir)
        return False


def extract_poses_for_video(stableanimator_dir, video_dir):
    """为单个视频提取人体姿态骨架"""
    try:
        # 检查模型文件是否存在
        dwpose_dir = os.path.join(stableanimator_dir, "checkpoints", "DWPose")
        yolox_model = os.path.join(dwpose_dir, "yolox_l.onnx")
        dwpose_model = os.path.join(dwpose_dir, "dw-ll_ucoco_384.onnx")
        
        if not os.path.exists(yolox_model) or not os.path.exists(dwpose_model):
            logger.warning(f"警告: DWPose 模型文件不存在，请确保以下文件存在:")
            logger.warning(f"  - {yolox_model}")
            logger.warning(f"  - {dwpose_model}")
            logger.warning(f"请执行以下命令下载模型文件:")
            logger.warning(f"  mkdir -p {dwpose_dir}")
            logger.warning(f"  wget -O {yolox_model} https://hf-mirror.com/yzd-v/DWPose/resolve/main/yolox_l.onnx")
            logger.warning(f"  wget -O {dwpose_model} https://hf-mirror.com/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx")
            return False
        
        # 检查图像目录
        images_dir = os.path.join(video_dir, "images")
        poses_dir = os.path.join(video_dir, "poses")
        
        if not os.path.exists(images_dir):
            logger.warning(f"警告: {video_dir} 中没有找到图像目录")
            return False
        
        os.makedirs(poses_dir, exist_ok=True)
        
        # 获取视频的类别和编号
        category = os.path.basename(os.path.dirname(video_dir))
        video_number = int(os.path.basename(video_dir))
        
        # 执行姿态提取
        current_dir = os.getcwd()
        os.chdir(stableanimator_dir)
        
        # 使用 training_skeleton_extraction.py 代替 extract_video_keypoints.py
        cmd = [
            "python", "DWPose/training_skeleton_extraction.py",
            f"--root_path={os.path.dirname(os.path.dirname(video_dir))}",
            f"--name={category}",
            f"--start={video_number}",
            f"--end={video_number}"
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # 回到原目录
        os.chdir(current_dir)
        
        # 检查是否成功生成骨架
        pose_files = glob.glob(f"{poses_dir}/*.png")
        if len(pose_files) == 0:
            logger.warning(f"未能为 {video_dir} 生成姿态骨架")
            return False
        
        logger.success(f"成功为 {video_dir} 提取姿态骨架")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"提取姿态骨架失败 {video_dir}: {e.stderr.decode('utf-8')}")
        return False
    except Exception as e:
        logger.error(f"提取姿态骨架时出错 {video_dir}: {str(e)}")
        # 确保回到原目录
        try:
            os.chdir(current_dir)
        except:
            pass
        return False


def update_video_mapping_csv(target_dir, category, folder_num, video_path):
    """更新视频映射 CSV 文件
    
    Args:
        target_dir: 目标目录基础路径
        category: 分类 ('rec' 或 'vec')
        folder_num: 文件夹编号
        video_path: 原始视频路径
    """
    # CSV 文件路径
    csv_path = os.path.join(target_dir, category, f"{category}_video_mapping.csv")
    
    # 获取原始视频文件名
    video_name = os.path.basename(video_path)
    
    # 创建或追加到 CSV 文件
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a') as f:
        # 如果文件不存在，添加标题行
        if not file_exists:
            f.write("folder_name,original_video_name\n")
        
        # 写入映射关系
        f.write(f"{folder_num},{video_name}\n")
    
    logger.info(f"已更新视频映射: {folder_num} -> {video_name}")


def process_video(args, video_path, rec_count, vec_count, extract_mask=True, extract_pose=True, forced_type=None):
    """处理单个视频"""
    video_name = os.path.basename(video_path).rsplit('.', 1)[0]
    logger.info(f"处理视频: {video_name}")
    
    # 获取视频分辨率
    width, height = get_video_resolution(video_path)
    if width == 0 or height == 0:
        logger.warning(f"跳过视频 {video_path} (无法获取分辨率)")
        return None, None
    
    # 确定视频类型 (rec 或 vec)
    if forced_type:
        # 使用命令行参数强制设置的类型
        target_type = forced_type
        is_rec = (forced_type == "rec")
        logger.info(f"强制使用视频类型: {target_type}")
    else:
        # 基于分辨率自动确定类型
        if width <= 512 and height <= 512:
            # 适合 rec (512x512)
            target_type = "rec"
            is_rec = True
        else:
            # 适合 vec (576x1024)
            target_type = "vec"
            is_rec = False
    
    # 设置文件夹编号
    if is_rec:
        folder_num = f"{rec_count:05d}"
    else:
        folder_num = f"{vec_count:05d}"
    
    # 创建目标目录 - 确保每个视频有唯一的目录
    video_dir = os.path.join(args.target_dir, target_type, folder_num)
    
    # 检查目录是否已存在，如果存在则递增编号
    original_dir = video_dir
    counter = 1
    while os.path.exists(video_dir) and os.path.basename(video_dir) != "raw_videos":
        folder_num = f"{int(folder_num) + 1:05d}"
        video_dir = os.path.join(args.target_dir, target_type, folder_num)
        counter += 1
        if counter > 1000:  # 防止无限循环
            logger.error(f"错误: 无法为 {video_path} 创建唯一目录")
            return None, None
    
    images_dir = os.path.join(video_dir, "images")
    
    os.makedirs(images_dir, exist_ok=True)
    
    logger.info(f"创建目录: {video_dir}")
    
    # 提取帧
    logger.info(f"从视频中提取帧... {video_path}")
    if not extract_frames(video_path, images_dir):
        logger.error(f"跳过视频 {video_path} (帧提取失败)")
        # 清理空目录
        try:
            shutil.rmtree(video_dir)
        except:
            pass
        return None, None
    
    # 检查是否成功提取了足够的帧
    frame_count = len(glob.glob(f"{images_dir}/*.png"))
    if frame_count < 5:  # 设置一个最小帧数阈值
        logger.warning(f"跳过视频 {video_path} (提取的帧数不足: {frame_count})")
        # 清理目录
        try:
            shutil.rmtree(video_dir)
        except:
            pass
        return None, None
    
    # 提取人脸遮罩
    if extract_mask:
        logger.info(f"提取人脸遮罩... {images_dir}")
        if not extract_face_masks(args.stableanimator_dir, images_dir, video_dir):
            logger.warning(f"人脸遮罩提取失败: {video_dir}")
    
    # 提取姿态骨架
    if extract_pose and not args.skip_pose:
        logger.info(f"提取姿态骨架... {video_dir}")
        if not extract_poses_for_video(args.stableanimator_dir, video_dir):
            logger.warning(f"姿态骨架提取失败: {video_dir}")
    
    logger.success(f"已完成 {video_path} 的处理")
    
    # 更新视频映射 CSV 文件
    update_video_mapping_csv(args.target_dir, target_type, folder_num, video_path)
    
    if is_rec:
        return video_dir, None
    else:
        return None, video_dir


def batch_process_videos(args, video_files, extract_mask=True, extract_pose=True):
    """处理多个视频文件"""
    # 获取已存在的目录
    existing_rec_dirs = sorted(glob.glob(os.path.join(args.target_dir, "rec", "*")))
    existing_vec_dirs = sorted(glob.glob(os.path.join(args.target_dir, "vec", "*")))
    
    # 过滤掉 raw_videos 目录
    existing_rec_dirs = [d for d in existing_rec_dirs if os.path.basename(d) != "raw_videos"]
    existing_vec_dirs = [d for d in existing_vec_dirs if os.path.basename(d) != "raw_videos"]
    
    # 处理计数器 - 从现有目录数量开始
    rec_count = len(existing_rec_dirs) + 1
    vec_count = len(existing_vec_dirs) + 1
    
    # 存储处理结果
    rec_dirs = []
    vec_dirs = []
    
    # 顺序处理视频，每个视频处理完成后保存结果
    for i, video_file in enumerate(tqdm(video_files, desc="处理视频")):
        # 根据命令行参数强制设置视频类型
        forced_type = None
        if args.video_resolution == 'rec':
            forced_type = 'rec'
        elif args.video_resolution == 'vec':
            forced_type = 'vec'
        
        rec_dir, vec_dir = process_video(
            args, video_file, rec_count + len(rec_dirs), vec_count + len(vec_dirs),
            extract_mask=extract_mask, extract_pose=extract_pose, forced_type=forced_type
        )
        
        if rec_dir:
            rec_dirs.append(rec_dir)
            # 更新路径文件以保存进度
            update_path_file(args.target_dir, "rec", existing_rec_dirs + rec_dirs)
        
        if vec_dir:
            vec_dirs.append(vec_dir)
            # 更新路径文件以保存进度
            update_path_file(args.target_dir, "vec", existing_vec_dirs + vec_dirs)
    
    # 合并现有目录和新处理的目录
    all_rec_dirs = existing_rec_dirs + rec_dirs
    all_vec_dirs = existing_vec_dirs + vec_dirs
    
    # 最终更新一次路径文件
    update_path_file(args.target_dir, "rec", all_rec_dirs)
    update_path_file(args.target_dir, "vec", all_vec_dirs)
    
    return all_rec_dirs, all_vec_dirs, rec_dirs, vec_dirs


def update_path_file(target_dir, category, dirs):
    """更新路径文件以保存进度"""
    path_file = os.path.join(target_dir, f"video_{category}_path.txt")
    with open(path_file, 'w') as f:
        for dir_path in dirs:
            f.write(f"{dir_path}\n")


def extract_all_frames(args):
    """仅提取所有视频的帧"""
    # 获取视频文件列表
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    video_files = []
    
    if args.raw_videos_dir:
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(args.raw_videos_dir, f"**/*{ext}"), recursive=True))
    else:
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(args.source_dir, f"**/*{ext}"), recursive=True))
    
    # 按视频名称排序
    video_files.sort(key=lambda x: os.path.basename(x))
    
    total_videos = len(video_files)
    logger.info(f"找到 {total_videos} 个视频文件")
    
    if total_videos == 0:
        logger.warning("没有找到视频文件，退出")
        return False
    
    # 处理视频，只提取帧，不提取人脸遮罩和姿态骨架
    all_rec_dirs, all_vec_dirs, rec_dirs, vec_dirs = batch_process_videos(
        args, video_files, extract_mask=False, extract_pose=False
    )
    
    logger.success("帧提取完成！")
    logger.info(f"总 rec 视频数量: {len(all_rec_dirs)}")
    logger.info(f"总 vec 视频数量: {len(all_vec_dirs)}")
    logger.info(f"新增 rec 视频数量: {len(rec_dirs)}")
    logger.info(f"新增 vec 视频数量: {len(vec_dirs)}")
    
    return True


def extract_all_faces(args):
    """仅提取所有帧的人脸遮罩"""
    # 查找所有已经提取了帧的视频目录
    video_dirs = []
    for category in ["rec", "vec"]:
        category_dir = os.path.join(args.target_dir, category)
        if os.path.exists(category_dir):
            for folder in os.listdir(category_dir):
                folder_path = os.path.join(category_dir, folder)
                # 排除原始视频目录
                if os.path.isdir(folder_path) and folder != "raw_videos":
                    video_dirs.append(folder_path)
    
    total_dirs = len(video_dirs)
    logger.info(f"找到 {total_dirs} 个视频目录")
    
    if total_dirs == 0:
        logger.warning("没有找到视频目录，请先运行抽帧:")
        logger.warning(f"  python {sys.argv[0]} --extract_frames_only")
        return False
    
    success_count = 0
    
    for video_dir in tqdm(video_dirs, desc="提取人脸遮罩"):
        images_dir = os.path.join(video_dir, "images")
        faces_dir = os.path.join(video_dir, "faces")
        
        if not os.path.exists(images_dir):
            logger.warning(f"警告: {video_dir} 中没有找到图像目录，请先运行抽帧:")
            logger.warning(f"  python {sys.argv[0]} --extract_frames_only")
            continue
        
        # 清空现有的人脸遮罩目录
        if args.force and os.path.exists(faces_dir):
            shutil.rmtree(faces_dir)
        
        # 提取人脸遮罩
        if extract_face_masks(args.stableanimator_dir, images_dir, video_dir):
            success_count += 1
    
    logger.success(f"人脸遮罩提取完成！成功处理 {success_count}/{total_dirs} 个目录")
    return True


def extract_all_poses(args):
    """仅提取所有帧的人体姿态骨架"""
    # 查找所有已经提取了帧的视频目录
    video_dirs = []
    for category in ["rec", "vec"]:
        category_dir = os.path.join(args.target_dir, category)
        if os.path.exists(category_dir):
            for folder in os.listdir(category_dir):
                folder_path = os.path.join(category_dir, folder)
                # 排除原始视频目录
                if os.path.isdir(folder_path) and folder != "raw_videos":
                    video_dirs.append(folder_path)
    
    total_dirs = len(video_dirs)
    logger.info(f"找到 {total_dirs} 个视频目录")
    
    if total_dirs == 0:
        logger.warning("没有找到视频目录，请先运行抽帧:")
        logger.warning(f"  python {sys.argv[0]} --extract_frames_only")
        return False
    
    success_count = 0
    
    for video_dir in tqdm(video_dirs, desc="提取姿态骨架"):
        poses_dir = os.path.join(video_dir, "poses")
        
        # 清空现有的姿态骨架目录
        if args.force and os.path.exists(poses_dir):
            shutil.rmtree(poses_dir)
        
        # 提取姿态骨架
        if extract_poses_for_video(args.stableanimator_dir, video_dir):
            success_count += 1
    
    logger.success(f"人体姿态骨架提取完成！成功处理 {success_count}/{total_dirs} 个目录")
    return True


def check_dataset(args):
    """检查数据集的完整性"""
    # 查找所有视频目录
    video_dirs = []
    for category in ["rec", "vec"]:
        category_dir = os.path.join(args.target_dir, category)
        if os.path.exists(category_dir):
            for folder in os.listdir(category_dir):
                folder_path = os.path.join(category_dir, folder)
                # 排除原始视频目录
                if os.path.isdir(folder_path) and folder != "raw_videos":
                    video_dirs.append(folder_path)
    
    total_dirs = len(video_dirs)
    logger.info(f"找到 {total_dirs} 个视频目录")
    
    if total_dirs == 0:
        logger.warning("没有找到视频目录，请先运行抽帧:")
        logger.warning(f"  python {sys.argv[0]} --extract_frames_only")
        return False
    
    issues = []
    
    for video_dir in tqdm(video_dirs, desc="检查数据集"):
        images_dir = os.path.join(video_dir, "images")
        faces_dir = os.path.join(video_dir, "faces")
        poses_dir = os.path.join(video_dir, "poses")
        
        # 检查目录是否存在
        if not os.path.exists(images_dir):
            issues.append(f"缺少图像目录: {images_dir}")
            continue
        
        if not os.path.exists(faces_dir):
            issues.append(f"缺少人脸目录: {faces_dir}")
        
        if not os.path.exists(poses_dir):
            issues.append(f"缺少姿态目录: {poses_dir}")
        
        # 检查文件数量
        image_files = glob.glob(f"{images_dir}/*.png")
        image_count = len(image_files)
        
        if image_count == 0:
            issues.append(f"图像目录为空: {images_dir}")
            continue
        
        if os.path.exists(faces_dir):
            face_count = len(glob.glob(f"{faces_dir}/*.png"))
            if face_count != image_count:
                issues.append(f"人脸遮罩数量不匹配: {video_dir} (图像: {image_count}, 人脸: {face_count})")
        
        if os.path.exists(poses_dir):
            pose_count = len(glob.glob(f"{poses_dir}/*.png"))
            if pose_count != image_count:
                issues.append(f"姿态骨架数量不匹配: {video_dir} (图像: {image_count}, 姿态: {pose_count})")
    
    # 打印问题
    if issues:
        logger.warning(f"发现 {len(issues)} 个问题:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        
        logger.info("\n可能的解决方案:")
        logger.info(f"  - 提取缺失的人脸遮罩: python {sys.argv[0]} --extract_faces_only")
        logger.info(f"  - 提取缺失的姿态骨架: python {sys.argv[0]} --extract_poses_only")
        logger.info(f"  - 强制重新提取人脸遮罩: python {sys.argv[0]} --extract_faces_only --force")
        return False
    else:
        logger.success("数据集完整性检查通过！")
        return True


def main():
    # 获取脚本所在目录作为项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description='StableAnimator 数据集准备工具')
    parser.add_argument('--source_dir', type=str, 
                        default=os.path.join(script_dir, 'animation_data', 'raw_videos'),
                        help='源视频目录')
    parser.add_argument('--video_resolution', type=str, choices=['rec', 'vec'], default='vec',
                        help='视频分辨率类型 (rec: 512x512, vec: 576x1024)')
    parser.add_argument('--raw_videos_dir', type=str, default=None,
                        help='原始视频目录 (如果不指定，将基于 --video_resolution 自动构建)')
    parser.add_argument('--target_dir', type=str, 
                        default=os.path.join(script_dir, 'animation_data'),
                        help='目标数据集目录')
    parser.add_argument('--stableanimator_dir', type=str, 
                        default=script_dir,
                        help='StableAnimator 代码目录')
    parser.add_argument('--max_workers', type=int, default=4, 
                        help='并行处理的最大线程数')
    parser.add_argument('--skip_pose', action='store_true',
                        help='跳过姿态提取步骤')
    parser.add_argument('--extract_frames_only', action='store_true',
                        help='仅提取视频帧')
    parser.add_argument('--extract_faces_only', action='store_true',
                        help='仅提取人脸遮罩')
    parser.add_argument('--extract_poses_only', action='store_true',
                        help='仅提取人体姿态骨架')
    parser.add_argument('--check_dataset', action='store_true',
                        help='检查数据集完整性')
    parser.add_argument('--force', action='store_true',
                        help='强制重新生成已存在的文件')
    args = parser.parse_args()
    
    # 如果未指定原始视频目录，则基于视频分辨率参数构建
    if args.raw_videos_dir is None:
        args.raw_videos_dir = os.path.join(args.target_dir, args.video_resolution, 'raw_videos')
    
    # 创建日志目录
    os.makedirs(os.path.join(script_dir, 'logs'), exist_ok=True)
    
    # 创建必要的目录结构
    os.makedirs(os.path.join(args.target_dir, "rec"), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, "vec"), exist_ok=True)
    os.makedirs(args.raw_videos_dir, exist_ok=True)
    
    # 根据参数执行不同的功能
    if args.check_dataset:
        logger.info("检查数据集完整性...")
        check_dataset(args)
        return
    
    if args.extract_frames_only:
        logger.info("仅提取视频帧...")
        extract_all_frames(args)
        return
    
    if args.extract_faces_only:
        logger.info("仅提取人脸遮罩...")
        extract_all_faces(args)
        return
    
    if args.extract_poses_only:
        logger.info("仅提取人体姿态骨架...")
        extract_all_poses(args)
        return
    
    # 完整处理流程
    logger.info("执行完整数据集准备流程...")
    
    # 获取视频文件列表
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    video_files = []
    
    if args.raw_videos_dir:
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(args.raw_videos_dir, f"**/*{ext}"), recursive=True))
    else:
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(args.source_dir, f"**/*{ext}"), recursive=True))
    
    # 按视频名称排序
    video_files.sort(key=lambda x: os.path.basename(x))
    
    total_videos = len(video_files)
    logger.info(f"找到 {total_videos} 个视频文件")
    
    if total_videos == 0:
        logger.warning("没有找到视频文件，退出")
        return
    
    # 处理视频，提取帧、人脸遮罩和姿态骨架
    all_rec_dirs, all_vec_dirs, rec_dirs, vec_dirs = batch_process_videos(
        args, video_files, extract_mask=True, extract_pose=not args.skip_pose
    )
    
    logger.success("数据集准备完成！")
    logger.info(f"总 rec 视频数量: {len(all_rec_dirs)}")
    logger.info(f"总 vec 视频数量: {len(all_vec_dirs)}")
    logger.info(f"新增 rec 视频数量: {len(rec_dirs)}")
    logger.info(f"新增 vec 视频数量: {len(vec_dirs)}")
    
    # 检查数据集完整性
    check_dataset(args)


def check_dependencies(stableanimator_dir):
    """检查必要的依赖是否已安装"""
    try:
        # 检查 loguru
        try:
            import loguru
            logger.info(f"loguru 版本: {loguru.__version__}")
        except ImportError:
            print("错误: 缺少 loguru 包")
            print("请安装: pip install loguru")
            return False
        
        # 检查 OpenCV
        try:
            import cv2
            logger.info(f"OpenCV 版本: {cv2.__version__}")
        except ImportError:
            logger.error("错误: 缺少 OpenCV 包")
            logger.error("请安装: pip install opencv-python")
            return False
        
        # 检查 ffmpeg
        try:
            result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                version_line = result.stdout.decode('utf-8').split('\n')[0]
                logger.info(f"ffmpeg 已安装: {version_line}")
            else:
                logger.warning("警告: ffmpeg 可能未正确安装，将使用 OpenCV 作为备选")
        except:
            logger.warning("警告: ffmpeg 未安装或不在 PATH 中，将使用 OpenCV 作为备选")
        
        # 检查 ffprobe
        try:
            result = subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                version_line = result.stdout.decode('utf-8').split('\n')[0]
                logger.info(f"ffprobe 已安装: {version_line}")
            else:
                logger.warning("警告: ffprobe 可能未正确安装，将使用 OpenCV 作为备选")
        except:
            logger.warning("警告: ffprobe 未安装或不在 PATH 中，将使用 OpenCV 作为备选")
        
        # 检查 StableAnimator 相关脚本
        face_mask_script = os.path.join(stableanimator_dir, "face_mask_extraction.py")
        if not os.path.exists(face_mask_script):
            logger.warning(f"警告: face_mask_extraction.py 文件不存在: {face_mask_script}")
            logger.warning("人脸遮罩提取可能会失败")
        
        return True
    except Exception as e:
        logger.error(f"依赖检查失败: {str(e)}")
        return False


def print_help_message():
    """打印帮助信息"""
    help_text = """
StableAnimator 数据集准备工具使用指南:
-----------------------------------
1. 完整处理流程:
   python prepare_dataset.py
   这将执行抽帧、人脸遮罩提取和姿态骨架提取的完整流程

2. 指定视频分辨率类型:
   python prepare_dataset.py --video_resolution=rec
   强制将所有视频处理为 rec 类型 (512x512)
   
   python prepare_dataset.py --video_resolution=vec
   强制将所有视频处理为 vec 类型 (576x1024)

3. 仅提取视频帧:
   python prepare_dataset.py --extract_frames_only
   从视频文件提取帧并创建数据集目录结构

4. 仅提取人脸遮罩:
   python prepare_dataset.py --extract_faces_only
   为已提取的帧生成人脸遮罩

5. 仅提取姿态骨架:
   python prepare_dataset.py --extract_poses_only
   为已提取的帧生成姿态骨架

6. 检查数据集完整性:
   python prepare_dataset.py --check_dataset
   检查数据集中是否有缺失的文件或目录

7. 强制重新生成:
   python prepare_dataset.py --extract_faces_only --force
   强制重新生成已存在的人脸遮罩

8. 跳过姿态提取:
   python prepare_dataset.py --skip_pose
   执行完整流程但跳过姿态提取步骤

9. 指定原始视频目录:
   python prepare_dataset.py --raw_videos_dir=/path/to/videos
   从指定目录读取原始视频文件
"""
    logger.info(help_text)


if __name__ == "__main__":
    # 获取脚本所在目录作为项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    logger.info("StableAnimator 数据集准备工具")
    
    # 解析命令行参数，获取 stableanimator_dir
    parser = argparse.ArgumentParser(description='StableAnimator 数据集准备工具')
    parser.add_argument('--stableanimator_dir', type=str, default=script_dir,
                        help='StableAnimator 代码目录')
    args, _ = parser.parse_known_args()
    
    # 创建日志目录
    os.makedirs(os.path.join(script_dir, 'logs'), exist_ok=True)
    
    # 检查依赖
    if not check_dependencies(args.stableanimator_dir):
        sys.exit(1)
    
    # 打印帮助信息
    if len(sys.argv) == 1 or '--help' in sys.argv or '-h' in sys.argv:
        print_help_message()
        if len(sys.argv) == 1:
            logger.info("\n没有指定参数，将执行完整处理流程...\n")
    
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.success(f"总耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")



