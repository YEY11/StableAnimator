#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StableAnimator 数据集自动化预处理脚本

本脚本旨在为 StableAnimator 项目提供一个全自动的数据集准备工具。
它能够处理大量的原始视频文件，并将其转换为模型训练所需的格式，包括：
- 视频抽帧（默认 16 FPS）
- 根据分辨率自动分类 (rec/vec)
- 提取人脸遮罩 (face mask)
- 提取人体姿态骨架 (pose)

主要功能与特性:
- 支持多进程并行处理，显著提升效率。
- 支持多 GPU 加速人脸和姿态提取。
- 具备断点续传功能，意外中断后可恢复进度。
- 详细的日志记录和直观的进度条。
- 模块化设计，支持完整流程或分步执行（如仅抽帧、仅提取人脸等）。
- 提供数据集完整性检查和清理工具。

使用示例:
- 完整处理 (默认16fps): python data_preprocess.py
- 指定抽帧为8fps: python data_preprocess.py --fps=8
- 仅抽帧: python data_preprocess.py --extract_frames_only
- 使用 GPU 0,1 加速: python data_preprocess.py --gpus=0,1
- 清理所有结果: python data_preprocess.py --clean --force

完整示例：同时指定原始视频目录、目标目录、GPU、抽帧频率等参数:
python data_preprocess.py --raw_videos_dir=animation_data/raw_videos --target_dir=animation_data/fashion_sub7_fps8 --gpus=0,1 --fps=8
"""

import os
import sys
import argparse
import subprocess
import glob
import shutil
import time
import cv2
from multiprocessing import Pool, Manager, cpu_count
from tqdm import tqdm
import pickle
import hashlib
import signal
import traceback
from loguru import logger

# --- 日志配置 ---
# 配置 loguru 日志记录器，用于输出信息到控制台和文件
logger.remove()
# 控制台输出格式
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
# 文件输出格式，带轮换和保留策略
logger.add("./logs/data_preprocess.log", rotation="10 MB", retention="1 week")

# --- 全局常量 ---
# 文件夹编号格式设置，例如 00001, 00002, ...
FOLDER_NUM_FORMAT = "{:05d}"


def get_video_resolution(video_path):
    """
    获取视频文件的分辨率（宽度和高度）。
    优先使用 ffprobe，如果失败则回退到 OpenCV。

    Args:
        video_path (str): 视频文件的路径。

    Returns:
        tuple[int, int]: (宽度, 高度)。如果失败则返回 (0, 0)。
    """
    try:
        # 优先使用 ffprobe，它通常更快速且资源占用少
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
        # ffprobe 失败，尝试使用 OpenCV 作为备选方案
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


def extract_frames(video_path, output_dir, fps=16):
    """
    从视频中提取帧并保存为 PNG 图片。支持指定抽帧频率 (FPS)。

    Args:
        video_path (str): 输入视频文件的路径。
        output_dir (str): 保存帧图像的输出目录。
        fps (int, optional): 目标抽帧频率。如果为 0，则使用视频原始帧率。默认为 16。

    Returns:
        bool: 如果成功提取帧则返回 True，否则返回 False。
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # 尝试使用 ffmpeg，效率更高
        try:
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-q:v", "1",  # 使用高质量 (1) 保存图片
            ]

            # 如果指定了FPS，则添加视频过滤器以改变帧率
            if fps > 0:
                cmd.extend(["-vf", f"fps={fps}"])

            cmd.extend([
                "-start_number", "0",
                f"{output_dir}/frame_%d.png"
            ])

            # 执行命令，并隐藏标准输出和错误，以防刷屏
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except Exception as ffmpeg_error:
            logger.warning(f"ffmpeg 提取帧失败，尝试使用 OpenCV: {str(ffmpeg_error)}")

            # 如果 ffmpeg 失败，回退到 OpenCV
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                raise Exception("无法打开视频文件")

            original_fps = video.get(cv2.CAP_PROP_FPS)
            # 确保原始帧率有效
            if original_fps <= 0:
                logger.warning(f"无法获取视频 {video_path} 的有效原始帧率，将逐帧提取。")
                fps = 0 # 强制逐帧提取

            frame_skip_interval = 0
            if fps > 0 and original_fps > 0:
                # 计算需要跳过的帧数。例如，原30fps，目标15fps，则每 30/15=2 帧取一帧。
                frame_skip_interval = round(original_fps / fps)

            read_count = 0
            saved_count = 0
            while True:
                success, frame = video.read()
                if not success:
                    break

                # 如果需要跳帧，则根据间隔保存
                if frame_skip_interval > 0:
                    if read_count % frame_skip_interval == 0:
                        output_path = os.path.join(output_dir, f"frame_{saved_count}.png")
                        cv2.imwrite(output_path, frame)
                        saved_count += 1
                # 否则，保存每一帧 (fps=0 的情况)
                else:
                    output_path = os.path.join(output_dir, f"frame_{saved_count}.png")
                    cv2.imwrite(output_path, frame)
                    saved_count += 1

                read_count += 1

            video.release()

            if saved_count == 0:
                raise Exception("未能提取任何帧")

            return True
    except Exception as e:
        logger.error(f"提取帧失败 {video_path}: {str(e)}")
        return False


def extract_face_masks(stableanimator_dir, image_folder, video_dir, gpu_id=None):
    """
    调用外部脚本提取人脸遮罩。

    Args:
        stableanimator_dir (str): StableAnimator 项目的根目录。
        image_folder (str): 包含帧图像的文件夹。
        video_dir (str): 当前视频的数据目录，用于存放生成的 'faces' 文件夹。
        gpu_id (int, optional): 用于执行此任务的 GPU ID。默认为 None。

    Returns:
        bool: 如果成功提取则返回 True，否则返回 False。
    """
    try:
        current_dir = os.getcwd()
        os.chdir(stableanimator_dir)  # 切换到项目根目录以正确执行脚本

        faces_dir = os.path.join(video_dir, "faces")
        os.makedirs(faces_dir, exist_ok=True)

        # 设置环境变量以选择特定 GPU
        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cmd = [
            "python", "face_mask_extraction.py",
            f"--image_folder={image_folder}"
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

        os.chdir(current_dir)  # 恢复原始工作目录

        # 检查人脸遮罩是否已生成在目标位置
        mask_count = len(glob.glob(f"{faces_dir}/*.png"))
        if mask_count == 0:
            # 如果目标位置没有，则在几个可能的默认输出位置查找并复制过来
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
                    logger.info(f"找到人脸遮罩目录: {faces_src}，正在复制...")
                    for face_file in glob.glob(f"{faces_src}/*.png"):
                        shutil.copy(face_file, faces_dir)
                    found = True
                    break

            if not found:
                logger.warning(f"未能为 {video_dir} 生成或找到人脸遮罩")
                return False

        logger.success(f"成功为 {video_dir} 提取人脸遮罩")
        return True
    except subprocess.CalledProcessError as e:
        # 捕获子进程执行错误，并打印详细的错误信息
        logger.error(f"提取人脸遮罩失败 {image_folder}: {e.stderr.decode('utf-8')}")
        os.chdir(current_dir)
        return False
    except Exception as e:
        logger.error(f"提取人脸遮罩时出错 {image_folder}: {str(e)}")
        # 确保在任何异常情况下都尝试恢复工作目录
        try:
            os.chdir(current_dir)
        except:
            pass
        return False


def extract_poses_for_video(stableanimator_dir, video_dir, gpu_id=None):
    """
    为单个视频目录中的图像提取人体姿态骨架。

    Args:
        stableanimator_dir (str): StableAnimator 项目的根目录。
        video_dir (str): 当前视频的数据目录, 包含 'images' 子目录。
        gpu_id (int, optional): 用于执行此任务的 GPU ID。默认为 None。

    Returns:
        bool: 如果成功提取则返回 True，否则返回 False。
    """
    try:
        # 检查姿态提取所需的模型文件是否存在
        dwpose_dir = os.path.join(stableanimator_dir, "checkpoints", "DWPose")
        yolox_model = os.path.join(dwpose_dir, "yolox_l.onnx")
        dwpose_model = os.path.join(dwpose_dir, "dw-ll_ucoco_384.onnx")

        if not os.path.exists(yolox_model) or not os.path.exists(dwpose_model):
            # 如果模型不存在，打印警告和下载说明
            logger.warning(f"警告: DWPose 模型文件不存在，请确保以下文件存在:")
            logger.warning(f"  - {yolox_model}")
            logger.warning(f"  - {dwpose_model}")
            logger.warning(f"请执行以下命令下载模型文件:")
            logger.warning(f"  mkdir -p {dwpose_dir}")
            logger.warning(f"  wget -O {yolox_model} https://hf-mirror.com/yzd-v/DWPose/resolve/main/yolox_l.onnx")
            logger.warning(f"  wget -O {dwpose_model} https://hf-mirror.com/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx")
            return False

        images_dir = os.path.join(video_dir, "images")
        poses_dir = os.path.join(video_dir, "poses")

        if not os.path.exists(images_dir):
            logger.warning(f"警告: {video_dir} 中没有找到图像目录")
            return False

        os.makedirs(poses_dir, exist_ok=True)

        # 获取视频的类别和编号，用于传递给外部脚本
        category = os.path.basename(os.path.dirname(video_dir))
        video_number = int(os.path.basename(video_dir))

        current_dir = os.getcwd()
        os.chdir(stableanimator_dir) # 切换工作目录

        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # 调用 DWPose 的训练骨架提取脚本
        cmd = [
            "python", "DWPose/training_skeleton_extraction.py",
            f"--root_path={os.path.dirname(os.path.dirname(video_dir))}",
            f"--name={category}",
            f"--start={video_number}",
            f"--end={video_number}"
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

        os.chdir(current_dir) # 恢复工作目录

        # 检查是否成功生成了姿态文件
        pose_files = glob.glob(f"{poses_dir}/*.png")
        if len(pose_files) == 0:
            logger.warning(f"未能为 {video_dir} 生成姿态骨架")
            return False

        logger.success(f"成功为 {video_dir} 提取姿态骨架")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"提取姿态骨架失败 {video_dir}: {e.stderr.decode('utf-8')}")
        os.chdir(current_dir)
        return False
    except Exception as e:
        logger.error(f"提取姿态骨架时出错 {video_dir}: {str(e)}")
        try:
            os.chdir(current_dir)
        except:
            pass
        return False


def read_mapping_csv(csv_path):
    """
    从 CSV 文件中读取文件夹编号和原始视频名称的映射关系。

    Args:
        csv_path (str): CSV 文件的路径。

    Returns:
        dict: 一个字典，键是原始视频名，值是文件夹编号。
    """
    mappings = {}
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                next(f, None)  # 跳过标题行
                for line in f:
                    parts = line.strip().split(',', 1)
                    if len(parts) == 2:
                        folder_name, video_name = parts
                        mappings[video_name] = folder_name
        except Exception as e:
            logger.error(f"读取映射文件 {csv_path} 失败: {str(e)}")
    return mappings


def write_mapping_csv(csv_path, mappings):
    """
    将文件夹编号和视频名称的映射关系写入 CSV 文件，并按文件夹名排序。

    Args:
        csv_path (str): 要写入的 CSV 文件路径。
        mappings (dict): 包含映射关系的字典 {video_name: folder_num}。

    Returns:
        bool: 写入成功返回 True，否则返回 False。
    """
    try:
        # 反转映射以便按文件夹名排序
        folder_to_video = {v: k for k, v in mappings.items()}
        sorted_folders = sorted(folder_to_video.keys())

        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("folder_name,original_video_name\n")
            for folder_name in sorted_folders:
                video_name = folder_to_video[folder_name]
                f.write(f"{folder_name},{video_name}\n")
        return True
    except Exception as e:
        logger.error(f"写入映射文件 {csv_path} 失败: {str(e)}")
        return False


def update_video_mapping_csv(target_dir, category, folder_num, video_path, lock):
    """
    更新视频映射 CSV 文件，使用锁保证进程安全。

    Args:
        target_dir (str): 目标目录基础路径。
        category (str): 分类 ('rec' 或 'vec')。
        folder_num (str): 文件夹编号。
        video_path (str): 原始视频路径。
        lock (multiprocessing.Lock): 进程锁。
    """
    csv_path = os.path.join(target_dir, category, f"{category}_video_mapping.csv")
    video_name = os.path.basename(video_path)

    with lock: # 使用锁确保多进程下文件读写安全
        mappings = read_mapping_csv(csv_path)
        mappings[video_name] = folder_num
        if write_mapping_csv(csv_path, mappings):
            logger.info(f"已更新视频映射: {folder_num} -> {video_name}")
            return True
        else:
            logger.error(f"更新视频映射失败: {folder_num} -> {video_name}")
            return False


def update_path_file(target_dir, category, dirs, lock):
    """
    更新路径文件 (video_{category}_path.txt) 以保存进度，使用锁保证进程安全。

    Args:
        target_dir (str): 目标目录基础路径。
        category (str): 分类 ('rec' 或 'vec')。
        dirs (list[str]): 要写入文件的目录路径列表。
        lock (multiprocessing.Lock): 进程锁。
    """
    path_file = os.path.join(target_dir, f"video_{category}_path.txt")
    with lock:
        with open(path_file, 'w', encoding='utf-8') as f:
            for dir_path in dirs:
                f.write(f"{dir_path}\n")


def save_progress_state(args, rec_dirs, vec_dirs, processed_videos):
    """
    使用 pickle 保存当前的处理进度状态，用于断点续传。

    Args:
        args (argparse.Namespace): 命令行参数。
        rec_dirs (list[str]): 已处理的 rec 目录列表。
        vec_dirs (list[str]): 已处理的 vec 目录列表。
        processed_videos (list[str]): 已处理视频的哈希值列表。

    Returns:
        bool: 保存成功返回 True，否则返回 False。
    """
    progress_file = os.path.join(args.target_dir, "progress_state.pkl")
    try:
        state = {
            'timestamp': time.time(),
            'rec_dirs': list(rec_dirs),
            'vec_dirs': list(vec_dirs),
            'processed_videos': list(processed_videos),
            'args': {
                'target_dir': args.target_dir,
                'video_resolution': args.video_resolution,
                'raw_videos_dir': args.raw_videos_dir,
                'skip_pose': args.skip_pose,
                'fps': args.fps,
            }
        }
        with open(progress_file, 'wb') as f:
            pickle.dump(state, f)
        logger.info(f"进度状态已保存: {progress_file}")
        return True
    except Exception as e:
        logger.error(f"保存进度状态失败: {str(e)}")
        return False


def load_progress_state(args):
    """
    从 pickle 文件加载之前的处理进度状态。

    Args:
        args (argparse.Namespace): 当前的命令行参数，用于兼容性检查。

    Returns:
        dict or None: 如果找到并成功加载兼容的进度文件，则返回状态字典，否则返回 None。
    """
    progress_file = os.path.join(args.target_dir, "progress_state.pkl")
    if not os.path.exists(progress_file):
        logger.info("未找到进度状态文件，将从头开始处理")
        return None

    try:
        with open(progress_file, 'rb') as f:
            state = pickle.load(f)

        # 检查状态文件是否与当前参数兼容，不兼容则从头开始
        if (state['args']['target_dir'] != args.target_dir or
            state['args']['video_resolution'] != args.video_resolution or
            state['args'].get('fps', 16) != args.fps): # 检查fps是否一致
            logger.warning("进度状态文件与当前参数不匹配（如目标目录或FPS），将从头开始处理")
            return None

        timestamp = state.get('timestamp', 0)
        time_diff = time.time() - timestamp
        hours, rem = divmod(time_diff, 3600)
        minutes, _ = divmod(rem, 60)

        logger.info(f"加载进度状态: 上次运行于 {int(hours)}小时 {int(minutes)}分钟前")
        logger.info(f"已处理视频数量: {len(state['processed_videos'])}")
        return state
    except Exception as e:
        logger.error(f"加载进度状态失败: {str(e)}")
        return None


def generate_video_hash(video_path):
    """
    为视频文件生成一个唯一的哈希值，用于在进度跟踪中标识视频，避免重复处理。
    哈希值基于文件名、文件大小和最后修改时间生成。

    Args:
        video_path (str): 视频文件的路径。

    Returns:
        str: 生成的 MD5 哈希字符串。
    """
    try:
        file_stat = os.stat(video_path)
        video_name = os.path.basename(video_path)
        hash_str = f"{video_name}_{file_stat.st_size}_{file_stat.st_mtime}"
        return hashlib.md5(hash_str.encode()).hexdigest()
    except Exception as e:
        logger.error(f"生成视频哈希失败 {video_path}: {str(e)}")
        return os.path.basename(video_path) # 失败时使用文件名作为备用


def process_video_task(args_dict):
    """
    处理单个视频的核心任务函数，设计用于多进程池。
    执行抽帧、人脸遮罩提取和姿态提取的完整流程。

    Args:
        args_dict (dict): 包含所有任务所需参数的字典。

    Returns:
        dict or None: 如果处理成功，返回包含处理结果信息的字典；否则返回 None。
    """
    # 解包参数，使代码更清晰
    video_path = args_dict['video_path']
    args = args_dict['args']
    extract_mask = args_dict['extract_mask']
    extract_pose = args_dict['extract_pose']
    forced_type = args_dict['forced_type']
    gpu_id = args_dict['gpu_id']
    shared_dict = args_dict['shared_dict']
    task_id = args_dict.get('task_id', 0)

    video_hash = generate_video_hash(video_path)

    try:
        # 1. 检查是否已处理
        with shared_dict['lock']:
            if video_hash in shared_dict['processed_hashes']:
                logger.info(f"跳过已处理的视频: {os.path.basename(video_path)}")
                return None
            shared_dict['in_progress_hashes'][video_hash] = 1

        video_name = os.path.basename(video_path).rsplit('.', 1)[0]
        logger.info(f"[{task_id}/{shared_dict['total_tasks']}] 开始处理: {video_name}")

        # 2. 获取分辨率并确定类型
        width, height = get_video_resolution(video_path)
        if width == 0 or height == 0:
            logger.warning(f"跳过视频 {video_path} (无法获取分辨率)")
            return None

        if forced_type:
            target_type = forced_type
            is_rec = (forced_type == "rec")
        else:
            if width <= 512 and height <= 512:
                target_type = "rec"
                is_rec = True
            else:
                target_type = "vec"
                is_rec = False

        # 3. 获取预分配的文件夹编号并创建目录
        folder_num = shared_dict['video_to_folder'].get(video_path)
        if not folder_num:
            logger.error(f"严重错误: 未能为视频 {video_path} 找到预分配的文件夹编号。")
            return None

        video_dir = os.path.join(args.target_dir, target_type, folder_num)
        images_dir = os.path.join(video_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        logger.info(f"[{task_id}] 视频 '{video_name}' -> 目录 '{video_dir}'")

        # 4. 执行核心预处理步骤
        # 步骤 4.1: 提取帧
        logger.info(f"[{task_id}] 抽帧 (FPS={args.fps})...")
        if not extract_frames(video_path, images_dir, fps=args.fps):
            logger.error(f"跳过视频 {video_path} (帧提取失败)")
            shutil.rmtree(video_dir, ignore_errors=True)
            return None

        frame_count = len(glob.glob(f"{images_dir}/*.png"))
        if frame_count < 5:
            logger.warning(f"跳过视频 {video_path} (提取的帧数不足: {frame_count})")
            shutil.rmtree(video_dir, ignore_errors=True)
            return None

        # 步骤 4.2: 提取人脸遮罩
        if extract_mask:
            logger.info(f"[{task_id}] 提取人脸遮罩...")
            if not extract_face_masks(args.stableanimator_dir, images_dir, video_dir, gpu_id):
                logger.warning(f"[{task_id}] 人脸遮罩提取失败: {video_dir}")

        # 步骤 4.3: 提取姿态骨架
        if extract_pose and not args.skip_pose:
            logger.info(f"[{task_id}] 提取姿态骨架...")
            if not extract_poses_for_video(args.stableanimator_dir, video_dir, gpu_id):
                logger.warning(f"[{task_id}] 姿态骨架提取失败: {video_dir}")

        logger.success(f"[{task_id}] 已完成 {video_name} 的处理")

        # 5. 更新元数据和进度
        csv_path = os.path.join(args.target_dir, target_type, f"{target_type}_video_mapping.csv")
        with shared_dict['lock']:
            mappings = read_mapping_csv(csv_path)
            mappings[os.path.basename(video_path)] = folder_num
            write_mapping_csv(csv_path, mappings)

        with shared_dict['lock']:
            shared_dict['processed_hashes'][video_hash] = 1
            if video_hash in shared_dict['in_progress_hashes']:
                del shared_dict['in_progress_hashes'][video_hash]

        result = {'type': target_type, 'dir': video_dir, 'is_rec': is_rec, 'video_path': video_path, 'video_hash': video_hash}

        with shared_dict['lock']:
            shared_dict['completed_count'] += 1
            if shared_dict['completed_count'] % 10 == 0 or shared_dict['completed_count'] == shared_dict['total_tasks']:
                # 定期或在最后保存进度
                all_rec = list(shared_dict['existing_rec_dirs']) + [r['dir'] for r in shared_dict['results_list'] if r['is_rec']]
                all_vec = list(shared_dict['existing_vec_dirs']) + [r['dir'] for r in shared_dict['results_list'] if not r['is_rec']]
                save_progress_state(args, all_rec, all_vec, list(shared_dict['processed_hashes'].keys()))

        return result

    except Exception as e:
        logger.error(f"处理视频 {video_path} 时出错: {str(e)}")
        logger.error(traceback.format_exc())
        with shared_dict['lock']:
            if video_hash in shared_dict['in_progress_hashes']:
                del shared_dict['in_progress_hashes'][video_hash]
        return None


def process_video_batch(args, video_files, extract_mask=True, extract_pose=True):
    """
    使用多进程并行处理一个批次的视频文件。

    Args:
        args (argparse.Namespace): 命令行参数。
        video_files (list[str]): 待处理的视频文件路径列表。
        extract_mask (bool, optional): 是否提取人脸遮罩。默认为 True。
        extract_pose (bool, optional): 是否提取姿态。默认为 True。

    Returns:
        tuple: (all_rec_dirs, all_vec_dirs, new_rec_dirs, new_vec_dirs)
    """
    existing_rec_dirs = sorted([d for d in glob.glob(os.path.join(args.target_dir, "rec", "*")) if os.path.isdir(d) and os.path.basename(d) != "raw_videos"])
    existing_vec_dirs = sorted([d for d in glob.glob(os.path.join(args.target_dir, "vec", "*")) if os.path.isdir(d) and os.path.basename(d) != "raw_videos"])

    manager = Manager()
    shared_dict = manager.dict({
        'lock': manager.Lock(),
        'processed_hashes': manager.dict(),
        'in_progress_hashes': manager.dict(),
        'video_to_folder': manager.dict(),
        'completed_count': 0,
        'results_list': manager.list(), # 用于收集结果
        'existing_rec_dirs': manager.list(existing_rec_dirs),
        'existing_vec_dirs': manager.list(existing_vec_dirs)
    })

    # 预分配文件夹编号
    sorted_videos = sorted(video_files, key=lambda x: os.path.basename(x))
    rec_next_num = len(existing_rec_dirs) + 1
    vec_next_num = len(existing_vec_dirs) + 1

    for video_path in sorted_videos:
        width, height = get_video_resolution(video_path)
        if width == 0 or height == 0: continue

        is_rec = (args.video_resolution == 'rec') or (args.video_resolution != 'vec' and width <= 512 and height <= 512)

        if is_rec:
            shared_dict['video_to_folder'][video_path] = FOLDER_NUM_FORMAT.format(rec_next_num)
            rec_next_num += 1
        else:
            shared_dict['video_to_folder'][video_path] = FOLDER_NUM_FORMAT.format(vec_next_num)
            vec_next_num += 1

    # 加载进度
    state = load_progress_state(args)
    if state and args.resume:
        for h in state['processed_videos']: shared_dict['processed_hashes'][h] = 1

    video_hashes = {generate_video_hash(v): v for v in video_files}
    video_files_to_process = [v for h, v in video_hashes.items() if h not in shared_dict['processed_hashes'].keys()]

    if args.force:
        logger.info("--force 参数已启用，将重新处理所有视频。")
        shared_dict['processed_hashes'].clear()
        video_files_to_process = sorted_videos

    logger.info(f"总视频文件数: {len(video_files)}, 待处理视频文件数: {len(video_files_to_process)}")
    if not video_files_to_process:
        logger.info("所有视频均已处理，无需操作。")
        return existing_rec_dirs, existing_vec_dirs, [], []

    shared_dict['total_tasks'] = len(video_files_to_process)
    gpu_ids = [int(gid.strip()) for gid in args.gpus.split(',')] if args.gpus else []

    task_args = [{
        'video_path': video_path, 'args': args,
        'extract_mask': extract_mask, 'extract_pose': extract_pose,
        'forced_type': args.video_resolution if args.video_resolution in ['rec', 'vec'] else None,
        'gpu_id': gpu_ids[i % len(gpu_ids)] if gpu_ids else None,
        'shared_dict': shared_dict, 'task_id': i + 1
    } for i, video_path in enumerate(video_files_to_process)]

    processed_results = []
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    try:
        signal.signal(signal.SIGINT, original_sigint_handler)
        with Pool(processes=args.max_workers) as pool:
            with tqdm(total=len(task_args), desc="处理视频") as pbar:
                for result in pool.imap_unordered(process_video_task, task_args):
                    if result:
                        processed_results.append(result)
                        shared_dict['results_list'].append(result) # 添加到共享列表
                    pbar.update(1)
    except KeyboardInterrupt:
        logger.warning("检测到中断信号，正在优雅地停止处理...")
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)
    
    # 最终结果整理
    rec_dirs = [r['dir'] for r in processed_results if r['is_rec']]
    vec_dirs = [r['dir'] for r in processed_results if not r['is_rec']]
    all_rec_dirs = existing_rec_dirs + rec_dirs
    all_vec_dirs = existing_vec_dirs + vec_dirs

    update_path_file(args.target_dir, "rec", all_rec_dirs, shared_dict['lock'])
    update_path_file(args.target_dir, "vec", all_vec_dirs, shared_dict['lock'])
    
    # 最后保存一次总进度
    all_processed_hashes = list(shared_dict['processed_hashes'].keys())
    save_progress_state(args, all_rec_dirs, all_vec_dirs, all_processed_hashes)

    return all_rec_dirs, all_vec_dirs, rec_dirs, vec_dirs


def get_video_source_path(args):
    """根据参数确定视频源目录"""
    if args.raw_videos_dir:
        video_source_path = args.raw_videos_dir
    else:
        video_source_path = args.source_dir
    logger.info(f"正在从目录 '{video_source_path}' 搜索视频文件...")
    return video_source_path

def find_video_files(video_source_path):
    """从指定路径查找视频文件"""
    if not os.path.isdir(video_source_path):
        logger.error(f"视频源目录不存在: '{video_source_path}'")
        logger.error("请使用 --raw_videos_dir /path/to/your/videos 指定正确的目录。")
        return None

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_source_path, f"**/*{ext}"), recursive=True))
    video_files.sort(key=lambda x: os.path.basename(x))
    logger.info(f"找到 {len(video_files)} 个视频文件")
    return video_files

def run_multiprocess_task(task_func, task_args, desc, max_workers):
    """通用的多进程任务执行器"""
    success_count = 0
    error_items = []
    with Pool(processes=max_workers) as pool, tqdm(total=len(task_args), desc=desc) as pbar:
        for i, result in enumerate(pool.imap_unordered(task_func, task_args)):
            if result:
                success_count += 1
            else:
                error_items.append(task_args[i])
            pbar.update(1)
    return success_count, error_items

def extract_all_frames(args):
    """模式：仅提取所有视频的帧"""
    video_source_path = get_video_source_path(args)
    video_files = find_video_files(video_source_path)
    if not video_files: return False

    all_rec_dirs, all_vec_dirs, rec_dirs, vec_dirs = process_video_batch(args, video_files, extract_mask=False, extract_pose=False)

    logger.success("帧提取完成！")
    logger.info(f"总 rec 视频: {len(all_rec_dirs)}, 本次新增: {len(rec_dirs)}")
    logger.info(f"总 vec 视频: {len(all_vec_dirs)}, 本次新增: {len(vec_dirs)}")
    return True

def get_all_subdirs(target_dir):
    """获取所有已处理的视频子目录"""
    video_dirs = []
    for category in ["rec", "vec"]:
        cat_dir = os.path.join(target_dir, category)
        if os.path.exists(cat_dir):
            video_dirs.extend([os.path.join(cat_dir, f) for f in os.listdir(cat_dir) if os.path.isdir(os.path.join(cat_dir, f)) and f != "raw_videos"])
    return video_dirs

def process_face_mask_task(args_dict):
    """用于多进程的单个目录人脸遮罩提取任务。"""
    video_dir, args, gpu_id, task_id = args_dict['video_dir'], args_dict['args'], args_dict['gpu_id'], args_dict.get('task_id', 0)
    try:
        images_dir = os.path.join(video_dir, "images")
        faces_dir = os.path.join(video_dir, "faces")
        if not os.path.exists(images_dir):
            logger.warning(f"[{task_id}] 警告: {video_dir} 中没有图像目录，跳过")
            return False

        if os.path.exists(faces_dir) and not args.force:
            image_count = len(glob.glob(f"{images_dir}/*.png"))
            if image_count > 0 and len(glob.glob(f"{faces_dir}/*.png")) >= image_count:
                logger.info(f"[{task_id}] 跳过已完成的人脸遮罩: {os.path.basename(video_dir)}")
                return True

        if args.force and os.path.exists(faces_dir): shutil.rmtree(faces_dir)

        logger.info(f"[{task_id}] 处理人脸遮罩: {os.path.basename(video_dir)}")
        return extract_face_masks(args.stableanimator_dir, images_dir, video_dir, gpu_id)
    except Exception as e:
        logger.error(f"[{task_id}] 处理人脸遮罩时出错 {video_dir}: {str(e)}\n{traceback.format_exc()}")
        return False

def extract_all_faces(args):
    """模式：为所有已抽帧的目录提取人脸遮罩"""
    video_dirs = get_all_subdirs(args.target_dir)
    if not video_dirs:
        logger.warning(f"未找到视频目录，请先运行 --extract_frames_only")
        return False

    logger.info(f"找到 {len(video_dirs)} 个待处理人脸遮罩的目录")
    gpu_ids = [int(gid.strip()) for gid in args.gpus.split(',')] if args.gpus else []
    task_args = [{'video_dir': vd, 'args': args, 'gpu_id': gpu_ids[i % len(gpu_ids)] if gpu_ids else None, 'task_id': i + 1} for i, vd in enumerate(video_dirs)]

    success_count, error_tasks = run_multiprocess_task(process_face_mask_task, task_args, "提取人脸遮罩", args.max_workers)

    if error_tasks:
        error_dirs = [t['video_dir'] for t in error_tasks]
        error_file = os.path.join(args.target_dir, "face_extraction_errors.txt")
        with open(error_file, 'w') as f: f.writelines(f"{d}\n" for d in error_dirs)
        logger.warning(f"部分目录处理失败，详情已保存到: {error_file}")

    logger.success(f"人脸遮罩提取完成！成功 {success_count}/{len(video_dirs)} 个目录")
    return True

def process_pose_task(args_dict):
    """用于多进程的单个目录姿态骨架提取任务。"""
    video_dir, args, gpu_id, task_id = args_dict['video_dir'], args_dict['args'], args_dict['gpu_id'], args_dict.get('task_id', 0)
    try:
        images_dir = os.path.join(video_dir, "images")
        poses_dir = os.path.join(video_dir, "poses")
        if not os.path.exists(images_dir):
            logger.warning(f"[{task_id}] 警告: {video_dir} 中没有图像目录，跳过")
            return False

        if os.path.exists(poses_dir) and not args.force:
            image_count = len(glob.glob(f"{images_dir}/*.png"))
            if image_count > 0 and len(glob.glob(f"{poses_dir}/*.png")) >= image_count:
                logger.info(f"[{task_id}] 跳过已完成的姿态骨架: {os.path.basename(video_dir)}")
                return True

        if args.force and os.path.exists(poses_dir): shutil.rmtree(poses_dir)

        logger.info(f"[{task_id}] 处理姿态骨架: {os.path.basename(video_dir)}")
        return extract_poses_for_video(args.stableanimator_dir, video_dir, gpu_id)
    except Exception as e:
        logger.error(f"[{task_id}] 处理姿态骨架时出错 {video_dir}: {str(e)}\n{traceback.format_exc()}")
        return False

def extract_all_poses(args):
    """模式：为所有已抽帧的目录提取人体姿态骨架"""
    video_dirs = get_all_subdirs(args.target_dir)
    if not video_dirs:
        logger.warning(f"未找到视频目录，请先运行 --extract_frames_only")
        return False

    logger.info(f"找到 {len(video_dirs)} 个待处理姿态骨架的目录")
    gpu_ids = [int(gid.strip()) for gid in args.gpus.split(',')] if args.gpus else []
    task_args = [{'video_dir': vd, 'args': args, 'gpu_id': gpu_ids[i % len(gpu_ids)] if gpu_ids else None, 'task_id': i + 1} for i, vd in enumerate(video_dirs)]

    success_count, error_tasks = run_multiprocess_task(process_pose_task, task_args, "提取姿态骨架", args.max_workers)

    if error_tasks:
        error_dirs = [t['video_dir'] for t in error_tasks]
        error_file = os.path.join(args.target_dir, "pose_extraction_errors.txt")
        with open(error_file, 'w') as f: f.writelines(f"{d}\n" for d in error_dirs)
        logger.warning(f"部分目录处理失败，详情已保存到: {error_file}")

    logger.success(f"人体姿态骨架提取完成！成功 {success_count}/{len(video_dirs)} 个目录")
    return True

def process_check_task(video_dir):
    """用于多进程的单个目录数据集完整性检查任务。"""
    try:
        issues = []
        images_dir = os.path.join(video_dir, "images")
        faces_dir = os.path.join(video_dir, "faces")
        poses_dir = os.path.join(video_dir, "poses")

        if not os.path.exists(images_dir):
            issues.append(f"缺少图像目录: {images_dir}")
            return issues

        image_count = len(glob.glob(f"{images_dir}/*.png"))
        if image_count == 0:
            issues.append(f"图像目录为空: {images_dir}")
            return issues

        if not os.path.exists(faces_dir): issues.append(f"缺少人脸目录: {faces_dir}")
        elif len(glob.glob(f"{faces_dir}/*.png")) != image_count:
            issues.append(f"人脸遮罩数量不匹配: {video_dir}")

        if not os.path.exists(poses_dir): issues.append(f"缺少姿态目录: {poses_dir}")
        elif len(glob.glob(f"{poses_dir}/*.png")) != image_count:
            issues.append(f"姿态骨架数量不匹配: {video_dir}")

        return issues
    except Exception as e:
        return [f"检查时出错: {video_dir} - {str(e)}"]

def check_dataset(args):
    """检查数据集的完整性"""
    logger.info("开始检查数据集完整性...")
    video_dirs = get_all_subdirs(args.target_dir)

    if not video_dirs:
        logger.warning("未找到任何视频目录进行检查。")
        return False

    all_issues = []
    with Pool(processes=args.max_workers) as pool, tqdm(total=len(video_dirs), desc="检查数据集") as pbar:
        for issues in pool.imap_unordered(process_check_task, video_dirs):
            if issues: all_issues.extend(issues)
            pbar.update(1)

    if all_issues:
        issues_file = os.path.join(args.target_dir, "dataset_issues.txt")
        with open(issues_file, 'w') as f: f.writelines(f"{issue}\n" for issue in all_issues)
        logger.warning(f"发现 {len(all_issues)} 个问题，详情已保存到: {issues_file}")
        for issue in all_issues[:20]: logger.warning(f"  - {issue}")
        if len(all_issues) > 20: logger.warning(f"  ... 更多问题请查看日志文件。")
        return False
    else:
        logger.success("数据集完整性检查通过！")
        return True

def clean_dataset(args):
    """清理所有处理生成的结果和进度文件"""
    logger.info("开始清理数据集和进度文件...")
    files_to_delete = [
        os.path.join(args.target_dir, "progress_state.pkl"),
        os.path.join(args.target_dir, "video_rec_path.txt"),
        os.path.join(args.target_dir, "video_vec_path.txt"),
        os.path.join(args.target_dir, "face_extraction_errors.txt"),
        os.path.join(args.target_dir, "pose_extraction_errors.txt"),
        os.path.join(args.target_dir, "dataset_issues.txt")
    ]
    for category in ["rec", "vec"]:
        files_to_delete.append(os.path.join(args.target_dir, category, f"{category}_video_mapping.csv"))

    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"已删除文件: {file_path}")
            except Exception as e:
                logger.error(f"删除文件 {file_path} 失败: {str(e)}")

    clean_dirs = get_all_subdirs(args.target_dir)

    if clean_dirs:
        logger.warning(f"将删除 {len(clean_dirs)} 个视频处理目录。")
        if not args.force:
            logger.warning("此操作不可撤销！如需继续，请添加 --force 参数。")
            return False

        for dir_path in clean_dirs:
            try:
                shutil.rmtree(dir_path)
                logger.info(f"已删除目录: {dir_path}")
            except Exception as e:
                logger.error(f"删除目录 {dir_path} 失败: {str(e)}")
        logger.success("数据集清理完成！")
    else:
        logger.info("没有找到需要清理的处理目录。")
    return True

def main():
    """脚本主入口函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description='StableAnimator 数据集准备工具',
        formatter_class=argparse.RawTextHelpFormatter
    )
    # 核心路径参数
    parser.add_argument('--source_dir', type=str, default=os.path.join(script_dir, 'animation_data', 'raw_videos'), help='默认的源视频目录')
    parser.add_argument('--raw_videos_dir', type=str, default=None, help='指定源视频目录 (如果设置，会覆盖 --source_dir)')
    parser.add_argument('--target_dir', type=str, default=os.path.join(script_dir, 'animation_data'), help='目标数据集目录')
    parser.add_argument('--stableanimator_dir', type=str, default=script_dir, help='StableAnimator 代码库的根目录')

    # 处理控制参数
    parser.add_argument('--video_resolution', type=str, choices=['rec', 'vec'], default=None, help='强制指定视频分辨率类型 (rec/vec)。默认为自动判断。')
    parser.add_argument('--fps', type=int, default=16, help='指定抽帧频率 (FPS)。0 表示使用视频原始帧率。默认: 16')
    parser.add_argument('--skip_pose', action='store_true', help='在完整流程中跳过姿态提取步骤')

    # 并发控制参数
    parser.add_argument('--max_workers', type=int, default=min(32, cpu_count()), help='并行处理的最大进程数')
    parser.add_argument('--gpus', type=str, default=None, help='用于处理的GPU ID，用逗号分隔，例如 "0,1,2,3"')

    # 功能模式参数
    parser.add_argument('--extract_frames_only', action='store_true', help='模式：仅提取视频帧')
    parser.add_argument('--extract_faces_only', action='store_true', help='模式：仅提取人脸遮罩')
    parser.add_argument('--extract_poses_only', action='store_true', help='模式：仅提取人体姿态骨架')
    parser.add_argument('--check_dataset', action='store_true', help='模式：检查数据集完整性')
    parser.add_argument('--clean', action='store_true', help='模式：清理所有处理结果和进度文件')

    # 行为修饰参数
    parser.add_argument('--force', action='store_true', help='强制重新生成已存在的文件或执行清理操作')
    parser.add_argument('--resume', action='store_true', default=True, help='从上次中断的地方继续处理 (默认启用)')

    args = parser.parse_args()

    # --- 目录创建 ---
    os.makedirs(os.path.join(args.target_dir, "rec"), exist_ok=True)
    os.makedirs(os.path.join(args.target_dir, "vec"), exist_ok=True)
    os.makedirs(args.source_dir, exist_ok=True)

    # --- 功能分派 ---
    if args.clean:
        if clean_dataset(args):
            is_only_clean = all(arg in ['--clean', '--force', sys.argv[0]] or (arg.startswith('--') and arg.split('=')[0] in ['--clean', '--force']) for arg in sys.argv[1:])
            if is_only_clean:
                logger.info("清理完成，退出程序。")
                return
        else:
            logger.warning("清理操作被取消或失败。")
            return

    logger.info(f"使用 {args.max_workers} 个工作进程进行并发处理。")
    if args.gpus: logger.info(f"使用 GPU: {args.gpus}")

    # 分步执行模式
    if args.check_dataset: check_dataset(args); return
    if args.extract_frames_only: extract_all_frames(args); return
    if args.extract_faces_only: extract_all_faces(args); return
    if args.extract_poses_only: extract_all_poses(args); return

    # --- 完整处理流程 ---
    logger.info("执行完整数据集准备流程...")
    video_source_path = get_video_source_path(args)
    video_files = find_video_files(video_source_path)
    if not video_files: return

    all_rec_dirs, all_vec_dirs, rec_dirs, vec_dirs = process_video_batch(
        args, video_files, extract_mask=True, extract_pose=not args.skip_pose
    )

    logger.success("数据集准备完成！")
    logger.info(f"总 rec 视频: {len(all_rec_dirs)}, 本次新增: {len(rec_dirs)}")
    logger.info(f"总 vec 视频: {len(all_vec_dirs)}, 本次新增: {len(vec_dirs)}")
    check_dataset(args)

def check_dependencies(stableanimator_dir):
    """检查脚本运行所需的外部依赖"""
    logger.info("开始检查依赖项...")
    dependencies_ok = True
    try:
        import loguru, cv2, tqdm, torch
        logger.info(f"loguru: {loguru.__version__}, OpenCV: {cv2.__version__}, tqdm: {tqdm.__version__}, PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"检测到 {torch.cuda.device_count()} 个可用 GPU。")
        else:
            logger.warning("未检测到可用的 GPU，人脸/姿态提取将使用 CPU。")
    except ImportError as e:
        logger.error(f"缺少必要的 Python 包: {e.name}。请运行 'pip install -r requirements.txt'")
        dependencies_ok = False

    for tool in ["ffmpeg", "ffprobe"]:
        try:
            result = subprocess.run([tool, "-version"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                logger.info(f"{tool} 已安装。")
            else:
                logger.warning(f"警告: {tool} 可能未正确安装或不在PATH中。")
        except FileNotFoundError:
            logger.error(f"错误: 未找到 {tool}。请确保它已安装并位于系统的 PATH 中。")
            dependencies_ok = False

    return dependencies_ok

def print_help_message():
    """打印详细的脚本使用指南"""
    help_text = """
StableAnimator 数据集准备工具使用指南:
-----------------------------------
1. 完整处理流程 (默认从 ./animation_data/raw_videos/ 读取视频):
   python data_preprocess.py

2. 从指定目录读取视频:
   python data_preprocess.py --raw_videos_dir /path/to/my/videos

3. 强制指定视频分辨率类型:
   python data_preprocess.py --video_resolution=rec  (强制所有视频为 512x512 类别)
   python data_preprocess.py --video_resolution=vec  (强制所有视频为 576x1024 类别)

4. 分步执行:
   python data_preprocess.py --extract_frames_only
   python data_preprocess.py --extract_faces_only
   python data_preprocess.py --extract_poses_only

5. 其他常用参数:
   --fps=8                  (以8fps抽帧)
   --skip_pose              (在完整流程中跳过姿态提取)
   --max_workers=8          (使用8个进程)
   --gpus=0,1               (使用GPU 0和1)
   --force                  (强制重新处理所有内容)
   --check_dataset          (检查数据集完整性)
   --clean --force          (清理所有处理结果)
"""
    print(help_text)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info("--- StableAnimator 数据集准备工具 ---")

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--stableanimator_dir', type=str, default=script_dir)
    pre_args, _ = pre_parser.parse_known_args()

    os.makedirs(os.path.join(script_dir, 'logs'), exist_ok=True)

    if not check_dependencies(pre_args.stableanimator_dir):
        sys.exit(1)

    if len(sys.argv) == 1 or '--help' in sys.argv or '-h' in sys.argv:
        print_help_message()
        if len(sys.argv) == 1:
            logger.info("\n提示: 未指定任何操作参数，将默认执行完整处理流程...\n")

    def signal_handler(sig, frame):
        logger.warning("\n检测到中断信号 (Ctrl+C)，正在尝试优雅地退出...")
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)

    start_time = time.time()
    try:
        main()
    except (KeyboardInterrupt, SystemExit) as e:
        if isinstance(e, SystemExit) and e.code == 0:
             pass
        else:
            logger.warning("处理被用户中断。")
    except Exception as e:
        logger.error(f"处理过程中发生严重错误: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        logger.success(f"总耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
