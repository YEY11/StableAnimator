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
from multiprocessing import Pool, Manager, cpu_count
from tqdm import tqdm
import pickle
import hashlib
import signal
import traceback
from loguru import logger

# 配置 loguru 日志
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add("./logs/data_preprocess.log", rotation="10 MB", retention="1 week")

# 文件夹编号格式设置
FOLDER_NUM_FORMAT = "{:05d}"  # 5位数，如：00001, 00002, ...


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


def extract_face_masks(stableanimator_dir, image_folder, video_dir, gpu_id=None):
    """提取人脸遮罩"""
    try:
        current_dir = os.getcwd()
        os.chdir(stableanimator_dir)
        
        # 创建保存人脸遮罩的目录
        faces_dir = os.path.join(video_dir, "faces")
        os.makedirs(faces_dir, exist_ok=True)
        
        # 设置环境变量以选择 GPU
        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        cmd = [
            "python", "face_mask_extraction.py",
            f"--image_folder={image_folder}"
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        
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
        try:
            os.chdir(current_dir)
        except:
            pass
        return False


def extract_poses_for_video(stableanimator_dir, video_dir, gpu_id=None):
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
        
        # 设置环境变量以选择 GPU
        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # 使用 training_skeleton_extraction.py 代替 extract_video_keypoints.py
        cmd = [
            "python", "DWPose/training_skeleton_extraction.py",
            f"--root_path={os.path.dirname(os.path.dirname(video_dir))}",
            f"--name={category}",
            f"--start={video_number}",
            f"--end={video_number}"
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        
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


def read_mapping_csv(csv_path):
    """读取映射 CSV 文件"""
    mappings = {}
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r') as f:
                # 跳过标题行
                next(f, None)
                for line in f:
                    parts = line.strip().split(',', 1)
                    if len(parts) == 2:
                        folder_name, video_name = parts
                        mappings[video_name] = folder_name
        except Exception as e:
            logger.error(f"读取映射文件 {csv_path} 失败: {str(e)}")
    return mappings


def write_mapping_csv(csv_path, mappings):
    """写入映射 CSV 文件，确保按文件夹名称排序"""
    try:
        # 反转映射，使得视频名称作为值，文件夹名称作为键
        folder_to_video = {}
        for video_name, folder_name in mappings.items():
            folder_to_video[folder_name] = video_name
        
        # 按文件夹名称排序
        sorted_folders = sorted(folder_to_video.keys())
        
        with open(csv_path, 'w') as f:
            f.write("folder_name,original_video_name\n")
            for folder_name in sorted_folders:
                video_name = folder_to_video[folder_name]
                f.write(f"{folder_name},{video_name}\n")
        return True
    except Exception as e:
        logger.error(f"写入映射文件 {csv_path} 失败: {str(e)}")
        return False


def update_video_mapping_csv(target_dir, category, folder_num, video_path, lock):
    """更新视频映射 CSV 文件，使用锁保证进程安全
    
    Args:
        target_dir: 目标目录基础路径
        category: 分类 ('rec' 或 'vec')
        folder_num: 文件夹编号
        video_path: 原始视频路径
        lock: 进程锁
    """
    # CSV 文件路径
    csv_path = os.path.join(target_dir, category, f"{category}_video_mapping.csv")
    
    # 获取原始视频文件名
    video_name = os.path.basename(video_path)
    
    # 使用锁确保进程安全
    with lock:
        # 读取现有映射
        mappings = read_mapping_csv(csv_path)
        
        # 添加或更新映射
        mappings[video_name] = folder_num
        
        # 写入映射
        if write_mapping_csv(csv_path, mappings):
            logger.info(f"已更新视频映射: {folder_num} -> {video_name}")
            return True
        else:
            logger.error(f"更新视频映射失败: {folder_num} -> {video_name}")
            return False


def update_path_file(target_dir, category, dirs, lock):
    """更新路径文件以保存进度，使用锁保证进程安全"""
    path_file = os.path.join(target_dir, f"video_{category}_path.txt")
    
    # 使用锁确保进程安全
    with lock:
        with open(path_file, 'w') as f:
            for dir_path in dirs:
                f.write(f"{dir_path}\n")


def save_progress_state(args, rec_dirs, vec_dirs, processed_videos):
    """保存处理进度状态"""
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
                'skip_pose': args.skip_pose
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
    """加载处理进度状态"""
    progress_file = os.path.join(args.target_dir, "progress_state.pkl")
    
    if not os.path.exists(progress_file):
        logger.info("未找到进度状态文件，将从头开始处理")
        return None
    
    try:
        with open(progress_file, 'rb') as f:
            state = pickle.load(f)
        
        # 检查状态文件是否与当前参数兼容
        if (state['args']['target_dir'] != args.target_dir or 
            state['args']['video_resolution'] != args.video_resolution):
            logger.warning("进度状态文件与当前参数不匹配，将从头开始处理")
            return None
        
        timestamp = state.get('timestamp', 0)
        time_diff = time.time() - timestamp
        hours, remainder = divmod(time_diff, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        logger.info(f"加载进度状态: 上次运行于 {int(hours)}小时 {int(minutes)}分钟前")
        logger.info(f"已处理视频数量: {len(state['processed_videos'])}")
        
        return state
    except Exception as e:
        logger.error(f"加载进度状态失败: {str(e)}")
        return None


def generate_video_hash(video_path):
    """生成视频文件的哈希值，用于唯一标识"""
    try:
        # 使用文件名、大小和修改时间创建哈希
        file_stat = os.stat(video_path)
        video_name = os.path.basename(video_path)
        hash_str = f"{video_name}_{file_stat.st_size}_{file_stat.st_mtime}"
        return hashlib.md5(hash_str.encode()).hexdigest()
    except Exception as e:
        logger.error(f"生成视频哈希失败 {video_path}: {str(e)}")
        # 如果失败，使用文件名作为备选
        return os.path.basename(video_path)


def process_video_task(args_dict):
    """处理单个视频的任务函数，用于多进程池"""
    # 解包参数
    video_path = args_dict['video_path']
    args = args_dict['args']
    rec_count = args_dict['rec_count']
    vec_count = args_dict['vec_count']
    extract_mask = args_dict['extract_mask']
    extract_pose = args_dict['extract_pose']
    forced_type = args_dict['forced_type']
    gpu_id = args_dict['gpu_id']
    shared_dict = args_dict['shared_dict']
    task_id = args_dict.get('task_id', 0)
    
    # 生成视频哈希用于标识
    video_hash = generate_video_hash(video_path)
    
    try:
        # 检查是否已处理过该视频
        with shared_dict['lock']:
            if video_hash in shared_dict['processed_hashes']:
                logger.info(f"跳过已处理的视频: {os.path.basename(video_path)}")
                return None
            # 标记为正在处理
            shared_dict['in_progress_hashes'][video_hash] = 1
        
        video_name = os.path.basename(video_path).rsplit('.', 1)[0]
        logger.info(f"处理视频 [{task_id}]: {video_name}")
        
        # 获取视频分辨率
        width, height = get_video_resolution(video_path)
        if width == 0 or height == 0:
            logger.warning(f"跳过视频 {video_path} (无法获取分辨率)")
            return None
        
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
        
        # 设置文件夹编号 - 使用预分配的编号
        with shared_dict['lock']:
            if video_path in shared_dict['video_to_folder']:
                folder_num = shared_dict['video_to_folder'][video_path]
            else:
                # 如果没有预分配（可能是后来添加的视频），则使用旧逻辑，但确保从1开始
                if is_rec:
                    folder_num = FOLDER_NUM_FORMAT.format(max(1, rec_count + len(shared_dict['rec_dirs'])))
                else:
                    folder_num = FOLDER_NUM_FORMAT.format(max(1, vec_count + len(shared_dict['vec_dirs'])))
        
        # 创建目标目录 - 确保每个视频有唯一的目录
        video_dir = os.path.join(args.target_dir, target_type, folder_num)
        
        # 检查目录是否已存在，如果存在则递增编号
        counter = 1
        original_folder_num = folder_num
        
        with shared_dict['lock']:
            while os.path.exists(video_dir) and os.path.basename(video_dir) != "raw_videos":
                folder_num = FOLDER_NUM_FORMAT.format(int(folder_num) + 1)
                video_dir = os.path.join(args.target_dir, target_type, folder_num)
                counter += 1
                if counter > 1000:  # 防止无限循环
                    logger.error(f"错误: 无法为 {video_path} 创建唯一目录")
                    return None
            
            # 创建目录结构
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
            return None
        
        # 检查是否成功提取了足够的帧
        frame_count = len(glob.glob(f"{images_dir}/*.png"))
        if frame_count < 5:  # 设置一个最小帧数阈值
            logger.warning(f"跳过视频 {video_path} (提取的帧数不足: {frame_count})")
            # 清理目录
            try:
                shutil.rmtree(video_dir)
            except:
                pass
            return None
        
        # 提取人脸遮罩
        if extract_mask:
            logger.info(f"提取人脸遮罩... {images_dir}")
            if not extract_face_masks(args.stableanimator_dir, images_dir, video_dir, gpu_id):
                logger.warning(f"人脸遮罩提取失败: {video_dir}")
        
        # 提取姿态骨架
        if extract_pose and not args.skip_pose:
            logger.info(f"提取姿态骨架... {video_dir}")
            if not extract_poses_for_video(args.stableanimator_dir, video_dir, gpu_id):
                logger.warning(f"姿态骨架提取失败: {video_dir}")
        
        logger.success(f"已完成 {video_path} 的处理")
        
        # 更新视频映射 CSV 文件
        csv_path = os.path.join(args.target_dir, target_type, f"{target_type}_video_mapping.csv")
        with shared_dict['lock']:
            # 读取现有映射
            mappings = read_mapping_csv(csv_path)
            
            # 添加或更新映射
            video_filename = os.path.basename(video_path)
            mappings[video_filename] = folder_num
            
            # 写入映射
            if write_mapping_csv(csv_path, mappings):
                logger.info(f"已更新视频映射: {folder_num} -> {video_filename}")
            else:
                logger.error(f"更新视频映射失败: {folder_num} -> {video_filename}")
        
        # 标记为已处理
        with shared_dict['lock']:
            shared_dict['processed_hashes'][video_hash] = 1
            if video_hash in shared_dict['in_progress_hashes']:
                del shared_dict['in_progress_hashes'][video_hash]
        
        # 返回处理结果
        result = {
            'type': target_type,
            'dir': video_dir,
            'is_rec': is_rec,
            'video_path': video_path,
            'video_hash': video_hash
        }
        
        # 定期保存进度
        with shared_dict['lock']:
            shared_dict['completed_count'] += 1
            if shared_dict['completed_count'] % 10 == 0:  # 每处理10个视频保存一次进度
                # 更新路径文件
                if is_rec:
                    shared_dict['rec_dirs'].append(video_dir)
                    rec_dirs = list(shared_dict['rec_dirs'])
                    vec_dirs = list(shared_dict['vec_dirs'])
                else:
                    shared_dict['vec_dirs'].append(video_dir)
                    rec_dirs = list(shared_dict['rec_dirs'])
                    vec_dirs = list(shared_dict['vec_dirs'])
                
                # 保存进度状态
                processed_hashes = list(shared_dict['processed_hashes'].keys())
                save_progress_state(args, rec_dirs, vec_dirs, processed_hashes)
        
        return result
    
    except Exception as e:
        logger.error(f"处理视频 {video_path} 时出错: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 从进行中列表移除
        with shared_dict['lock']:
            if video_hash in shared_dict['in_progress_hashes']:
                del shared_dict['in_progress_hashes'][video_hash]
        
        return None


def process_video_batch(args, video_files, extract_mask=True, extract_pose=True):
    """使用多进程并行处理多个视频文件"""
    # 获取已存在的目录
    existing_rec_dirs = sorted(glob.glob(os.path.join(args.target_dir, "rec", "*")))
    existing_vec_dirs = sorted(glob.glob(os.path.join(args.target_dir, "vec", "*")))
    
    # 过滤掉 raw_videos 目录
    existing_rec_dirs = [d for d in existing_rec_dirs if os.path.basename(d) != "raw_videos"]
    existing_vec_dirs = [d for d in existing_vec_dirs if os.path.basename(d) != "raw_videos"]
    
    # 处理计数器 - 从现有目录数量开始，但至少从1开始
    rec_count = max(1, len(existing_rec_dirs))
    vec_count = max(1, len(existing_vec_dirs))
    
    # 创建进程安全的共享对象
    manager = Manager()
    shared_dict = manager.dict()
    shared_dict['lock'] = manager.Lock()
    shared_dict['rec_dirs'] = manager.list()
    shared_dict['vec_dirs'] = manager.list()
    shared_dict['processed_hashes'] = manager.dict()  # 已处理视频的哈希字典
    shared_dict['in_progress_hashes'] = manager.dict()  # 正在处理的视频哈希字典
    shared_dict['completed_count'] = 0  # 已完成处理的视频数量
    
    # 预分配文件夹编号 - 这是确保顺序一致的关键
    shared_dict['video_to_folder'] = manager.dict()  # 存储视频路径到文件夹编号的映射
    
    # 按文件名排序视频
    sorted_videos = sorted(video_files, key=lambda x: os.path.basename(x))
    
    # 预分配文件夹编号，从1开始
    rec_next_num = max(1, rec_count)  # 确保至少从1开始
    vec_next_num = max(1, vec_count)  # 确保至少从1开始
    for video_path in sorted_videos:
        # 获取视频分辨率
        width, height = get_video_resolution(video_path)
        if width == 0 or height == 0:
            continue  # 跳过无法获取分辨率的视频
            
        # 确定视频类型
        forced_type = None
        if args.video_resolution == 'rec':
            forced_type = 'rec'
            is_rec = True
        elif args.video_resolution == 'vec':
            forced_type = 'vec'
            is_rec = False
        else:
            # 基于分辨率自动确定类型
            if width <= 512 and height <= 512:
                is_rec = True
            else:
                is_rec = False
                
        # 分配编号
        if is_rec:
            folder_num = FOLDER_NUM_FORMAT.format(rec_next_num)
            rec_next_num += 1
        else:
            folder_num = FOLDER_NUM_FORMAT.format(vec_next_num)
            vec_next_num += 1
            
        # 存储分配的编号
        shared_dict['video_to_folder'][video_path] = folder_num
    
    # 尝试加载之前的进度
    state = load_progress_state(args)
    if state:
        # 恢复之前的进度
        for dir_path in state['rec_dirs']:
            if dir_path not in existing_rec_dirs and os.path.exists(dir_path):
                shared_dict['rec_dirs'].append(dir_path)
        
        for dir_path in state['vec_dirs']:
            if dir_path not in existing_vec_dirs and os.path.exists(dir_path):
                shared_dict['vec_dirs'].append(dir_path)
        
        for video_hash in state['processed_videos']:
            shared_dict['processed_hashes'][video_hash] = 1
    
    # 检查现有的映射文件，避免重复处理
    processed_video_names = set()
    
    for category in ["rec", "vec"]:
        mapping_file = os.path.join(args.target_dir, category, f"{category}_video_mapping.csv")
        if os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r') as f:
                    # 跳过标题行
                    next(f, None)
                    for line in f:
                        parts = line.strip().split(',', 1)
                        if len(parts) > 1:
                            # 将原始视频文件名添加到已处理集合中
                            processed_video_names.add(parts[1].strip())
            except Exception as e:
                logger.warning(f"读取映射文件 {mapping_file} 时出错: {str(e)}")
    
    # 为每个视频生成哈希
    video_hashes = {generate_video_hash(v): v for v in video_files}
    
    # 过滤掉已处理的视频
    already_processed = set(shared_dict['processed_hashes'].keys())
    video_files_to_process = [v for h, v in video_hashes.items() if h not in already_processed]
    
    # 根据文件名也过滤一次
    video_files_to_process = [v for v in video_files_to_process if os.path.basename(v) not in processed_video_names]
    
    # 保持排序
    video_files_to_process = sorted(video_files_to_process, key=lambda x: os.path.basename(x))
    
    logger.info(f"总视频文件数: {len(video_files)}, 待处理视频文件数: {len(video_files_to_process)}")
    
    # 在这里添加额外检查
    if len(video_files_to_process) == 0:
        # 检查是否存在实际的输出目录
        total_existing_dirs = len(existing_rec_dirs) + len(existing_vec_dirs)
        if total_existing_dirs == 0:
            logger.warning("所有视频标记为已处理，但未找到处理结果。可能需要使用 --force 参数重新处理")
            # 如果用户指定了 --force 参数或没有找到处理结果，则重置处理状态
            if args.force:
                logger.info("重置处理状态，将重新处理所有视频")
                shared_dict['processed_hashes'].clear()
                video_files_to_process = sorted(video_files, key=lambda x: os.path.basename(x))
            else:
                logger.info("所有视频已处理，无需再次处理")
                return existing_rec_dirs + list(shared_dict['rec_dirs']), existing_vec_dirs + list(shared_dict['vec_dirs']), [], []
        else:
            logger.info("所有视频已处理，无需再次处理")
            return existing_rec_dirs + list(shared_dict['rec_dirs']), existing_vec_dirs + list(shared_dict['vec_dirs']), [], []
    
    # 分配 GPU ID（如果有多个 GPU）
    gpu_ids = []
    if args.gpus:
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(',')]
    
    # 准备任务参数
    task_args = []
    for i, video_path in enumerate(video_files_to_process):
        # 根据命令行参数强制设置视频类型
        forced_type = None
        if args.video_resolution == 'rec':
            forced_type = 'rec'
        elif args.video_resolution == 'vec':
            forced_type = 'vec'
        
        # 如果有 GPU，循环分配
        gpu_id = None
        if gpu_ids:
            gpu_id = gpu_ids[i % len(gpu_ids)]
        
        task_args.append({
            'video_path': video_path,
            'args': args,
            'rec_count': rec_count,
            'vec_count': vec_count,
            'extract_mask': extract_mask,
            'extract_pose': extract_pose,
            'forced_type': forced_type,
            'gpu_id': gpu_id,
            'shared_dict': shared_dict,
            'task_id': i + 1  # 任务ID，从1开始
        })
    
    # 设置信号处理，优雅地处理中断
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    # 使用进程池处理视频
    rec_dirs = []
    vec_dirs = []
    processed_results = []
    
    try:
        # 创建进度条
        with tqdm(total=len(task_args), desc="处理视频") as pbar:
            # 使用进程池
            pool = Pool(processes=args.max_workers)
            
            # 恢复信号处理
            signal.signal(signal.SIGINT, original_sigint_handler)
            
            # 使用 imap_unordered 以获得更好的负载均衡
            for result in pool.imap_unordered(process_video_task, task_args):
                if result:
                    processed_results.append(result)
                    if result['is_rec']:
                        rec_dirs.append(result['dir'])
                    else:
                        vec_dirs.append(result['dir'])
                
                pbar.update(1)
                
                # 每处理10个视频保存一次中间结果
                if len(processed_results) % 10 == 0:
                    # 更新路径文件
                    with shared_dict['lock']:
                        update_path_file(args.target_dir, "rec", existing_rec_dirs + rec_dirs, shared_dict['lock'])
                        update_path_file(args.target_dir, "vec", existing_vec_dirs + vec_dirs, shared_dict['lock'])
    
    except KeyboardInterrupt:
        logger.warning("检测到中断信号，正在优雅地停止处理...")
        pool.terminate()
        pool.join()
        
        # 保存当前进度
        logger.info("保存处理进度...")
        all_rec_dirs = existing_rec_dirs + rec_dirs
        all_vec_dirs = existing_vec_dirs + vec_dirs
        
        # 更新路径文件
        update_path_file(args.target_dir, "rec", all_rec_dirs, shared_dict['lock'])
        update_path_file(args.target_dir, "vec", all_vec_dirs, shared_dict['lock'])
        
        # 保存进度状态
        processed_hashes = [result['video_hash'] for result in processed_results if 'video_hash' in result]
        save_progress_state(args, all_rec_dirs, all_vec_dirs, processed_hashes)
        
        logger.info("处理已中断，但进度已保存。您可以稍后继续处理。")
        return all_rec_dirs, all_vec_dirs, rec_dirs, vec_dirs
    
    except Exception as e:
        logger.error(f"处理视频批次时出错: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 尝试关闭进程池
        try:
            pool.terminate()
            pool.join()
        except:
            pass
        
        # 保存当前进度
        logger.info("尝试在错误后保存进度...")
        all_rec_dirs = existing_rec_dirs + rec_dirs
        all_vec_dirs = existing_vec_dirs + vec_dirs
        
        # 更新路径文件
        update_path_file(args.target_dir, "rec", all_rec_dirs, shared_dict['lock'])
        update_path_file(args.target_dir, "vec", all_vec_dirs, shared_dict['lock'])
        
        # 保存进度状态
        processed_hashes = [result['video_hash'] for result in processed_results if 'video_hash' in result]
        save_progress_state(args, all_rec_dirs, all_vec_dirs, processed_hashes)
        
        logger.info("处理出错，但部分进度已保存。请修复错误后继续处理。")
        return all_rec_dirs, all_vec_dirs, rec_dirs, vec_dirs
    
    finally:
        # 恢复原始信号处理
        signal.signal(signal.SIGINT, original_sigint_handler)
        
        # 确保进程池正确关闭
        try:
            pool.close()
            pool.join()
        except:
            pass
    
    # 转换为普通列表
    all_rec_dirs = existing_rec_dirs + rec_dirs
    all_vec_dirs = existing_vec_dirs + vec_dirs
    
    # 最终更新一次路径文件
    update_path_file(args.target_dir, "rec", all_rec_dirs, shared_dict['lock'])
    update_path_file(args.target_dir, "vec", all_vec_dirs, shared_dict['lock'])
    
    # 保存最终进度
    processed_hashes = [result['video_hash'] for result in processed_results if 'video_hash' in result]
    save_progress_state(args, all_rec_dirs, all_vec_dirs, processed_hashes)
    
    return all_rec_dirs, all_vec_dirs, rec_dirs, vec_dirs


def extract_all_frames(args):
    """仅提取所有视频的帧，使用多进程处理"""
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
    all_rec_dirs, all_vec_dirs, rec_dirs, vec_dirs = process_video_batch(
        args, video_files, extract_mask=False, extract_pose=False
    )
    
    logger.success("帧提取完成！")
    logger.info(f"总 rec 视频数量: {len(all_rec_dirs)}")
    logger.info(f"总 vec 视频数量: {len(all_vec_dirs)}")
    logger.info(f"新增 rec 视频数量: {len(rec_dirs)}")
    logger.info(f"新增 vec 视频数量: {len(vec_dirs)}")
    
    return True


def process_face_mask_task(args_dict):
    """处理单个目录的人脸遮罩提取任务"""
    video_dir = args_dict['video_dir']
    args = args_dict['args']
    gpu_id = args_dict['gpu_id']
    task_id = args_dict.get('task_id', 0)
    
    try:
        images_dir = os.path.join(video_dir, "images")
        faces_dir = os.path.join(video_dir, "faces")
        
        if not os.path.exists(images_dir):
            logger.warning(f"警告: {video_dir} 中没有找到图像目录，请先运行抽帧")
            return False
        
        # 检查是否需要提取（如果已经存在足够的遮罩文件且不强制重新生成）
        if os.path.exists(faces_dir) and not args.force:
            image_count = len(glob.glob(f"{images_dir}/*.png"))
            face_count = len(glob.glob(f"{faces_dir}/*.png"))
            
            if face_count >= image_count:
                logger.info(f"跳过已完成的人脸遮罩: {video_dir} (已有: {face_count}/{image_count})")
                return True
        
        # 清空现有的人脸遮罩目录
        if args.force and os.path.exists(faces_dir):
            shutil.rmtree(faces_dir)
        
        # 提取人脸遮罩
        logger.info(f"处理人脸遮罩 [{task_id}]: {os.path.basename(video_dir)}")
        return extract_face_masks(args.stableanimator_dir, images_dir, video_dir, gpu_id)
    
    except Exception as e:
        logger.error(f"处理人脸遮罩时出错 {video_dir}: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def extract_all_faces(args):
    """仅提取所有帧的人脸遮罩，使用多进程处理"""
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
    
    # 分配 GPU ID（如果有多个 GPU）
    gpu_ids = []
    if args.gpus:
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(',')]
    
    # 准备任务参数
    task_args = []
    for i, video_dir in enumerate(video_dirs):
        # 如果有 GPU，循环分配
        gpu_id = None
        if gpu_ids:
            gpu_id = gpu_ids[i % len(gpu_ids)]
        
        task_args.append({
            'video_dir': video_dir,
            'args': args,
            'gpu_id': gpu_id,
            'task_id': i + 1
        })
    
    # 设置信号处理
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    success_count = 0
    error_dirs = []
    
    try:
        # 创建进度条
        with tqdm(total=len(task_args), desc="提取人脸遮罩") as pbar:
            # 使用进程池
            pool = Pool(processes=args.max_workers)
            
            # 恢复信号处理
            signal.signal(signal.SIGINT, original_sigint_handler)
            
            # 处理任务
            for i, result in enumerate(pool.imap_unordered(process_face_mask_task, task_args)):
                if result:
                    success_count += 1
                else:
                    error_dirs.append(task_args[i]['video_dir'])
                
                pbar.update(1)
    
    except KeyboardInterrupt:
        logger.warning("检测到中断信号，正在优雅地停止处理...")
        pool.terminate()
        pool.join()
        
        # 保存未完成的目录列表
        error_file = os.path.join(args.target_dir, "face_extraction_errors.txt")
        with open(error_file, 'w') as f:
            for dir_path in error_dirs:
                f.write(f"{dir_path}\n")
        
        logger.info(f"处理已中断。已完成 {success_count}/{total_dirs} 个目录。")
        logger.info(f"未完成的目录已保存到: {error_file}")
        return False
    
    except Exception as e:
        logger.error(f"提取人脸遮罩时出错: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 尝试关闭进程池
        try:
            pool.terminate()
            pool.join()
        except:
            pass
        
        return False
    
    finally:
        # 恢复原始信号处理
        signal.signal(signal.SIGINT, original_sigint_handler)
        
        # 确保进程池正确关闭
        try:
            pool.close()
            pool.join()
        except:
            pass
    
    # 如果有错误，保存错误目录列表
    if error_dirs:
        error_file = os.path.join(args.target_dir, "face_extraction_errors.txt")
        with open(error_file, 'w') as f:
            for dir_path in error_dirs:
                f.write(f"{dir_path}\n")
        
        logger.warning(f"部分目录处理失败，详情已保存到: {error_file}")
    
    logger.success(f"人脸遮罩提取完成！成功处理 {success_count}/{total_dirs} 个目录")
    return True


def process_pose_task(args_dict):
    """处理单个目录的姿态骨架提取任务"""
    video_dir = args_dict['video_dir']
    args = args_dict['args']
    gpu_id = args_dict['gpu_id']
    task_id = args_dict.get('task_id', 0)
    
    try:
        images_dir = os.path.join(video_dir, "images")
        poses_dir = os.path.join(video_dir, "poses")
        
        if not os.path.exists(images_dir):
            logger.warning(f"警告: {video_dir} 中没有找到图像目录，请先运行抽帧")
            return False
        
        # 检查是否需要提取（如果已经存在足够的姿态文件且不强制重新生成）
        if os.path.exists(poses_dir) and not args.force:
            image_count = len(glob.glob(f"{images_dir}/*.png"))
            pose_count = len(glob.glob(f"{poses_dir}/*.png"))
            
            if pose_count >= image_count:
                logger.info(f"跳过已完成的姿态骨架: {video_dir} (已有: {pose_count}/{image_count})")
                return True
        
        # 清空现有的姿态骨架目录
        if args.force and os.path.exists(poses_dir):
            shutil.rmtree(poses_dir)
        
        # 提取姿态骨架
        logger.info(f"处理姿态骨架 [{task_id}]: {os.path.basename(video_dir)}")
        return extract_poses_for_video(args.stableanimator_dir, video_dir, gpu_id)
    
    except Exception as e:
        logger.error(f"处理姿态骨架时出错 {video_dir}: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def extract_all_poses(args):
    """仅提取所有帧的人体姿态骨架，使用多进程处理"""
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
    
    # 分配 GPU ID（如果有多个 GPU）
    gpu_ids = []
    if args.gpus:
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(',')]
    
    # 准备任务参数
    task_args = []
    for i, video_dir in enumerate(video_dirs):
        # 如果有 GPU，循环分配
        gpu_id = None
        if gpu_ids:
            gpu_id = gpu_ids[i % len(gpu_ids)]
        
        task_args.append({
            'video_dir': video_dir,
            'args': args,
            'gpu_id': gpu_id,
            'task_id': i + 1
        })
    
    # 设置信号处理
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    success_count = 0
    error_dirs = []
    
    try:
        # 创建进度条
        with tqdm(total=len(task_args), desc="提取姿态骨架") as pbar:
            # 使用进程池
            pool = Pool(processes=args.max_workers)
            
            # 恢复信号处理
            signal.signal(signal.SIGINT, original_sigint_handler)
            
            # 处理任务
            for i, result in enumerate(pool.imap_unordered(process_pose_task, task_args)):
                if result:
                    success_count += 1
                else:
                    error_dirs.append(task_args[i]['video_dir'])
                
                pbar.update(1)
    
    except KeyboardInterrupt:
        logger.warning("检测到中断信号，正在优雅地停止处理...")
        pool.terminate()
        pool.join()
        
        # 保存未完成的目录列表
        error_file = os.path.join(args.target_dir, "pose_extraction_errors.txt")
        with open(error_file, 'w') as f:
            for dir_path in error_dirs:
                f.write(f"{dir_path}\n")
        
        logger.info(f"处理已中断。已完成 {success_count}/{total_dirs} 个目录。")
        logger.info(f"未完成的目录已保存到: {error_file}")
        return False
    
    except Exception as e:
        logger.error(f"提取姿态骨架时出错: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 尝试关闭进程池
        try:
            pool.terminate()
            pool.join()
        except:
            pass
        
        return False
    
    finally:
        # 恢复原始信号处理
        signal.signal(signal.SIGINT, original_sigint_handler)
        
        # 确保进程池正确关闭
        try:
            pool.close()
            pool.join()
        except:
            pass
    
    # 如果有错误，保存错误目录列表
    if error_dirs:
        error_file = os.path.join(args.target_dir, "pose_extraction_errors.txt")
        with open(error_file, 'w') as f:
            for dir_path in error_dirs:
                f.write(f"{dir_path}\n")
        
        logger.warning(f"部分目录处理失败，详情已保存到: {error_file}")
    
    logger.success(f"人体姿态骨架提取完成！成功处理 {success_count}/{total_dirs} 个目录")
    return True


def process_check_task(args_dict):
    """检查单个目录的数据集完整性"""
    video_dir = args_dict['video_dir']
    
    try:
        issues = []
        images_dir = os.path.join(video_dir, "images")
        faces_dir = os.path.join(video_dir, "faces")
        poses_dir = os.path.join(video_dir, "poses")
        
        # 尝试查找可能的替代目录（处理不同的数字格式）
        if not os.path.exists(images_dir):
            # 检查是否有文件夹名格式问题
            dir_name = os.path.basename(video_dir)
            parent_dir = os.path.dirname(video_dir)
            
            # 尝试不同的格式
            if dir_name.isdigit():
                # 如果是数字格式的文件夹名
                num_value = int(dir_name)
                
                # 尝试不同位数的格式
                possible_formats = [
                    FOLDER_NUM_FORMAT.format(num_value),  # 使用全局格式
                    f"{num_value:04d}",  # 4位数
                    f"{num_value:05d}",  # 5位数
                    f"{num_value}",      # 无前导零
                ]
                
                for fmt in possible_formats:
                    if fmt != dir_name:  # 跳过当前目录名
                        alt_dir = os.path.join(parent_dir, fmt)
                        alt_images_dir = os.path.join(alt_dir, "images")
                        if os.path.exists(alt_images_dir):
                            logger.debug(f"找到替代图像目录: {alt_images_dir} (替代 {images_dir})")
                            images_dir = alt_images_dir
                            faces_dir = os.path.join(alt_dir, "faces")
                            poses_dir = os.path.join(alt_dir, "poses")
                            break
        
        # 检查目录是否存在
        if not os.path.exists(images_dir):
            issues.append(f"缺少图像目录: {images_dir}")
            return issues
        
        if not os.path.exists(faces_dir):
            issues.append(f"缺少人脸目录: {faces_dir}")
        
        if not os.path.exists(poses_dir):
            issues.append(f"缺少姿态目录: {poses_dir}")
        
        # 检查文件数量
        image_files = glob.glob(f"{images_dir}/*.png")
        image_count = len(image_files)
        
        if image_count == 0:
            issues.append(f"图像目录为空: {images_dir}")
            return issues
        
        if os.path.exists(faces_dir):
            face_count = len(glob.glob(f"{faces_dir}/*.png"))
            if face_count != image_count:
                issues.append(f"人脸遮罩数量不匹配: {video_dir} (图像: {image_count}, 人脸: {face_count})")
        
        if os.path.exists(poses_dir):
            pose_count = len(glob.glob(f"{poses_dir}/*.png"))
            if pose_count != image_count:
                issues.append(f"姿态骨架数量不匹配: {video_dir} (图像: {image_count}, 姿态: {pose_count})")
        
        return issues
    
    except Exception as e:
        logger.error(f"检查目录 {video_dir} 时出错: {str(e)}")
        return [f"检查时出错: {video_dir} - {str(e)}"]


def check_dataset(args):
    """检查数据集的完整性，使用多进程处理"""
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
    
    # 准备任务参数
    task_args = [{'video_dir': video_dir} for video_dir in video_dirs]
    
    # 设置信号处理
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    all_issues = []
    
    try:
        # 创建进度条
        with tqdm(total=len(task_args), desc="检查数据集") as pbar:
            # 使用进程池
            pool = Pool(processes=args.max_workers)
            
            # 恢复信号处理
            signal.signal(signal.SIGINT, original_sigint_handler)
            
            # 处理任务
            for issues in pool.imap_unordered(process_check_task, task_args):
                if issues:
                    all_issues.extend(issues)
                pbar.update(1)
    
    except KeyboardInterrupt:
        logger.warning("检测到中断信号，正在优雅地停止检查...")
        pool.terminate()
        pool.join()
        
        # 保存已发现的问题
        if all_issues:
            issues_file = os.path.join(args.target_dir, "dataset_issues.txt")
            with open(issues_file, 'w') as f:
                for issue in all_issues:
                    f.write(f"{issue}\n")
            
            logger.info(f"检查已中断。已发现 {len(all_issues)} 个问题，详情已保存到: {issues_file}")
        else:
            logger.info("检查已中断。尚未发现问题。")
        
        return False
    
    except Exception as e:
        logger.error(f"检查数据集时出错: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 尝试关闭进程池
        try:
            pool.terminate()
            pool.join()
        except:
            pass
        
        return False
    
    finally:
        # 恢复原始信号处理
        signal.signal(signal.SIGINT, original_sigint_handler)
        
        # 确保进程池正确关闭
        try:
            pool.close()
            pool.join()
        except:
            pass
    
    # 打印问题
    if all_issues:
        # 保存所有问题到文件
        issues_file = os.path.join(args.target_dir, "dataset_issues.txt")
        with open(issues_file, 'w') as f:
            for issue in all_issues:
                f.write(f"{issue}\n")
        
        logger.warning(f"发现 {len(all_issues)} 个问题，详情已保存到: {issues_file}")
        
        # 在控制台显示部分问题
        max_display = min(20, len(all_issues))
        for i in range(max_display):
            logger.warning(f"  - {all_issues[i]}")
        
        if len(all_issues) > max_display:
            logger.warning(f"  ... 还有 {len(all_issues) - max_display} 个问题 (见 {issues_file})")
        
        logger.info("\n可能的解决方案:")
        logger.info(f"  - 提取缺失的人脸遮罩: python {sys.argv[0]} --extract_faces_only")
        logger.info(f"  - 提取缺失的姿态骨架: python {sys.argv[0]} --extract_poses_only")
        logger.info(f"  - 强制重新提取人脸遮罩: python {sys.argv[0]} --extract_faces_only --force")
        return False
    else:
        logger.success("数据集完整性检查通过！")
        return True


def clean_dataset(args):
    """清理所有处理结果和进度文件，恢复到处理前的状态"""
    logger.info("开始清理数据集和进度文件...")
    
    # 要清理的目标路径
    progress_file = os.path.join(args.target_dir, "progress_state.pkl")
    rec_path_file = os.path.join(args.target_dir, "video_rec_path.txt")
    vec_path_file = os.path.join(args.target_dir, "video_vec_path.txt")
    rec_mapping_file = os.path.join(args.target_dir, "rec", "rec_video_mapping.csv")
    vec_mapping_file = os.path.join(args.target_dir, "vec", "vec_video_mapping.csv")
    face_errors_file = os.path.join(args.target_dir, "face_extraction_errors.txt")
    pose_errors_file = os.path.join(args.target_dir, "pose_extraction_errors.txt")
    dataset_issues_file = os.path.join(args.target_dir, "dataset_issues.txt")
    
    # 删除进度文件
    files_to_delete = [
        progress_file, rec_path_file, vec_path_file,
        rec_mapping_file, vec_mapping_file,
        face_errors_file, pose_errors_file, dataset_issues_file
    ]
    
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"已删除文件: {file_path}")
            except Exception as e:
                logger.error(f"删除文件 {file_path} 失败: {str(e)}")
    
    # 删除处理结果目录
    clean_dirs = []
    for category in ["rec", "vec"]:
        category_dir = os.path.join(args.target_dir, category)
        if os.path.exists(category_dir):
            for folder in os.listdir(category_dir):
                folder_path = os.path.join(category_dir, folder)
                # 保留原始视频目录
                if os.path.isdir(folder_path) and folder != "raw_videos":
                    clean_dirs.append(folder_path)
    
    # 确认清理
    if clean_dirs:
        logger.warning(f"将删除 {len(clean_dirs)} 个视频处理目录。")
        
        if not args.force:
            logger.warning("此操作不可撤销！如需继续，请添加 --force 参数。")
            return False
        
        # 删除处理目录
        for dir_path in clean_dirs:
            try:
                shutil.rmtree(dir_path)
                logger.info(f"已删除目录: {dir_path}")
            except Exception as e:
                logger.error(f"删除目录 {dir_path} 失败: {str(e)}")
        
        logger.success("数据集清理完成！")
        return True
    else:
        logger.info("没有找到需要清理的处理目录。")
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
    parser.add_argument('--max_workers', type=int, default=min(32, cpu_count()),
                        help='并行处理的最大进程数 (默认使用所有CPU核心，但不超过32)')
    parser.add_argument('--gpus', type=str, default=None,
                        help='用于处理的GPU ID，用逗号分隔，例如 "0,1,2,3" (默认使用所有可见GPU)')
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
    parser.add_argument('--resume', action='store_true',
                        help='从上次中断的地方继续处理')
    parser.add_argument('--retry_failed', action='store_true',
                        help='重试之前失败的任务')
    parser.add_argument('--clean', action='store_true',
                        help='清理所有处理结果和进度文件，恢复到处理前的状态')
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
    
    # 如果指定了 clean 参数，先执行清理
    if args.clean:
        if clean_dataset(args):
            if len(sys.argv) == 2 or (len(sys.argv) == 3 and args.force):  # 如果只有 --clean 或 --clean --force 参数，清理完成后退出
                logger.info("清理完成，退出程序")
                return
        else:
            logger.warning("清理操作被取消或失败")
            return
    
    # 输出并发设置信息
    logger.info(f"使用 {args.max_workers} 个工作进程进行并发处理")
    if args.gpus:
        logger.info(f"使用 GPU: {args.gpus}")
    
    # 根据参数执行不同的功能
    if args.check_dataset:
        logger.info("检查数据集完整性...")
        check_dataset(args)
        return
    
    if args.retry_failed:
        if args.extract_faces_only:
            logger.info("重试失败的人脸遮罩提取任务...")
            error_file = os.path.join(args.target_dir, "face_extraction_errors.txt")
            if os.path.exists(error_file):
                with open(error_file, 'r') as f:
                    error_dirs = [line.strip() for line in f if line.strip()]
                
                if error_dirs:
                    logger.info(f"找到 {len(error_dirs)} 个失败的目录，开始重试...")
                    
                    # 准备参数
                    task_args = []
                    gpu_ids = []
                    if args.gpus:
                        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(',')]
                    
                    for i, video_dir in enumerate(error_dirs):
                        gpu_id = None
                        if gpu_ids:
                            gpu_id = gpu_ids[i % len(gpu_ids)]
                        
                        task_args.append({
                            'video_dir': video_dir,
                            'args': args,
                            'gpu_id': gpu_id,
                            'task_id': i + 1
                        })
                    
                    # 创建进度条
                    success_count = 0
                    with tqdm(total=len(task_args), desc="重试人脸遮罩提取") as pbar:
                        with Pool(processes=args.max_workers) as pool:
                            for result in pool.imap_unordered(process_face_mask_task, task_args):
                                if result:
                                    success_count += 1
                                pbar.update(1)
                    
                    logger.success(f"重试完成！成功修复 {success_count}/{len(error_dirs)} 个目录")
                else:
                    logger.info("没有找到失败的任务记录")
            else:
                logger.warning(f"未找到失败记录文件: {error_file}")
            return
        
        elif args.extract_poses_only:
            logger.info("重试失败的姿态骨架提取任务...")
            error_file = os.path.join(args.target_dir, "pose_extraction_errors.txt")
            if os.path.exists(error_file):
                with open(error_file, 'r') as f:
                    error_dirs = [line.strip() for line in f if line.strip()]
                
                if error_dirs:
                    logger.info(f"找到 {len(error_dirs)} 个失败的目录，开始重试...")
                    
                    # 准备参数
                    task_args = []
                    gpu_ids = []
                    if args.gpus:
                        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpus.split(',')]
                    
                    for i, video_dir in enumerate(error_dirs):
                        gpu_id = None
                        if gpu_ids:
                            gpu_id = gpu_ids[i % len(gpu_ids)]
                        
                        task_args.append({
                            'video_dir': video_dir,
                            'args': args,
                            'gpu_id': gpu_id,
                            'task_id': i + 1
                        })
                    
                    # 创建进度条
                    success_count = 0
                    with tqdm(total=len(task_args), desc="重试姿态骨架提取") as pbar:
                        with Pool(processes=args.max_workers) as pool:
                            for result in pool.imap_unordered(process_pose_task, task_args):
                                if result:
                                    success_count += 1
                                pbar.update(1)
                    
                    logger.success(f"重试完成！成功修复 {success_count}/{len(error_dirs)} 个目录")
                else:
                    logger.info("没有找到失败的任务记录")
            else:
                logger.warning(f"未找到失败记录文件: {error_file}")
            return
        
        else:
            logger.warning("--retry_failed 选项需要与 --extract_faces_only 或 --extract_poses_only 一起使用")
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
    all_rec_dirs, all_vec_dirs, rec_dirs, vec_dirs = process_video_batch(
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
        
        # 检查 GPU 状态
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
                logger.info(f"检测到 {gpu_count} 个 GPU:")
                for i, name in enumerate(gpu_names):
                    logger.info(f"  GPU {i}: {name}")
            else:
                logger.warning("警告: 未检测到可用的 GPU，将使用 CPU 处理")
        except ImportError:
            logger.warning("警告: 未安装 PyTorch，无法检测 GPU 状态")
        
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
   python data_preprocess.py
   这将执行抽帧、人脸遮罩提取和姿态骨架提取的完整流程

2. 指定视频分辨率类型:
   python data_preprocess.py --video_resolution=rec
   强制将所有视频处理为 rec 类型 (512x512)
   
   python data_preprocess.py --video_resolution=vec
   强制将所有视频处理为 vec 类型 (576x1024)

3. 仅提取视频帧:
   python data_preprocess.py --extract_frames_only
   从视频文件提取帧并创建数据集目录结构

4. 仅提取人脸遮罩:
   python data_preprocess.py --extract_faces_only
   为已提取的帧生成人脸遮罩

5. 仅提取姿态骨架:
   python data_preprocess.py --extract_poses_only
   为已提取的帧生成姿态骨架

6. 检查数据集完整性:
   python data_preprocess.py --check_dataset
   检查数据集中是否有缺失的文件或目录

7. 强制重新生成:
   python data_preprocess.py --extract_faces_only --force
   强制重新生成已存在的人脸遮罩

8. 跳过姿态提取:
   python data_preprocess.py --skip_pose
   执行完整流程但跳过姿态提取步骤

9. 指定原始视频目录:
   python data_preprocess.py --raw_videos_dir=/path/to/videos
   从指定目录读取原始视频文件

10. 控制并发处理:
    python data_preprocess.py --max_workers=8
    使用 8 个进程并行处理视频 (默认使用所有CPU核心，但不超过32)

11. 指定使用的 GPU:
    python data_preprocess.py --gpus=0,1
    使用 GPU 0 和 1 进行处理 (多个 GPU 用逗号分隔)

12. 从中断处继续:
    python data_preprocess.py --resume
    从上次中断的地方继续处理

13. 重试失败的任务:
    python data_preprocess.py --extract_faces_only --retry_failed
    重试之前失败的人脸遮罩提取任务

    python data_preprocess.py --extract_poses_only --retry_failed
    重试之前失败的姿态骨架提取任务
    
14. 清理所有处理结果:
    python data_preprocess.py --clean
    删除所有处理结果和进度文件，恢复到处理前的状态
    
    python data_preprocess.py --clean --force
    强制清理，不需要确认

15. 清理后立即开始处理:
    python data_preprocess.py --clean --force [其他参数]
    清理所有结果后立即开始处理
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
    parser.add_argument('--gpus', type=str, default=None,
                        help='用于处理的GPU ID，用逗号分隔，例如 "0,1,2,3" (默认使用所有可见GPU)')
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
    
    # 注册信号处理函数，用于优雅地处理 Ctrl+C
    def signal_handler(sig, frame):
        logger.warning("\n检测到中断信号 (Ctrl+C)，正在尝试优雅地退出...")
        logger.warning("请稍等，正在保存当前进度...")
        # 让主程序处理中断
        sys.exit(1)
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        start_time = time.time()
        main()
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.success(f"总耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
    except KeyboardInterrupt:
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.warning(f"处理被用户中断。运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
        sys.exit(1)
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        
        elapsed_time = time.time() - start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.error(f"处理失败。运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
        sys.exit(1)

