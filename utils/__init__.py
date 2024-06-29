from openai import OpenAI
import os
import json
import time
import functools
import requests
import cv2
import numpy as np
from PIL import Image
from loguru import logger
from io import BytesIO
from contextlib import contextmanager

cache_data_dir = os.environ.get("CACHE_DATA_DIR", "./tmp/")


def generate_random_dir():
    template_id = str(time.time()).replace(".", "")
    _save_dir = os.path.join(cache_data_dir, str(template_id))
    os.makedirs(_save_dir, exist_ok=True)
    return _save_dir


@contextmanager
def safe_dir():
    _save_dir = generate_random_dir()
    try:
        yield _save_dir
        if bool(os.environ.get("MOVE_SAFE_DIR", True)):
            os.system(f"rm -rf {_save_dir}")
    except Exception as e:
        raise e


def logger_execute_time(doc="执行时间"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """计算方法执行时间，并输出执行时间"""
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"{doc}: {execution_time}")
            return result

        return wrapper

    return decorator


def pretty_print_dict(doc="参数如下", **kwargs):
    placeholder = "*" * 100
    s = f"\n{placeholder}\n{doc}:\n"
    for k, v in kwargs.items():
        s += f"\n{k}: {v}"
    s += f"\n{placeholder}"
    logger.info(s)


class VideoProcessor:
    def read_video(self, video_path: str):
        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            raise Exception("无法打开视频文件")
        return video

    @logger_execute_time(doc="间隔抽帧")
    def extract_frames_by_interval(self, video_path: str, _dir: str, interval=25, start_frame=0, end_frame=2**1024):
        # 初始化帧计数器和图像计数器
        frame_count = 0
        image_count = 0
        messages = []
        video = self.read_video(video_path)
        while True:
            success, frame = video.read()
            frame_count += 1
            if not success:
                break
            if frame_count < start_frame:
                continue
            if frame_count > end_frame:
                break
            # 仅处理每隔frame_interval帧
            if frame_count % interval == 0:
                image_path = os.path.join(_dir, f"{frame_count}.jpg")
                cv2.imwrite(image_path, frame)
                messages.append({"image": image_path})
                image_count += 1

        video.release()
        return messages