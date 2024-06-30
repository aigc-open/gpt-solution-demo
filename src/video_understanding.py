import cv2
import requests
import os, time
import numpy as np
from PIL import Image
from loguru import logger
from io import BytesIO
from GeneralAgent import Agent
import google.generativeai as genai
from utils import logger_execute_time, pretty_print_dict, safe_dir, VideoProcessor


class VideoUnderstandingByGPT4o(VideoProcessor):

    def __init__(self):
        self.client = Agent('You are a helpful agent.',
                            api_key=os.environ['OPENAI_API_KEY'],
                            base_url=os.environ['OPENAI_BASE_URL'],
                            model="gpt-4o",
                            disable_python_run=True)

    @logger_execute_time(doc="视频理解")
    def run(self, video_path: str, interval=25, prompt="", start_frame=0, end_frame=2**1024):
        pretty_print_dict(
            doc="参数如下", **{"间隔帧": interval, "提示词": prompt, "视频起始帧": start_frame, "视频结束帧": end_frame})
        messages = [prompt]
        with safe_dir() as _dir:
            messages_ = self.extract_frames_by_interval(
                video_path=video_path, interval=interval, _dir=_dir, start_frame=start_frame, end_frame=end_frame)  # 抽帧
            messages.extend(messages_)
            out = self.client.user_input(messages)
        return out


class VideoUnderstandingByGemini(VideoUnderstandingByGPT4o):
    def __init__(self):
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        self.client = genai.GenerativeModel(os.environ['GOOGLE_MODEL'])

    @logger_execute_time(doc="单个文件上传")
    def upload_file(self, filename):
        file_ = genai.upload_file(path=filename)
        while file_.state.name == "PROCESSING":
            time.sleep(10)
            file_ = genai.get_file(file_.name)

        if file_.state.name == "FAILED":
            raise ValueError(file_.state.name)
        return file_

    @logger_execute_time(doc="所有文件上传")
    def batch_upload_file(self, filenames):
        out = []
        for filename in filenames:
            file_ = self.upload_file(filename["image"])
            out.append(file_)
        return out

    @logger_execute_time(doc="视频理解")
    def run(self, video_path: str, interval=25, prompt="", mode="image"):
        with safe_dir(remove=False) as _dir:
            if mode == "image":
                filenames = self.extract_frames_by_interval(
                    video_path=video_path, interval=interval, _dir=_dir)  # 抽帧
                all_files = self.batch_upload_file(filenames=filenames)
            elif mode == "video":
                all_files = [self.upload_file(filename=video_path)]
            else:
                raise Exception("不支持的模式")
            messages = all_files
            messages.append(prompt)
            response = self.client.generate_content(messages,
                                                    request_options={"timeout": 600})
            out = response.candidates[0].content.parts[0].text
        return out


class BadmintonCompetitionByGPT4o(VideoUnderstandingByGPT4o):
    """羽毛球比赛发球积分识别"""

    def __init__(self):
        self.client = Agent('You are a helpful agent.',
                            api_key=os.environ['OPENAI_API_KEY'],
                            base_url=os.environ['OPENAI_BASE_URL'],
                            model=os.environ['OPENAI_MODEL'],
                            disable_python_run=True)
        self.client_sumary = Agent('You are a helpful agent.',
                                   api_key=os.environ['OPENAI_API_KEY'],
                                   base_url=os.environ['OPENAI_BASE_URL'],
                                   model=os.environ['OPENAI_SUMARY_MODEL'],
                                   disable_python_run=True)
        self.fps = 30
        self.sleep_time = 5
        self.skip_times = 1

    def group(self, data, group_size=4):
        out = [data[i:i+group_size] for i in range(0, len(data), group_size)]
        logger.info(f"一共有序列{len(out)}组")
        return out

    @logger_execute_time(doc="视频理解")
    def run(self, video_path: str,
            interval=25,
            predict_prompt="",
            sumary_prompt="",
            start_frame=0,
            end_frame=2**1024,
            group_size=10,
            fps=30,
            sleep_time=5,
            skip_times=1,
            role=["甲", "乙"]
            ):
        self.fps = fps
        self.sleep_time = sleep_time
        self.skip_times = skip_times
        pretty_print_dict(doc="参数如下", **{"间隔帧": interval, "提示词": predict_prompt, "视频帧率": fps, "GPT4o间隔时间": sleep_time,
                   "检测到发球方后跳过帧检测次数": skip_times, "视频起始帧": start_frame, "视频结束帧": f"{end_frame}({end_frame/fps}s)", "分组大小": group_size})
        result = []
        with safe_dir() as _dir:
            messages_ = self.extract_frames_by_interval(
                video_path=video_path, interval=interval, _dir=_dir, start_frame=start_frame, end_frame=end_frame)  # 抽帧
            skip = False
            skip_times = 1
            last_segment_messages_ = []
            for segment_messages_ in self.group(messages_, group_size=group_size):
                last_segment_messages_ = segment_messages_
                segment_messages_.extend(last_segment_messages_)
                if skip is True:
                    if self.skip_times <= 0:
                        skip = False
                        pass
                    else:
                        logger.info(f"跳过本预测帧: {segment_messages_}")
                        skip_times += 1
                        if skip_times <= self.skip_times:
                            continue
                        skip_times = 1
                        skip = False
                        continue
                if len(segment_messages_) < group_size:
                    logger.info(
                        f"跳过本预测帧: {len(segment_messages_)} {segment_messages_}")
                    continue
                logger.info(f"本次预测帧: {segment_messages_}")
                total_seconds, minutes, seconds = self.cal_video_time_by_tps(segment_messages_[
                                                                             0]["image"], interval=interval, seq_len=len(segment_messages_))
                
                out = self.image_predict(
                    image_messages=segment_messages_, prompt=predict_prompt)
                out = self.sumary(text=out, prompt=sumary_prompt)
                if out.strip() in role:
                    skip = True
                logger.info(
                    f"时间{total_seconds}s({minutes}:{seconds})处: 发球方: {out}")
                result.append({"winer": out.strip(
                ), "total_seconds": total_seconds, "time": f"{minutes}:{seconds}"})
                time.sleep(self.sleep_time)
        return result

    def cal_video_time_by_tps(self, file_path, interval, seq_len):
        """根据序列计算帧所在时间"""
        # 提取文件名部分
        file_name_with_extension = os.path.basename(file_path)
        # 分离文件名和扩展名
        file_name, file_extension = os.path.splitext(file_name_with_extension)
        total_seconds = (int(file_name)+interval*seq_len//2) // self.fps
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return total_seconds, minutes, seconds

    @logger_execute_time(doc="图片序列预测")
    def image_predict(self, image_messages: list, prompt: str):
        self.client.clear()
        image_messages.insert(0, prompt)
        return self.client.user_input(image_messages)

    @logger_execute_time(doc="结果总结")
    def sumary(self, text, prompt=""):
        messages_ = [f"## 背景知识\n```{text}```\n"+prompt]
        self.client_sumary.clear()
        out = self.client_sumary.user_input(messages_)
        return out