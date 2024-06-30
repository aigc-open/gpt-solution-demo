from src.video_understanding import BadmintonCompetitionByGPT4o
from utils import pretty_print_dict

predict_prompt = """
下面的图片序列是一个羽毛球比赛。请你仔细观察这些图片动作。合理分析, 注意可能存在发球直接失误
## 我们约定:
- 近处是甲方，远处是乙方

## 问题:
图片序列有4种情况: 甲乙都在比赛、甲方在发球、乙方在发球、一方把球送给对方、休息。
请分析图片内容，最终得到是哪一种情况。
"""

sumary_prompt = """
## 请你根据以上内容的结论直接回答谁在发球
- 如果没有发球就说没有，送球，双方都在比赛，休息也不算发球，这一类直接返回没有, 
- 返回格式要求：甲/乙/没有/
"""


def run(video_path="A.mp4",
        interval=12, end_frame=800,
        start_frame=0,
        group_size=10,
        skip_times=0,
        config=["乙", "甲", "沒有"]
        ):
    config.extend(["未知"]*100)
    res:list = BadmintonCompetitionByGPT4o().run(video_path=video_path, interval=interval, predict_prompt=predict_prompt,
                                            sumary_prompt=sumary_prompt, end_frame=end_frame,
                                            start_frame=start_frame,
                                            group_size=group_size, skip_times=skip_times)
    data = {}
    for idx, value in enumerate(res):
        if value["winer"] == config[idx]:
            data[f"結果{idx}"] = f"預測值: {} 期望值: {} 測試： pass"
        else:
            data[f"結果{idx}"] = f"預測值: {} 期望值: {} 測試： error"
    pretty_print_dict(doc="結果", **data)


if __name__ == "__main__":
    run(video_path="A.mp4", interval=12, end_frame=800, group_size=10, config=["乙", "甲", "沒有"])