# -*- encoding: utf-8 -*-
# here put the import lib
import os

import cv2
import numpy as np

def is_video_file(file_path):
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv']  # 常见的视频文件扩展名
    file_extension = os.path.splitext(file_path)[1].lower()  # 获取文件的扩展名并转为小写
    return file_extension in video_extensions


def video_to_spike(
    sourefolder='F:\\tmp\\citystreet01', 
    savefolder_debug=None, 
    threshold=5.0,
    init_noise=True,
    format="png",
    is_video=False,
    ):
    """
        函数说明
        :param 参数名: 参数说明
        :return: 返回值名: 返回值说明
    """

    if is_video_file(sourefolder):
        is_video = True
        video_capture = cv2.VideoCapture(sourefolder)  # 替换为你的视频文件路径

        # 检查视频是否成功打开
        if not video_capture.isOpened():
            print("cannot open the video")
            exit()

        # 获取视频的帧率和分辨率
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 初始化一个空的 NumPy 数组来保存视频帧数据
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        datas = np.zeros((frame_count, height, width, 3), dtype=np.uint8)

        # 逐帧读取视频并保存到 NumPy 数组中
        frame_idx = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            datas[frame_idx] = frame
            frame_idx += 1

        # 释放视频捕捉对象
        video_capture.release()
        frame0 = datas[0]

    else:
        filelist = sorted(os.listdir(sourefolder))
        datas = [fn for fn in filelist if fn.endswith(format)]
        frame0 = cv2.imread(os.path.join(sourefolder, datas[0]))

    T = len(datas)

    H, W, C = frame0.shape

    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    spikematrix = np.zeros([T, H, W], np.uint8)

    if init_noise:
        integral = np.random.random(size=([H,W])) * threshold
    else:
        integral = np.random.zeros(size=([H,W]))
    
    Thr = np.ones_like(integral).astype(np.float32) * threshold

    for t in range(0, T):
        if is_video:
            frame = datas[t]
        else:
            frame = cv2.imread(os.path.join(sourefolder, datas[t]))

        if C > 1:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = gray / 255.0
        integral += gray
        fire = (integral - Thr) >= 0
        fire_pos = fire.nonzero()
        
        integral[fire_pos] -= threshold
        spikematrix[t][fire_pos] = 1

    if savefolder_debug:
        np.save(os.path.join(savefolder_debug, "spike_debug.npy"), spikematrix)

    return spikematrix
