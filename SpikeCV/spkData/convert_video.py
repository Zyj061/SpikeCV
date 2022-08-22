# -*- encoding: utf-8 -*-
'''
@File    :   convert_video.py
@Time    :   2022/07/25 23:48:57
@Author  :   Jiyuan Zhang 
@Version :   0.0
@Contact :   jyzhang@stu.pku.edu.cn
'''

# here put the import lib
import os

import cv2
import numpy as np


def video_to_spike(
    sourefolder='F:\\tmp\\citystreet01', 
    savefolder_debug=None, 
    threshold=5.0,
    init_noise=True,
    format="png",
    ):
    """
        函数说明
        :param 参数名: 参数说明
        :return: 返回值名: 返回值说明
    """

    filelist = sorted(os.listdir(sourefolder))
    datas = [fn for fn in filelist if fn.endswith(format)]
    
    T = len(datas)
    
    frame0 = cv2.imread(os.path.join(sourefolder, datas[0]))
    H, W, C = frame0.shape

    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    spikematrix = np.zeros([T, H, W], np.uint8)

    if init_noise:
        integral = np.random.random(size=([H,W])) * threshold
    else:
        integral = np.random.zeros(size=([H,W]))
    
    Thr = np.ones_like(integral).astype(np.float32) * threshold

    for t in range(0, T):
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
