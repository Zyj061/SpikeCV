# -*- coding: utf-8 -*- 
# @Time : 2022/7/21
# @Author : Rui Zhao
# @File : tfp.py
import numpy as np
import torch
import cv2


class TFP:
    def __init__(self, spike_h, spike_w, device):
        self.spike_h = spike_h
        self.spike_w = spike_w
        self.device = device


    def spikes2images(self, spikes, half_win_length):
        '''
        Texture From Playback (TFP) 算法
        将spikes整体转换为一段由TFP算法重构的图像
        
        输入：
        spikes: T x H x W 的numpy张量, 类型: 整型与浮点皆可
        half_win_length: 对于要转换为图像的时刻点而言, 左右各参考的脉冲帧数量

        输出：
        ImageMatrix: T' x H x W 的numpy张量, 其中T' = T - (2 x half_win_length)
        类型: uint8, 取值范围: 0 ~ 255
        '''

        T = spikes.shape[0]
        T_im = T - 2*half_win_length

        if T_im < 0:
            raise ValueError('The length of spike stream {:d} is not enough for half window length {:d}'.format(T, half_win_length))
        
        spikes = torch.from_numpy(spikes).to(self.device).float()
        ImageMatrix = torch.zeros([T_im, self.spike_h, self.spike_w]).to(self.device)

        for ts in range(half_win_length, T-half_win_length):
            ImageMatrix[ts - half_win_length] = spikes[ts-half_win_length : ts+half_win_length+1].mean(dim=0) * 255

        ImageMatrix = ImageMatrix.cpu().detach().numpy().astype(np.uint8)

        return ImageMatrix


    def spikes2frame(self, spikes, key_ts, half_win_length):
        '''
        Texture From Playback (TFP) 算法
        从spikes中获取某一帧TFP图像
        
        输入：
        spikes: T x H x W 的numpy张量, 类型: 整型与浮点皆可
        half_win_length: 对于要转换为图像的时刻点而言, 左右各参考的脉冲帧数量
        key_ts: 要获取的图像在脉冲中的时间戳

        输出：
        Image: H x W的numpy张量, 类型: uint8, 取值范围: 0 ~ 255
        '''

        T = spikes.shape[0]

        if (key_id - half_win_length < 0) or (key_id + half_win_length > T):
            raise ValueError('The length of spike stream {:d} is not enough for half window length {:d} at key time stamp {:d}'.format(T, half_win_length, key_ts))
        
        spikes = spikes[key_ts - half_win_length : key_ts + half_win_length + 1]
        spikes = torch.from_numpy(sipkes).to(self.device).float()

        Image = spikes.mean(dim=0) * 255
        Image = Image.cpu().detach().numpy().astype(np.uint8)

        return Image
