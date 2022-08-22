# -*- coding: utf-8 -*- 
# @Time : 2022/7/21
# @Author : Rui Zhao
# @File : tfi.py
import numpy as np
import torch
import cv2


class TFI:
    def __init__(self, spike_h, spike_w, device):
        self.spike_h = spike_h
        self.spike_w = spike_w
        self.device = device


    def spikes2images(self, spikes, max_search_half_window=20):
        '''
        Texture From Interval (TFI) 算法
        将spikes整体转换为一段由TFI算法重构的图像
        
        输入：
        spikes: T x H x W 的numpy张量, 类型: 整型与浮点皆可
        max_search_half_window: 对于要转换为图像的时刻点而言, 左右各参考的最大脉冲帧数量，超过这个数字就不搜了

        输出：
        ImageMatrix: T' x H x W 的numpy张量, 其中T' = T - (2 x max_search_half_window)
        类型: uint8, 取值范围: 0 ~ 255
        '''

        T = spikes.shape[0]
        T_im = T - 2*max_search_half_window

        if T_im < 0:
            raise ValueError('The length of spike stream {:d} is not enough for max_search half window length {:d}'.format(T, max_search_half_window))
        
        spikes = torch.from_numpy(spikes).to(self.device).float()
        ImageMatrix = torch.zeros([T_im, self.spike_h, self.spike_w]).to(self.device)

        pre_idx = -1 * torch.ones([T, self.spike_h, self.spike_w]).float().to(self.device)
        cur_idx = -1 * torch.ones([T, self.spike_h, self.spike_w]).float().to(self.device)
        
        for ii in range(T):
            if ii > 0:
                pre_idx[ii] = cur_idx[ii-1]
                cur_idx[ii] = cur_idx[ii-1]
            cur_spk = spikes[ii]
            cur_idx[ii][cur_spk==1] = ii

        diff = cur_idx - pre_idx

        interval = -1 * torch.ones([T, self.spike_h, self.spike_w]).float().to(self.device)
        for ii in range(T-1, 0-1, -1):
            interval[ii][diff[ii]!=0] = diff[ii][diff[ii]!=0]
            if ii < T-1:
                interval[ii][diff[ii]==0] = interval[ii+1][diff[ii]==0]
        
        # boundary
        interval[interval==-1] = 257
        interval[pre_idx==-1] = 257
        ImageMatrix_uncrop = 255 / interval
        ImageMatrix = ImageMatrix_uncrop[max_search_half_window:-max_search_half_window].cpu().detach().numpy().astype(np.uint8)
        

        return ImageMatrix

    
    def spikes2frame(self, spikes, key_ts, max_search_half_window=20):
        '''
        Texture From Interval (TFI) 算法
        从spikes中获取某一帧TFI图像
        
        输入：
        spikes: T x H x W 的numpy张量, 类型: 整型与浮点皆可
        key_ts: 要获取的图像在脉冲中的时间戳
        max_search_half_window: 对于要转换为图像的时刻点而言, 左右各参考的最大脉冲帧数量，超过这个数字就不搜了

        输出：
        Image: H x W的numpy张量, 类型: uint8, 取值范围: 0 ~ 255
        '''

        T = spikes.shape[0]

        if (key_ts - max_search_half_window // 2 < 0) or (key_ts + max_search_half_window // 2 > T):
            raise ValueError('The length of spike stream {:d} is not enough for even max_search half window length {:d} // 2 = {:d} at key time stamp {:d}'.format(T, max_search_half_window, max_search_half_window//2, key_ts))
        
        spikes = torch.from_numpy(spikes).to(self.device).float()

        formmer_index = torch.zeros([self.spike_h, self.spike_w]).to(self.device)
        latter_index = torch.zeros([self.spike_h, self.spike_w]).to(self.device)

        start_t = max(key_ts - max_search_half_window + 1, 1)
        end_t = min(key_ts + max_search_half_window, T)

        for ii in range(key_ts, start_t-1, -1):
            formmer_index += ii * spikes[ii, :, :] * (1 - torch.sign(formmer_index).to(self.device))

        for ii in range(key_ts+1, end_t+1):
            latter_index += ii * spikes[ii, :, :] * (1 - torch.sign(latter_index).to(self.device))

        interval = latter_index - formmer_index
        interval[interval == 0] = 2*max_search_half_window
        interval[latter_index == 0] = 2*max_search_half_window
        interval[formmer_index == 0] = 2*max_search_half_window
        interval = interval

        Image = 255 / interval
        Image = Image.cpu().detach().numpy().astype(np.uint8)

        return Image
