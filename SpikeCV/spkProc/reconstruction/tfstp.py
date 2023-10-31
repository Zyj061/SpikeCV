# -*- coding: utf-8 -*- 
# @Time : 2022/7/16 19:47 
# @Author : Yajing Zheng
# @File : tfstp.py
from spkProc.filters.stp_filters_torch import STPFilter
import numpy as np
import torch
import cv2


class TFSTP:

    def __init__(self, spike_h, spike_w, device):
        self.spike_h = spike_h
        self.spike_w = spike_w
        self.device = device

        # parameters of the STP model
        stpPara = {}
        stpPara['u0'] = 0.15
        stpPara['D'] = 0.05 * 20
        stpPara['F'] = 0.5 * 20
        stpPara['f'] = 0.15

        stpPara['time_unit'] = 1

        self.stp_filter = STPFilter(self.spike_h, self.spike_w, device, **stpPara)

    @staticmethod
    def spike2interval(spikes):
        '''
        transform the Vidar spikes to inter-spike-intervals
        :param spikes: Vidar Spikes
        :return: inter-spike-interval, ISI
        '''
        timestamps = spikes.shape[0]
        spike_h = spikes.shape[1]
        spike_w = spikes.shape[2]
        isi = np.ones([timestamps, spike_h, spike_w]) * np.inf

        for i in range(spike_h):
            for j in range(spike_w):
                tmp_spk = spikes[:, i, j]
                spike_idx = np.array(np.nonzero(tmp_spk))
                if spike_idx.shape[1] > 1:
                    tmp_interval = spike_idx[0, 1:] - spike_idx[0, 0:-1]
                    for k in range(len(tmp_interval)):
                        isi[spike_idx[0, k] + 1: spike_idx[0, k + 1] + 1, i, j] = tmp_interval[k]

        return isi

    def spikes2images_online(self, spikes):
        '''
        High-speed image reconstruction at each timestamp with only previous input spikes.
        :param spikes: Vidar spike streams
        :return: ImageMatrix, reconstructing high-speed videos
        '''

        timestamps = spikes.shape[0]
        ImageMatrix = np.zeros([timestamps, self.spike_h, self.spike_w], np.uint8)

        for t in range(timestamps):
            input_spk = torch.from_numpy(spikes[t, :, :]).to(self.device)
            self.stp_filter.update_dynamics(t, input_spk)

            rho_u = -1 / (self.stp_filter.F * torch.log((self.stp_filter.u - self.stp_filter.u0) /
                                                        (self.stp_filter.F - self.stp_filter.u0 +
                                                         self.stp_filter.u * (1 - self.stp_filter.f))))

            rho_R = -1 / (self.stp_filter.D * torch.log((1 - self.stp_filter.R) /
                                                        (1 - self.stp_filter.R * (1 - self.stp_filter.u))))

            image = rho_u + rho_R

            image = image.cpu().detach().numpy()
            image = (image - image.min()) / (image.max() - image.min()) * 255
            ImageMatrix[t, :, :] = cv2.equalizeHist(image.astype(np.uint8))

        return ImageMatrix

    def spikes2images_offline(self, spikes):
        '''
        high-speed reconstruction using the whole spike stream, which will firstly obtaining
        the inter-spike-interval (ISI) firstly
        :param spikes: Vidar Spikes
        :return: ImageMatrix
        '''

        timestamps = spikes.shape[0]
        ImageMatrix = np.zeros([timestamps, self.spike_h, self.spike_w], np.uint8)
        intervals = self.spike2interval(spikes)

        R, u = self.stp_filter.update_dynamic_offline(spikes, intervals)
        for t in range(timestamps):

            rho_u = -1 / (self.stp_filter.F * torch.log((u[t, :, :] - self.stp_filter.u0) /
                                                        (self.stp_filter.F - self.stp_filter.u0 +
                                                         u[t, :, :] * (1 - self.stp_filter.f))))

            rho_R = -1 / (self.stp_filter.D * torch.log((1 - R[t, :, :]) /
                                                        (1 - R[t, :, :] * (1 - u[t, :, :]))))

            image = rho_u + rho_R

            image = image.cpu().detach().numpy()
            if image.max() != image.min():
                image = (image - image.min()) / (image.max() - image.min()) * 255

            ImageMatrix[t, :, :] = cv2.equalizeHist(image.astype(np.uint8))

        return ImageMatrix
