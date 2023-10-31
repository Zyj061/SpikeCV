# -*- coding: utf-8 -*- 
# @Time : 2021/11/19 16:25 
# @Author : Yajing Zheng
# @File : stp_filters.py
import numpy as np
from scipy import signal


class STPFilter:

    def __init__(self, spike_h, spike_w):
        self.self.spike_h = spike_h
        self.self.spike_w = spike_w

        # stp parameters
        self.u0 = 0.1
        self.D = 0.02
        self.F = 1.7
        self.f = 0.11

        self.r0 = 1
        self.filterThr = 0.1  # filter threshold

        self.r_old = np.ones(self.spike_h, self.spike_w) * self.r0
        self.u_old = np.ones(self.spike_h, self.spike_w) * self.u0
        self.r_tmp = np.ones(self.spike_h, self.spike_w) * self.r0

        # LIF detect layer parameters
        self.detectVoltage = np.zeros(self.spike_h, self.spike_w)
        self.lifConv_weight = np.ones(3, 3) * 3.0

        self.voltageMin = -8
        self.lifThr = 2

        self.filter_spk = np.zeros(self.spike_h, self.spike_w)
        self.lif_spk = np.zeros(self.spike_h, self.spike_w)
        self.spikePrevMnt = np.zeros(self.spike_h, self.spike_w)

    def update_dynamics(self, curT, spikes):

        spikeCurMnt = self.spikePrevMnt
        spikeCurMnt[spikes] = curT + 1
        dttimes = spikeCurMnt - self.spikePrevMnt
        dttimes = dttimes / 2000
        exp_D = np.exp((-dttimes[spikes] / self.D))
        self.r_old[spikes] = 1 - (1 - self.r_old[spikes] * (1 - self.u_old[spikes])) * exp_D
        exp_F = np.exp((-dttimes[spikes] / self.F))
        self.u_old[spikes] = self.u0 + (
                self.u_old[spikes] + self.f * (1 - self.u_old[spikes]) - self.u0) * exp_F

        tmp_diff = np.abs(self.r_old - self.r_tmp)

        self.filter_spk[:] = 0
        self.filter_spk[spikes & (tmp_diff >= self.filterThr)] = 1

        self.r_tmp = self.r_old
        self.spikePrevMnt = spikeCurMnt
        del dttimes, exp_D, exp_F, tmp_diff

    def local_connect(self, spikes):
        inputSpk = np.reshape(spikes, (1, 1, self.self.spike_h, self.self.spike_w)).float()
        # tmp_fired = spikes != 0
        self.detectVoltage[spikes == False] -= 1
        tmpRes = signal.convolve2d(inputSpk, self.lifConv_weight)
        tmpRes = np.squeeze(tmpRes)
        self.detectVoltage += tmpRes.data
        self.detectVoltage[self.detectVoltage < self.voltageMin] = self.voltageMin

        self.lif_spk[:] = 0
        self.lif_spk[self.detectVoltage >= self.lifThr] = 1
        # self.detectVoltage[(self.detectVoltage < self.lifThr) & (self.detectVoltage > 0)] = 0
        self.detectVoltage[self.detectVoltage >= self.lifThr] *= 0.8

        del inputSpk, tmpRes
