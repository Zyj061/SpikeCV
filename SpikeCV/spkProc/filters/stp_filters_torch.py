# -*- coding: utf-8 -*- 
# @Time : 2021/11/19 16:25 
# @Author : Yajing Zheng
# @File : stp_filters_torch.py
import torch


class STPFilter:

    def __init__(self, spike_h, spike_w, device, **STPargs):
        self.spike_h = spike_h
        self.spike_w = spike_w
        self.device = device

        # specify stp parameters
        if STPargs.get('u0', None) is None:
            self.u0 = 0.1
            self.D = 0.02
            self.F = 1.7
            self.f = 0.11
            self.time_unit = 2000

        else:
            self.u0 = STPargs.get('u0')
            self.D = STPargs.get('D')
            self.F = STPargs.get('F')
            self.f = STPargs.get('f')
            self.time_unit = STPargs.get('time_unit')

        self.r0 = 1

        self.R = torch.ones(self.spike_h, self.spike_w) * self.r0
        self.u = torch.ones(self.spike_h, self.spike_w) * self.u0
        self.r_old = torch.ones(self.spike_h, self.spike_w) * self.r0

        self.R = self.R.to(self.device)
        self.u = self.u.to(self.device)
        self.r_old = self.r_old.to(self.device)

        # LIF detect layer parameters
        self.detectVoltage = torch.zeros(self.spike_h, self.spike_w).to(self.device)
        if STPargs.get('lifSize', None) is None:
            lifSize = 3
        else:
            lifSize = STPargs.get('lifSize')

        self.lifConv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(lifSize, lifSize), padding=(1, 1),
                                       bias=False)
        self.lifConv.weight.data = torch.ones(1, 1, lifSize, lifSize) * 3.0

        self.lifConv = self.lifConv.to(self.device)
        if STPargs.get('filterThr', None) is None:
            self.filterThr = 0.1  # filter threshold
            self.voltageMin = -8
            self.lifThr = 2
        else:
            self.filterThr = STPargs.get('filterThr')
            self.voltageMin = STPargs.get('voltageMin')
            self.lifThr = STPargs.get('lifThr')

        self.filter_spk = torch.zeros(self.spike_h, self.spike_w).to(self.device)
        self.lif_spk = torch.zeros(self.spike_h, self.spike_w).to(self.device)
        self.spikePrevMnt = torch.zeros([self.spike_h, self.spike_w], device=self.device)

    def update_dynamics(self, curT, spikes):

        spikeCurMnt = self.spikePrevMnt.detach().clone()
        spike_bool = spikes.bool()
        spikeCurMnt[spike_bool] = curT + 1
        dttimes = spikeCurMnt - self.spikePrevMnt
        dttimes = dttimes / self.time_unit
        exp_D = torch.exp((-dttimes[spike_bool] / self.D))
        self.R[spike_bool] = 1 - (1 - self.R[spike_bool] * (1 - self.u[spike_bool])) * exp_D
        exp_F = torch.exp((-dttimes[spike_bool] / self.F))
        self.u[spike_bool] = self.u0 + (
                self.u[spike_bool] + self.f * (1 - self.u[spike_bool]) - self.u0) * exp_F

        tmp_diff = torch.abs(self.R - self.r_old)

        self.filter_spk[:] = 0
        self.filter_spk[spike_bool & (tmp_diff >= self.filterThr)] = 1

        self.r_old = self.R.detach().clone()
        self.spikePrevMnt = spikeCurMnt.detach().clone()
        del spikeCurMnt, dttimes, exp_D, exp_F, tmp_diff

    def update_dynamic_offline(self, spikes, intervals):

        isi_num = intervals.shape[0]
        R = torch.ones(isi_num, self.spike_h, self.spike_w) * self.r0
        u = torch.ones(isi_num, self.spike_h, self.spike_w) * self.u0
        prev_isi = intervals[0, :, :]

        for t in range(1, isi_num):
            tmp_isi = intervals[t, :, :]
            update_idx = (tmp_isi != prev_isi) & (spikes[t, :, :] == 0) | (tmp_isi == 1)
            tmp_isi = torch.from_numpy(tmp_isi).to(self.device).float()

            exp_D = torch.exp((-tmp_isi[update_idx] / self.D))
            self.R[update_idx] = 1 - (1 - self.R[update_idx] * (1 - self.u[update_idx])) * exp_D
            exp_F = torch.exp((-tmp_isi[update_idx] / self.F))
            self.u[update_idx] = self.u0 + (
                    self.u[update_idx] + self.f * (1 - self.u[update_idx]) - self.u0) * exp_F

            R[t, :, :] = self.R.detach().clone()
            u[t, :, :] = self.u.detach().clone()

        return R, u

    def local_connect(self, spikes):
        inputSpk = torch.reshape(spikes, (1, 1, self.spike_h, self.spike_w)).float()
        # tmp_fired = spikes != 0
        self.detectVoltage[spikes == False] -= 1
        tmpRes = self.lifConv(inputSpk)
        tmpRes = torch.squeeze(tmpRes).to(self.device)
        self.detectVoltage += tmpRes.data
        self.detectVoltage[self.detectVoltage < self.voltageMin] = self.voltageMin

        self.lif_spk[:] = 0
        self.lif_spk[self.detectVoltage >= self.lifThr] = 1
        # self.detectVoltage[(self.detectVoltage < self.lifThr) & (self.detectVoltage > 0)] = 0
        self.detectVoltage[self.detectVoltage >= self.lifThr] *= 0.8

        del inputSpk, tmpRes

    def local_connect_offline(self, spikes):
        timestamps = spikes.shape[0]
        tmp_voltage = []
        lif_spk = []

        for iSpk in range(timestamps):
            tmp_spikes = spikes[iSpk]
            tmp_spk = torch.from_numpy(spikes[iSpk]).to(self.device)
            inputSpk = torch.reshape(tmp_spk, (1, 1, self.spike_h, self.spike_w)).float()
            # tmp_fired = spikes != 0
            self.detectVoltage[tmp_spikes == 0] -= 1
            tmpRes = self.lifConv(inputSpk)
            tmpRes = torch.squeeze(tmpRes).to(self.device)
            self.detectVoltage += tmpRes.data
            self.detectVoltage[self.detectVoltage < self.voltageMin] = self.voltageMin

            self.lif_spk[:] = 0
            self.lif_spk[self.detectVoltage >= self.lifThr] = 1
            # self.detectVoltage[(self.detectVoltage < self.lifThr) & (self.detectVoltage > 0)] = 0
            self.detectVoltage[self.detectVoltage >= self.lifThr] *= 0.8
            tmp_voltage.append(self.detectVoltage.cpu().detach().numpy())
            lif_spk.append(self.lif_spk.cpu().detach().numpy())

        del inputSpk, tmpRes
        return tmp_voltage, lif_spk
