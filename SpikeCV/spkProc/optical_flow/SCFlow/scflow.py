# -*- coding: utf-8 -*- 
# @Time : 2022/7/21
# @Author : Rui Zhao
# @File : scflow.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.init import kaiming_normal_, constant_
from spkProc.optical_flow.SCFlow.layers import predict_flow, conv_s, conv, deconv
from spkProc.optical_flow.SCFlow.utils import flow_warp
from spkProc.optical_flow.SCFlow.corr import corr
__all__ = ['scflow']


class SpikeRepresentation(nn.Module):
    def __init__(self, data_length=25, batchNorm=False):
        super(SpikeRepresentation, self).__init__()
        self.batchNorm = batchNorm

        # conv for spike tensor
        self.conv_s1 = conv(self.batchNorm,  25, 32, kernel_size=3, stride=1)
        self.conv_s2 = conv(self.batchNorm,  32, 32, kernel_size=3, stride=1)

    def warp_slices(self, seq, flow, dt=10):
        b, c, h, w = seq.shape
        seq = seq.reshape([b*c, 1, h, w])
        flow_factor = (torch.linspace(-12, 12, steps=25) / dt).cuda()
        flow_factor = flow_factor.reshape([1, c, 1, 1, 1])
        flow = flow.unsqueeze(dim=1)
        factored_flow = flow * flow_factor
        factored_flow = factored_flow.reshape([b*c, 2, h, w])
        seq = flow_warp(seq, factored_flow)
        seq = seq.reshape([b, c, h, w])
        return seq

    def forward(self, seq, flow_input, dt=10, warp=False):
        # mask generation
        if warp:
            flow_input = flow_warp(flow_input, -1*flow_input)
        
        flow = flow_input.clone().detach()

        # warp every slice in seq
        seq = self.warp_slices(seq, flow, dt=dt)

        # conv for spike tensor
        rep = self.conv_s2(self.conv_s1(seq))

        return rep


class FeatureEncoder(nn.Module):
    def __init__(self, num_chs, batchNorm=False):
        super(FeatureEncoder, self).__init__()
        self.batchNorm = batchNorm
        self.num_chs = num_chs
        self.conv_list = nn.ModuleList()

        for l, (ch_in, ch_out) in enumerate(zip(num_chs[:-1], num_chs[1:])):
            if l == 0:
                layer = nn.Sequential(
                    conv(self.batchNorm, ch_in, ch_out, kernel_size=3, stride=1),
                    conv(self.batchNorm, ch_out, ch_out, kernel_size=3, stride=1)
                )
            else:    
                layer = nn.Sequential(
                    conv(self.batchNorm, ch_in, ch_out, kernel_size=3, stride=2),
                    conv(self.batchNorm, ch_out, ch_out, kernel_size=3, stride=1)
                )
            self.conv_list.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv_module in self.conv_list:
            x = conv_module(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


class FlowEstimator(nn.Module):
    def __init__(self, ch_in, batchNorm=False):
        super(FlowEstimator, self).__init__()
        self.batchNorm = batchNorm
        self.conv1 = conv(self.batchNorm, ch_in, 96, kernel_size=3, stride=1)
        self.conv2 = conv(self.batchNorm, ch_in + 96, 64, kernel_size=3, stride=1)
        self.conv3 = conv(self.batchNorm, ch_in + 96 + 64, 32, kernel_size=3, stride=1)
        self.conv4 = conv_s(self.batchNorm, ch_in + 96 + 64 +32, 2, kernel_size=3, stride=1)

    def forward(self, x):
        x1 = torch.cat([self.conv1(x), x], dim=1)
        x2 = torch.cat([self.conv2(x1), x1], dim=1)
        x3 = torch.cat([self.conv3(x2), x2], dim=1)
        x_out = self.conv4(x3)
        return x_out


class scflow(nn.Module):
    def __init__(self, batchNorm):
        super(scflow, self).__init__()
        self.batchNorm = batchNorm
        self.search_range = 4
        self.num_chs = [32, 32, 64, 96, 128]
        self.output_level = 4
        self.leakyReLU = nn.LeakyReLU(0.1, inplace=True)
        
        self.spike_representation = SpikeRepresentation( batchNorm=self.batchNorm)
        self.feature_encoder = FeatureEncoder(num_chs=self.num_chs, batchNorm=self.batchNorm)
        self.dim_corr = (self.search_range * 2 + 1) ** 2
        self.num_ch_in = 32 + self.dim_corr + 2
        self.flow_estimators = FlowEstimator(self.num_ch_in)
        
        self.conv_1x1 = nn.ModuleList([conv_s(False, 128, 32, kernel_size=1, stride=1),
                                        conv_s(False, 96, 32, kernel_size=1, stride=1),
                                        conv_s(False, 64, 32, kernel_size=1, stride=1),
                                        conv_s(False, 32, 32, kernel_size=1, stride=1)])

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]
    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]
    def num_parameters(self):
        return sum([p.data.nelement() if p.requires_grad else 0 for p in self.parameters()])

    def init_weights(self):
        for layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, seq1, seq2, flow, dt=10):
        res_dict = {}

        flows = []

        x1_repre = self.spike_representation(seq1, flow, warp=False, dt=dt)
        x2_repre = self.spike_representation(seq2, flow, warp=True, dt=dt)

        res_dict['x1_repre'] = x1_repre.abs().mean(dim=1, keepdim=True)
        res_dict['x2_repre'] = x2_repre.abs().mean(dim=1, keepdim=True)

        x1_pym = self.feature_encoder(x1_repre)
        x2_pym = self.feature_encoder(x2_repre)

        b, c, h, w = x1_pym[0].shape
        init_dtype = x1_pym[0].dtype
        init_device = x1_pym[0].device
        flow = torch.zeros(b, 2, h, w, dtype=init_dtype, device=init_device).float()

        for l, (x1, x2) in enumerate(zip(x1_pym, x2_pym)):

            # warping
            if l == 0:
                x2_warp = x2
            else:
                flow = F.interpolate(flow * 2, scale_factor=2, mode='bilinear', align_corners=True)
                x2_warp = flow_warp(x2, flow)
            
            # correlation
            out_corr = corr(x1, x2_warp)

            # flow estimating
            x1_1x1 = self.conv_1x1[l](x1)
            flow_res = self.flow_estimators(torch.cat([out_corr, x1_1x1, flow], dim=1))
            flow = flow + flow_res


            flows.append(flow)
        
        return flows[::-1], res_dict


def get_scflow(data=None, batchNorm=False):
    model = scflow(batchNorm=batchNorm)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    else:
        model.init_weights()
    return model

