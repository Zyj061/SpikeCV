# -*- coding: utf-8 -*-
# @Time : 2026/07/06
# @File : bsf_model.py

import torch
import torch.nn as nn

from spikecv.spkProc.reconstruction.BSF_Recon.rep import MODF
from spikecv.spkProc.reconstruction.BSF_Recon.align import Multi_Granularity_Align


class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()

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


def split_and_b_cat(x):
    x0 = x[:, 10 - 10:10 + 10 + 1].clone()
    x1 = x[:, 20 - 10:20 + 10 + 1].clone()
    x2 = x[:, 30 - 10:30 + 10 + 1].clone()
    x3 = x[:, 40 - 10:40 + 10 + 1].clone()
    x4 = x[:, 50 - 10:50 + 10 + 1].clone()
    return torch.cat([x0, x1, x2, x3, x4], dim=0)


class Encoder(nn.Module):
    def __init__(self, base_dim=64, layers=4, act=nn.ReLU()):
        super().__init__()
        self.conv_list = nn.ModuleList()
        for _ in range(layers):
            self.conv_list.append(
                nn.Sequential(
                    nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
                    act,
                    nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
                )
            )
        self.act = act

    def forward(self, x):
        for conv in self.conv_list:
            x = self.act(conv(x) + x)
        return x


class BSF(BasicModel):
    """BSF network for spike camera image reconstruction (CVPR 2024)."""

    def __init__(self, act=nn.ReLU()):
        super().__init__()
        self.offset_groups = 4
        self.corr_max_disp = 3

        self.rep = MODF(base_dim=64, act=act)
        self.encoder = Encoder(base_dim=64, layers=4, act=act)
        self.align = Multi_Granularity_Align(base_dim=64, groups=self.offset_groups, act=act, sc=3)
        self.recons = nn.Sequential(
            nn.Conv2d(64 * 5, 64 * 3, kernel_size=3, padding=1),
            act,
            nn.Conv2d(64 * 3, 64, kernel_size=3, padding=1),
            act,
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, input_dict):
        dsft_dict = input_dict['dsft_dict']
        dsft_b_cat = {
            'dsft11': split_and_b_cat(dsft_dict['dsft11']),
            'dsft12': split_and_b_cat(dsft_dict['dsft12']),
            'dsft21': split_and_b_cat(dsft_dict['dsft21']),
            'dsft22': split_and_b_cat(dsft_dict['dsft22']),
        }

        feat_b_cat = self.rep(dsft_b_cat)
        feat_b_cat = self.encoder(feat_b_cat)
        feat_list = feat_b_cat.chunk(5, dim=0)
        feat_list_align = self.align(feat_list=feat_list)
        out = self.recons(torch.cat(feat_list_align, dim=1))
        return out
