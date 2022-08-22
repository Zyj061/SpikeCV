# -*- coding: utf-8 -*- 
# @Time : 2022/7/21
# @Author : Rui Zhao
# @File : layers.py

import torch.nn as nn
import torch.nn.functional as F
import torch


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dilation=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=((kernel_size-1)*dilation)//2, bias=True, dilation=dilation),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=((kernel_size-1)*dilation)//2, bias=True, dilation=dilation),
            nn.LeakyReLU(0.1, inplace=True)
        )


def conv_s(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dilation=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=((kernel_size-1)*dilation)//2, bias=True, dilation=dilation),
            nn.BatchNorm2d(out_planes)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=((kernel_size-1)*dilation)//2, bias=True, dilation=dilation),
        )


def predict_flow(batchNorm, in_planes):
    if batchNorm:
        return nn.Sequential(
                nn.BatchNorm2d(32),
                nn.Conv2d(in_planes,2,kernel_size=1,stride=1,padding=0,bias=True),
            )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, 2, kernel_size=1, stride=1, padding=0, bias=True),
        )


def deconv(batchNorm, in_planes, out_planes):
    if batchNorm:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(in_planes),
            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True))

def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]

