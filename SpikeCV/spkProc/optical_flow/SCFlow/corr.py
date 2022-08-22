# -*- coding: utf-8 -*- 
# @Time : 2022/7/21
# @Author : Rui Zhao
# @File : corr.py

import torch.nn as nn
import torch.nn.functional as F

try:
    from spatial_correlation_sampler import spatial_correlation_sample
except ImportError as e:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=ImportWarning)
        warnings.warn("failed to load custom correlation module"
                      "which is needed for FlowNetC", ImportWarning)


def corr(input1, input2):
    out_corr = spatial_correlation_sample(input1,
                                          input2,
                                          kernel_size=1,
                                          patch_size=9,
                                          stride=1,
                                          padding=0,
                                          dilation_patch=1)
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
    return F.leaky_relu_(out_corr, 0.1)


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]