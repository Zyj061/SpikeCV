import torch
import torch.nn as nn

from . import basicblocks as B
from . import common as common


class NoiseEstimator(nn.Module):
    def __init__(self, n=39, feats=64, nb=2):
        super(NoiseEstimator, self).__init__()
        self.pool = nn.AvgPool2d((2, 2), stride=(2, 2))
        self.encoder_r = SpikeEncoder(n, feats)
        self.encoder_g1 = SpikeEncoder(n, feats)
        self.encoder_g2 = SpikeEncoder(n, feats)
        self.encoder_b = SpikeEncoder(n, feats)
        self.fusion = B.sequential(
            B.upsample_convtranspose(feats*4, feats, mode='2L'), 
            *[B.conv(feats, feats, mode='CL') for _ in range(nb)],
            B.conv(feats, 1, mode='CL', bias=True)
        )

    def forward(self, spk, mask):
        r = self.encoder_r(self.pool(spk * mask[:,0:1,:,:]) * 4)  # n, t, h, w -> n, feats, h, w
        g1 = self.encoder_g1(self.pool(spk * mask[:,1:2,:,:]) * 4)
        g2 = self.encoder_g2(self.pool(spk * mask[:,2:3,:,:]) * 4)
        b = self.encoder_b(self.pool(spk * mask[:,3:4,:,:]) * 4)
        rgb_feats = torch.cat((torch.cat((torch.cat((r, g1) ,axis=1), g2) ,axis=1), b) ,axis=1)
        noise_map = self.fusion(rgb_feats)
        return noise_map


class SpikeEncoder(nn.Module):
    def __init__(self, in_channels=8, feats=64, nb=4):
        super(SpikeEncoder, self).__init__()
        self.head = B.sequential(B.conv(in_channels, feats, mode='CL'))
        self.body = B.sequential(*[
            common.ResBlock(
                feats, 3, act_type='leakyrelu', bias=True, res_scale=1
            ) for _ in range(nb)])
 
    def forward(self, x):
        output = self.head(x)
        output = self.body(output) 
        return output
