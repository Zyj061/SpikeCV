import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

from . import basicblocks as B


class Representation(nn.Module):
    def __init__(self, wsize, blocks, n_feats=[8, 64, 64], nb=2):
        super(Representation, self).__init__()
        self.encoder_r = MutiScale3DEncoder(1, n_feats[0])
        self.encoder_g = MutiScale3DEncoder(1, n_feats[0]*2)
        self.encoder_b = MutiScale3DEncoder(1, n_feats[0])
        self.aligner = MultiColorAlignmentBlock(wsize, blocks)
        self.decoder_r = B.sequential(
            B.conv(n_feats[1]*blocks, n_feats[2], mode='CL', bias=True),
            *[B.conv(n_feats[2], n_feats[2], mode='CL') for _ in range(nb)],
            B.conv(n_feats[2], 1, mode='CL', bias=True))
        self.decoder_g = B.sequential(
            B.conv(n_feats[1]*blocks, n_feats[2], mode='CL', bias=True),
            *[B.conv(n_feats[2], n_feats[2], mode='CL') for _ in range(nb)],
            B.conv(n_feats[2], 1, mode='CL', bias=True))
        self.decoder_b = B.sequential(
            B.conv(n_feats[1]*blocks, n_feats[2], mode='CL', bias=True),
            *[B.conv(n_feats[2], n_feats[2], mode='CL') for _ in range(nb)],
            B.conv(n_feats[2], 1, mode='CL', bias=True))

    def forward(self, spk, mask):
        x_r = spk * mask[:,0:1,:,:] # n, t, h, w
        x_g = spk * (mask[:,1:2,:,:] + mask[:,2:3,:,:])
        x_b = spk * mask[:,3:4,:,:]
        x_r_feats = self.encoder_r(x_r.unsqueeze(1)) # n, 1, t, h, w
        x_g_feats = self.encoder_g(x_g.unsqueeze(1))
        x_b_feats = self.encoder_b(x_b.unsqueeze(1))
        x_r_feats, x_g_feats, x_b_feats = self.aligner(
            x_r_feats.squeeze(1), x_g_feats.squeeze(1), x_b_feats.squeeze(1), mask) # n, n_feats[1], h, w
        r = self.decoder_r(x_r_feats) # n, 1, h, w
        g = self.decoder_g(x_g_feats)
        b = self.decoder_b(x_b_feats)
        repre = torch.cat((torch.cat((r, g), axis=1), b), axis=1)
        return repre


class MutiScale3DEncoder(nn.Module):
    def __init__(self, in_channels=1, feats=16):
        super(MutiScale3DEncoder, self).__init__()
        self.conv_1 = nn.Conv3d(in_channels, feats, (3, 3, 3), padding=(1, 1, 1))
        self.conv_2 = nn.Conv3d(in_channels, feats, (5, 3, 3), padding=(2, 1, 1))
        self.conv_3 = nn.Conv3d(in_channels, feats, (7, 3, 3), padding=(3, 1, 1))
        self.ca = CAB3d(in_channels=feats*3)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.tail = B.sequential(
            nn.Conv3d(feats*3, feats, (3, 3, 3), padding=(1, 1, 1)),
            nn.Conv3d(feats, 1, (3, 3, 3), padding=(1, 1, 1))
        )
        
    def forward(self, spk):
        """
        spk: (N, C, D, H, W)
        """
        output_1 = self.lrelu(self.conv_1(spk))
        output_2 = self.lrelu(self.conv_2(spk))
        output_3 = self.lrelu(self.conv_3(spk))
        output = torch.cat((torch.cat((output_1, output_2), 1), output_3), 1)
        output = self.tail(output * self.ca(output))
        return output


class MultiColorAlignmentBlock(nn.Module): # short cut?
    def __init__(self, wsize=15, blocks=6, n_feats=[32, 64], nb=2, groups=4):
        super(MultiColorAlignmentBlock, self).__init__()
        self.wsize = wsize
        self.blocks = blocks
        self.down_r = B.sequential(
             B.conv(wsize, n_feats[0], mode='CL', bias=True),
             *[B.conv(n_feats[0], n_feats[0], mode='CL', bias=True) for _ in range(nb)],
             B.conv(n_feats[0], n_feats[1], mode='CL', bias=True))
        self.down_g = B.sequential(
             B.conv(wsize, n_feats[0], mode='CL', bias=True),
             *[B.conv(n_feats[0], n_feats[0], mode='CL', bias=True) for _ in range(nb)],
             B.conv(n_feats[0], n_feats[1], mode='CL', bias=True))
        self.down_b = B.sequential(
             B.conv(wsize, n_feats[0], mode='CL', bias=True),
             *[B.conv(n_feats[0], n_feats[0], mode='CL', bias=True) for _ in range(nb)],
             B.conv(n_feats[0], n_feats[1], mode='CL', bias=True))
        self.offset_r = nn.Conv2d(n_feats[1] * 2 +  1, 18 * groups, 3, 1, 1, bias=True)
        self.dcn_r = DeformConv2d(n_feats[1], n_feats[1], 3, 1, 1, 1, groups)
        self.offset_g = nn.Conv2d(n_feats[1] * 2 +  1, 18 * groups, 3, 1, 1, bias=True)
        self.dcn_g = DeformConv2d(n_feats[1], n_feats[1], 3, 1, 1, 1, groups)
        self.offset_b = nn.Conv2d(n_feats[1] * 2 +  1, 18 * groups, 3, 1, 1, bias=True)
        self.dcn_b = DeformConv2d(n_feats[1], n_feats[1], 3, 1, 1, 1, groups)
        self.offset_fusion = B.conv(18 * groups * 3, 18 * groups, mode='CL', bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, r, g, b, mask): # offset fuse module
        r_blocks = self._divide_features(r)
        g_blocks = self._divide_features(g)
        b_blocks = self._divide_features(b)
        n = len(r_blocks)
        mid_index = n // 2
        r_mid = self.down_r(r_blocks[mid_index])
        g_mid = self.down_g(g_blocks[mid_index])
        b_mid = self.down_b(b_blocks[mid_index])
        for i in range(n):
            if i != mid_index:
                r_block = self.down_r(r_blocks[i])
                g_block = self.down_g(g_blocks[i])
                b_block = self.down_b(b_blocks[i])
                offset_r = torch.cat([mask[:,0:1,:,:], r_block, r_mid], dim=1) 
                offset_r = self.lrelu(self.offset_r(offset_r))
                offset_g = torch.cat([mask[:,1:2,:,:] +  mask[:,2:3,:,:], g_block, g_mid], dim=1) 
                offset_g = self.lrelu(self.offset_g(offset_g))
                offset_b = torch.cat([mask[:,3:4,:,:], b_block, b_mid], dim=1) 
                offset_b = self.lrelu(self.offset_b(offset_b))
                offset = self.offset_fusion(torch.cat((torch.cat((offset_r , offset_g), axis=1), offset_b), axis=1))
                r_aligned = self.dcn_r(r_block, offset)
                g_aligned = self.dcn_g(g_block, offset)
                b_aligned = self.dcn_b(b_block, offset)
            else:
                r_aligned = r_mid
                g_aligned = g_mid
                b_aligned = b_mid
            r_feats = r_aligned if i == 0 else torch.cat((r_feats, r_aligned), axis=1)
            g_feats = g_aligned if i == 0 else torch.cat((g_feats, g_aligned), axis=1)
            b_feats = b_aligned if i == 0 else torch.cat((b_feats, b_aligned), axis=1)
        return r_feats, g_feats, b_feats
    
    def _divide_features(self, x):
        n = x.shape[1]
        feats = []
        stride = (n - self.wsize) // (self.blocks - 1)
        for i in range(self.blocks):
            feats.append(x[:,i*stride:i*stride+self.wsize,:,:])
        return feats


class CAB3d(nn.Module):
    def __init__(self, in_channels, reduction=2):
        super(CAB3d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.ca_block = nn.Sequential(
                nn.Conv3d(in_channels, in_channels // reduction, 1, padding=0, bias=True),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv3d(in_channels // reduction, in_channels, 1, padding=0, bias=True),
                nn.Softmax(dim=1))

    def forward(self, x):
        x = self.avg_pool(x)
        weight = self.ca_block(x)
        return weight
