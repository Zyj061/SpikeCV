import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class crop(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N, C, H, W = x.shape
        x = x[0:N, 0:C, 0:H-1, 0:W]
        return x

class shift(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift_down = nn.ZeroPad2d((0,0,1,0))
        self.crop = crop()

    def forward(self, x):
        x = self.shift_down(x)
        x = self.crop(x)
        return x

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, blind=True,stride=1,padding=0,kernel_size=3):
        super().__init__()
        self.blind = blind
        if blind:
            self.shift_down = nn.ZeroPad2d((0,0,1,0))
            self.crop = crop()
        self.replicate = nn.ReplicationPad2d(1)
#         self.conv = nn.Conv2d(in_channels, out_channels, 3, bias=bias)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding,bias=bias) 
        self.relu = nn.LeakyReLU(0.1, inplace=True)
#         self.ln = nn.GroupNorm(1,out_channels)

    def forward(self, x):
        if self.blind:
            x = self.shift_down(x)
        x = self.replicate(x)
        x = self.conv(x)
        x = self.relu(x)
        
        if self.blind:
            x = self.crop(x)
        return x

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=False, bn=False, bias=True,blind=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self,bias=False,blind=False):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False,bias=bias,blind=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

def weights_init_rcan(m):
    """
    custom weights initialization called on netG and netD
    https://github.com/pytorch/examples/blob/master/dcgan/main.py
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if classname.find('BasicConv') != -1:
            m.conv.weight.data.normal_(0.0, 0.02)
            if m.bn != None:
                m.bn.bias.data.fill_(0)
        else:
            m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
class Temporal_Fusion(nn.Module):

    def __init__(self, nf=64, nframes=3, center=1,bias=False):
        super(Temporal_Fusion, self).__init__()
        self.center = center

        self.tAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=bias)
        self.tAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=bias)

        self.fea_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=bias)

        self.sAtt_1 = nn.Conv2d(nframes * nf, nf, 1, 1, bias=bias)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(nf * 2, nf, 1, 1, bias=bias)
        self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=bias)
        self.sAtt_4 = nn.Conv2d(nf, nf, 1, 1, bias=bias)
        self.sAtt_5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=bias)
        self.sAtt_L1 = nn.Conv2d(nf, nf, 1, 1, bias=bias)
        self.sAtt_L2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=bias)
        self.sAtt_L3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=bias)
        self.sAtt_add_1 = nn.Conv2d(nf, nf, 1, 1, bias=bias)
        self.sAtt_add_2 = nn.Conv2d(nf, nf, 1, 1, bias=bias)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nonlocal_fea):
        B, N, C, H, W = nonlocal_fea.size()  

        emb_ref = self.tAtt_2(nonlocal_fea[:, self.center, :, :, :].clone())
        emb = self.tAtt_1(nonlocal_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1) 
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1)
        cor_prob = cor_prob.view(B, -1, H, W)
        nonlocal_fea = nonlocal_fea.view(B, -1, H, W) * cor_prob

        fea = self.lrelu(self.fea_fusion(nonlocal_fea))

        att = self.lrelu(self.sAtt_1(nonlocal_fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))

        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        fea = fea * att * 2 + att_add

        return fea