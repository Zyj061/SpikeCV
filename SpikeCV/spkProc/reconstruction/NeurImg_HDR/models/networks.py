import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02, gpu_ids=''):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s, gpu_id is %s.' % (init_type, gpu_ids))
    net.apply(init_func)  


def load_pretrained_vgg16(net, gpu_ids):
    net_module = net.module
    net_module.features = torch.nn.DataParallel(net_module.features)
    new_model_dict = net_module.state_dict()
    ckpt = torch.load('models/vgg16_best.pth', map_location=lambda storage, loc: storage.cuda(gpu_ids[0]))
    vgg16_dict = ckpt['state_dict']
    pretrained_dict = {k: v for k, v in vgg16_dict.items() if k in new_model_dict}
    print(len(pretrained_dict.keys()))
    new_model_dict.update(pretrained_dict)
    net_module.load_state_dict(new_model_dict)
    

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs

    init_weights(net, init_type, init_gain=init_gain, gpu_ids=gpu_ids)
    return net


def define_ColorNet(netColor='image', norm='instance', init_type='normal', init_gain=0.02, n_blocks=9, state_nc=32, gpu_ids=[]):
    net = None
    if netColor == 'image':
        net = ColorNet_Image(n_blocks=n_blocks, norm_type=norm)
    elif netColor == 'video':
        net = ColorNet_Video(n_blocks=n_blocks, norm_type=norm, state_nc=state_nc)

    init_net(net, init_type, init_gain, gpu_ids)
    return net

def define_UpsampleNet(gpu_ids=[], scale=2):
    net = None
    net = Upsampling_Net(scale)

    if len(gpu_ids) > 0:
        net.to(gpu_ids[0])    
        if scale == 2:
            net.load_state_dict(torch.load('../spkProc/reconstruction/NeurImg_HDR/checkpoints/upsample_net/upsample_net_2x.pth'))
        elif scale == 4:
            net.load_state_dict(torch.load('../spkProc/reconstruction/NeurImg_HDR/checkpoints/upsample_net/upsample_net_4x.pth'))
        elif scale == 8:
            net.load_state_dict(torch.load('../spkProc/reconstruction/NeurImg_HDR/checkpoints/upsample_net/upsample_net_8x.pth'))            
        
        # single GPU checkpoints loaded on multi GPUs model
        # kwargs={'map_location':lambda storage, loc: storage.cuda(gpu_ids)}
        # state_dict = torch.load('models/UpsampleNet.pth')
        # # create new OrderedDict that does not contain `module.`
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = 'module.' + k # remove `module.`
        #     new_state_dict[name] = v
        # # load params
        # net.load_state_dict(new_state_dict)
        
        net.eval()
    return net

        
def define_G(norm='instance', init_type='normal',init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = LuminanceFusion_Net(norm_layer=norm_layer)

    init_net(net, init_type, init_gain, gpu_ids)
    return net


class CN_Conv2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size = 4, stride = 2, padding = 1):
        super(CN_Conv2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,kernel_size = kernel_size, stride = stride, padding = padding),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            nn.InstanceNorm2d(out_ch)
        )

    def forward(self, input):
        return self.conv(input)


class CN_UpConv2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CN_UpConv2D, self).__init__()
        self.deconv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1,inplace=True),
            nn.InstanceNorm2d(out_ch) 
        )

    def forward(self, input):
        return self.deconv(input)


class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, norm=None, bias=True, act=nn.LeakyReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(1):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, stride=1, padding=1, bias=bias))
            if norm == 'instance':
                m.append(nn.InstanceNorm2d(n_feats))
            elif norm == 'batch':
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        
        return res


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y.expand_as(x)


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x * psi, psi
    
    
class spikes_conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=4, stride=2, padding=1, norm_layer=nn.InstanceNorm2d):
        super(spikes_conv2d, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            norm_layer(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x
    
    
class ColorNet_Image(nn.Module):
    def __init__(self, n_blocks=5, norm_type='instance'):
        super(ColorNet_Image, self).__init__()
        u_sequence = []
        u_sequence += [
            CN_Conv2D(2, 32, 3, 2, 1),
            CN_Conv2D(32, 64, 3, 2, 1),
            CN_UpConv2D(64, 32),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 1, 3, 1, 1)
        ]
        self.u_dequan = nn.Sequential(*u_sequence)
        
        v_sequence = []
        v_sequence += [
            CN_Conv2D(2, 32, 3, 2, 1),
            CN_Conv2D(32, 64, 3, 2, 1),
            CN_UpConv2D(64, 32),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 1, 3, 1, 1)
        ]
        self.v_dequan = nn.Sequential(*v_sequence)
        
        self.y_conv = CN_Conv2D(1, 64, 3, 1, 1)
        self.u_conv = CN_Conv2D(1, 64, 3, 1, 1)
        self.v_conv = CN_Conv2D(1, 64, 3, 1, 1)
        
        self.r_fuse = CN_Conv2D(128, 64, 1, 1, 0)
        self.r_conv = CN_Conv2D(64, 32, 3, 1, 1)
        self.g_fuse = CN_Conv2D(192, 64, 1, 1, 0)
        self.g_conv = CN_Conv2D(64, 32, 3, 1, 1)
        self.b_fuse = CN_Conv2D(128, 64, 1, 1, 0)
        self.b_conv = CN_Conv2D(64, 32, 3, 1, 1)
        
        recon_sequence = []
        
        n_downsampling = 2
        ngf = 32*3
        for i in range(n_downsampling):  # downsample the feature map
            mult = 2 ** i
            recon_sequence += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.LeakyReLU(0.1, True)
            ]

        for i in range(n_blocks):  # ResNet
            recon_sequence += [
                ResBlock(ngf * 2 ** n_downsampling, 3, norm=norm_type)
            ]

        for i in range(n_downsampling):  # upsample the feature map
            mult = 2 ** (n_downsampling - i)
            recon_sequence += [
                # nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1,  bias=True),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.LeakyReLU(0.1, True)
            ]
        
        recon_sequence += [nn.Conv2d(32*3, 3, 3, 1, 1)]
        self.reconstruct = nn.Sequential(*recon_sequence)
        
        # SE channel-wise attention
        self.SEAtt_r = SELayer(channel=64, reduction=32)
        self.SEAtt_g = SELayer(channel=64, reduction=32)
        self.SEAtt_b = SELayer(channel=64, reduction=32)
        
    def forward(self, y, u, v):
        u_f1 = self.u_dequan[0](torch.cat([u,y],1))
        u_f2 = self.u_dequan[1:-2](u_f1)
        u_res = self.u_dequan[-2:](torch.cat([u_f2, u_f1],1))
        v_f1 = self.v_dequan[0](torch.cat([v,y],1))
        v_f2 = self.v_dequan[1:-2](v_f1)
        v_res = self.v_dequan[-2:](torch.cat([v_f2, v_f1],1))
        u_float = u + u_res
        v_float = v + v_res
        y_feat = self.y_conv(y)
        u_feat = self.u_conv(u_float)
        v_feat = self.v_conv(v_float)
        r_c = self.r_conv(self.SEAtt_r(self.r_fuse(torch.cat([y_feat,v_feat],1))))
        g_c = self.g_conv(self.SEAtt_g(self.g_fuse(torch.cat([y_feat,u_feat,v_feat],1))))
        b_c = self.b_conv(self.SEAtt_b(self.b_fuse(torch.cat([y_feat,u_feat],1))))
        out = torch.sigmoid(self.reconstruct(torch.cat([r_c,g_c,b_c],1)))
        
        return out   
    
    
class ColorNet_Video(nn.Module):
    def __init__(self, n_blocks=5, state_nc=32, norm_type='instance'):
        super(ColorNet_Video, self).__init__()
        u_sequence = []
        u_sequence += [
            CN_Conv2D(2, 32, 3, 2, 1),
            CN_Conv2D(32, 64, 3, 2, 1),
            CN_UpConv2D(64, 32),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 1, 3, 1, 1)
        ]
        self.u_dequan = nn.Sequential(*u_sequence)
        self.state_nc = state_nc
        
        v_sequence = []
        v_sequence += [
            CN_Conv2D(2, 32, 3, 2, 1),
            CN_Conv2D(32, 64, 3, 2, 1),
            CN_UpConv2D(64, 32),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 1, 3, 1, 1)
        ]
        self.v_dequan = nn.Sequential(*v_sequence)
        
        self.y_conv = CN_Conv2D(1, 64, 3, 1, 1)
        self.u_conv = CN_Conv2D(1, 64, 3, 1, 1)
        self.v_conv = CN_Conv2D(1, 64, 3, 1, 1)
        
        self.r_fuse = CN_Conv2D(128, 64, 1, 1, 0)
        self.r_conv = CN_Conv2D(64, 32, 3, 1, 1)
        self.g_fuse = CN_Conv2D(192, 64, 1, 1, 0)
        self.g_conv = CN_Conv2D(64, 32, 3, 1, 1)
        self.b_fuse = CN_Conv2D(128, 64, 1, 1, 0)
        self.b_conv = CN_Conv2D(64, 32, 3, 1, 1)
        recon_sequence = []
        n_downsampling = 2
        
        # fuse rgb by 1x1 conv
        self.fuse_rgb = nn.Conv2d(32*3, 32, 1, 1)
        
        # add last state
        ngf = 32+self.state_nc
        for i in range(n_downsampling):  # downsample the feature map
            mult = 2 ** i
            recon_sequence += [
                nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(ngf*mult*2),
                nn.LeakyReLU(0.1, True)
            ]

        for i in range(n_blocks):  # ResNet
            recon_sequence += [
                ResBlock(ngf*2**n_downsampling, 3, norm=norm_type)
            ]
        
        for i in range(n_downsampling):  # upsample the feature map
            mult = 2 ** (n_downsampling - i)
            recon_sequence += [
                # nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(ngf*mult, int(ngf*mult/2), kernel_size=3, stride=1, padding=1,  bias=True),
                nn.InstanceNorm2d(int(ngf*mult/2)),
                nn.LeakyReLU(0.1, True)
            ]
        recon_sequence += [nn.Conv2d(ngf, self.state_nc, 3, 1, 1)]
        self.reconstruct = nn.Sequential(*recon_sequence)
        
        self.activate = nn.Conv2d(self.state_nc, 3, 3, 1, 1)
        
        # SE channel-wise attention
        self.SEAtt_r = SELayer(channel=64, reduction=32)
        self.SEAtt_g = SELayer(channel=64, reduction=32)
        self.SEAtt_b = SELayer(channel=64, reduction=32)
        
    def forward(self, y, u, v, last_state):
        batch_size = y.data.size()[0]
        spatial_size = y.data.size()[2:]
        # generate empty prev_state, if None is provided
        if last_state is None:
            state_size = [batch_size, self.state_nc] + list(spatial_size)
            # zero init
            last_state = torch.zeros(state_size).to(y.device)
        
        u_f1 = self.u_dequan[0](torch.cat([u,y],1))
        u_f2 = self.u_dequan[1:-2](u_f1)
        u_res = self.u_dequan[-2:](torch.cat([u_f2, u_f1],1))
        v_f1 = self.v_dequan[0](torch.cat([v,y],1))
        v_f2 = self.v_dequan[1:-2](v_f1)
        v_res = self.v_dequan[-2:](torch.cat([v_f2, v_f1],1))
        u_float = u + u_res
        v_float = v + v_res
        y_feat = self.y_conv(y)
        u_feat = self.u_conv(u_float)
        v_feat = self.v_conv(v_float)
        r_c = self.r_conv(self.SEAtt_r(self.r_fuse(torch.cat([y_feat,v_feat],1))))
        g_c = self.g_conv(self.SEAtt_g(self.g_fuse(torch.cat([y_feat,u_feat,v_feat],1))))
        b_c = self.b_conv(self.SEAtt_b(self.b_fuse(torch.cat([y_feat,u_feat],1))))
        fused_rgb = self.fuse_rgb(torch.cat([r_c,g_c,b_c], 1))

        # if last_state is None:
        #     # state_size = [batch_size, self.state_nc] + list(spatial_size)
        #     # duplicate the fused_rgb to last_state
        #     last_state = fused_rgb.detach()

        last_state = self.reconstruct(torch.cat([fused_rgb, last_state], 1))
        out = torch.sigmoid(self.activate(last_state))
        
        return out, last_state
 

class LuminanceFusion_Net(nn.Module):
    def __init__(self, norm_layer=nn.InstanceNorm2d):
        super(LuminanceFusion_Net, self).__init__()
        self.sequence = []
        self.sequence += [nn.Conv2d(1, 64, 3, padding=1)]
        self.sequence += [norm_layer(64)]
        self.sequence += [nn.ReLU(inplace=True)]
        self.sequence += [nn.Conv2d(64, 64, 3, padding=1)]
        self.sequence += [norm_layer(64)]
        self.sequence += [nn.ReLU(inplace=True)]
        self.sequence += [nn.MaxPool2d(2)] # 6
        self.sequence += [nn.Conv2d(65, 128, 3, padding=1)]
        self.sequence += [norm_layer(128)]
        self.sequence += [nn.ReLU(inplace=True)]
        self.sequence += [nn.Conv2d(128, 128, 3, padding=1)]
        self.sequence += [norm_layer(128)]
        self.sequence += [nn.ReLU(inplace=True)]
        self.sequence += [nn.MaxPool2d(2)] # 13
        self.sequence += [nn.Conv2d(160, 256, 3, padding=1)]
        self.sequence += [norm_layer(256)]
        self.sequence += [nn.ReLU(inplace=True)]
        self.sequence += [nn.Conv2d(256, 256, 3, padding=1)]
        self.sequence += [norm_layer(256)]
        self.sequence += [nn.ReLU(inplace=True)]
        self.sequence += [nn.Conv2d(256, 256, 3, padding=1)]
        self.sequence += [norm_layer(256)]
        self.sequence += [nn.ReLU(inplace=True)]
        self.sequence += [nn.MaxPool2d(2)] # 23
        self.sequence += [nn.Conv2d(320, 512, 3, padding=1)]
        self.sequence += [norm_layer(512)]
        self.sequence += [nn.ReLU(inplace=True)]
        self.sequence += [nn.Conv2d(512, 512, 3, padding=1)]
        self.sequence += [norm_layer(512)]
        self.sequence += [nn.ReLU(inplace=True)]
        self.sequence += [nn.Conv2d(512, 512, 3, padding=1)]
        self.sequence += [norm_layer(512)]
        self.sequence += [nn.ReLU(inplace=True)]
        self.sequence += [nn.MaxPool2d(2)] # 33
        self.sequence += [nn.Conv2d(640, 512, 3, padding=1)]
        self.sequence += [norm_layer(512)]
        self.sequence += [nn.ReLU(inplace=True)]
        self.sequence += [nn.Conv2d(512, 512, 3, padding=1)]
        self.sequence += [norm_layer(512)]
        self.sequence += [nn.ReLU(inplace=True)]
        self.sequence += [nn.Conv2d(512, 512, 3, padding=1)]
        self.sequence += [norm_layer(512)]
        self.sequence += [nn.ReLU(inplace=True)]
        self.en_pool = nn.MaxPool2d(2)
        self.features = nn.Sequential(*self.sequence)
        self.en_conv = nn.Conv2d(768, 1024, 3, padding=1)
        self.en_norm = norm_layer(1024)
        
        # Decoder
        self.deconv0 = CN_UpConv2D(1024, 512)
        self.fu_conv1 = nn.Conv2d(1280, 512, kernel_size=1, stride=1, padding=0)
        self.deconv1 = CN_UpConv2D(512, 512)
        self.fu_conv2 = nn.Conv2d(1152, 512, kernel_size=1, stride=1, padding=0)
        self.deconv2 = CN_UpConv2D(512, 256)
        self.fu_conv3 = nn.Conv2d(576, 256, kernel_size=1, stride=1, padding=0)
        self.deconv3 = CN_UpConv2D(256, 128)
        self.fu_conv4 = nn.Conv2d(288, 128, kernel_size=1, stride=1, padding=0)
        self.deconv4 = CN_UpConv2D(128, 64)
        self.fu_conv5 = nn.Conv2d(129, 64, 1)
        self.fu_conv6 = nn.Conv2d(64, 32, 1)
        self.de_norm5 = norm_layer(32)
        self.fu_conv7 = nn.Conv2d(96, 16, 1)
        self.final_conv1 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.final_conv2 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0, bias=True)
        
        # -------------------intensity map encoder----------------------------
        self.s_down1 = spikes_conv2d(1, 32, 4)
        self.s_down2 = spikes_conv2d(32, 64, 4)
        self.s_down3 = spikes_conv2d(64, 128, 4)
        self.s_down4 = spikes_conv2d(128, 256, 4)

        # -------------------- Attention Map --------------------------
        self.att_gate = Attention_block(32, 64, 16)

    def forward(self, i_y, spikes):
        s1 = self.s_down1(spikes) # 256 * 32
        s2 = self.s_down2(s1)     # 128 * 64
        s3 = self.s_down3(s2)     # 64 * 128
        s4 = self.s_down4(s3)     # 32 * 256
        
        c1 = nn.Sequential(*self.sequence[:6])(i_y) # 512
        c1spikes = torch.cat((c1, spikes), 1) # 512 * 65
        c2 = nn.Sequential(*self.sequence[6:13])(c1spikes) # 256 
        c2s1 = torch.cat((c2, s1), 1) # 256 * 128+32=160
        c3 = nn.Sequential(*self.sequence[13:23])(c2s1) # 128*256
        c3s2 = torch.cat((c3, s2), 1) # 128 * (256+64=320)            
        c4 = nn.Sequential(*self.sequence[23:33])(c3s2) # 64 * 512
        c4s3 = torch.cat((c4, s3), 1) # 64 * (512+128=640)
        c5 = nn.Sequential(*self.sequence[33:])(c4s3) # 32 * 512
        c5s4 = torch.cat((c5, s4), 1) # 32 * (512+256=768)
        ft = self.en_norm(F.relu(self.en_conv(self.en_pool(c5s4)))) # 16*1024
        d1 = self.deconv0(ft) # 32*512
        d2 = self.deconv1( # 64*512
            self.fu_conv1(torch.cat((d1, c5s4), 1)))
        d3 = self.deconv2( # 128*256
            self.fu_conv2(torch.cat((d2, c4s3), 1)))
        d4 = self.deconv3( # 256*128
            self.fu_conv3(torch.cat((d3, c3s2), 1)))
        d5 = self.deconv4( # 512*64
            self.fu_conv4(torch.cat((d4, c2s1), 1)))
        d6 = self.fu_conv6(self.fu_conv5(torch.cat((d5, c1spikes), 1))) # 32
        d7 = F.leaky_relu(self.de_norm5(d6)) # 32
        c1_am, att_map = self.att_gate(d7, c1)
        out = self.fu_conv7(torch.cat((d7, c1_am), 1))
        out = torch.sigmoid(self.final_conv2(self.final_conv1(out)))
        
        return out, att_map


# DenseSRNet
def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


class Dense_Block(nn.Module):
    def __init__(self, channel_in):
        super(Dense_Block, self).__init__()

        self.relu = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=96, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=112, out_channels=16, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(conv1))
        cout2_dense = self.relu(torch.cat([conv1,conv2], 1))
        conv3 = self.relu(self.conv3(cout2_dense))
        cout3_dense = self.relu(torch.cat([conv1,conv2,conv3], 1))
        conv4 = self.relu(self.conv4(cout3_dense))
        cout4_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4], 1))
        conv5 = self.relu(self.conv5(cout4_dense))
        cout5_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5], 1))
        conv6 = self.relu(self.conv6(cout5_dense))
        cout6_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5,conv6], 1))
        conv7 = self.relu(self.conv7(cout6_dense))
        cout7_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5,conv6,conv7], 1))
        conv8 = self.relu(self.conv8(cout7_dense))
        cout8_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5,conv6,conv7,conv8], 1))

        return cout8_dense


class Upsampling_Net(nn.Module):
    def __init__(self, scale):
        super(Upsampling_Net, self).__init__()
        self.scale = scale
        self.relu = nn.PReLU()
        self.lowlevel = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bottleneck = nn.Conv2d(in_channels=1152, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False)
        self.denseblock1 = self.make_layer(Dense_Block, 128)
        self.denseblock2 = self.make_layer(Dense_Block, 256)
        self.denseblock3 = self.make_layer(Dense_Block, 384)
        self.denseblock4 = self.make_layer(Dense_Block, 512)
        self.denseblock5 = self.make_layer(Dense_Block, 640)
        self.denseblock6 = self.make_layer(Dense_Block, 768)
        self.denseblock7 = self.make_layer(Dense_Block, 896)
        self.denseblock8 = self.make_layer(Dense_Block, 1024)
        self.pixel_shuffle = nn.PixelShuffle(2)

        if self.scale == 2:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.PReLU()
                )
            self.reconstruction = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        elif self.scale == 4:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.PReLU(),
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.PReLU()
                )
            self.reconstruction = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        elif self.scale == 8:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.PReLU(),
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
                nn.PReLU(),
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
                nn.PReLU()
                )
            self.reconstruction = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):    
        residual = self.relu(self.lowlevel(x))
        out = self.denseblock1(residual)
        concat = torch.cat([residual,out], 1)
        out = self.denseblock2(concat)
        concat = torch.cat([concat,out], 1)
        out = self.denseblock3(concat)
        concat = torch.cat([concat,out], 1)
        out = self.denseblock4(concat)
        concat = torch.cat([concat,out], 1)
        out = self.denseblock5(concat)
        concat = torch.cat([concat,out], 1)
        out = self.denseblock6(concat)
        concat = torch.cat([concat,out], 1)
        out = self.denseblock7(concat)
        concat = torch.cat([concat,out], 1)
        out = self.denseblock8(concat)
        out = torch.cat([concat,out], 1)
        out = self.bottleneck(out)
        out = self.deconv(out)
        out = self.reconstruction(out)
        base = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        out = out + base
        
        # Test 
        out = torch.clamp(out, min=-1, max=1)
        
        return out

    
