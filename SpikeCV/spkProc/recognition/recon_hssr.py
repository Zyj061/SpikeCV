import torch
from torch import nn as nn
from torch.nn import functional as F
import torchvision
import numpy as np
import torchvision.models as models


class DefromConv(nn.Module):
    def __init__(self, in_channel, out_channel, k_size, pad):
        super(DefromConv, self).__init__()
        self.s = k_size
        self.pad = pad
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=self.s, stride=1, padding=1) 
 
        self.conv_offset = nn.Conv2d(in_channel, 2*self.s*self.s, kernel_size=self.s, stride=1, padding=self.s//2)
        init_offset = torch.Tensor(np.zeros([2*self.s*self.s, in_channel, self.s, self.s]))
        self.conv_offset.weight = torch.nn.Parameter(init_offset)
 
        self.conv_mask = nn.Conv2d(in_channel, self.s*self.s, kernel_size=self.s, stride=1, padding=self.s//2)
        init_mask = torch.Tensor(np.zeros([self.s*self.s, in_channel, self.s, self.s])+np.array([0.5]))
        self.conv_mask.weight = torch.nn.Parameter(init_mask) 
 
    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x)) 
        out = torchvision.ops.deform_conv2d(input=x, offset=offset, 
                                            weight=self.conv.weight, 
                                            mask=mask, padding=(self.pad, self.pad))
        return out


class MotionEnhance(nn.Module):
    def __init__(self, T):
        super(MotionEnhance, self).__init__()
        k_size = 5
        atten_size = 7
        self.deform_conv = DefromConv(T, 64, k_size, k_size//2)
        self.conv = nn.Conv2d(2, 1, kernel_size=atten_size, padding=atten_size//2)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(64)
    
    def forward(self, x) :
        x = self.deform_conv(x)
        max_result,_ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        attn = torch.cat([max_result, avg_result], 1)
        attn = self.conv(attn)
        map = self.sigmoid(attn)
        x = x * map + x
        x = self.bn(x)

        return x



class Encoder(nn.Module):
    def __init__(self, in_channel=20):
        super(Encoder, self).__init__()

        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 64, (in_channel,3,3), stride=1, padding=(0,1,1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

        self.layer_1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.layer_4 = nn.Sequential(
            nn.Conv2d(256+128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.layer_5 = nn.Sequential(
            nn.Conv2d(128+64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.pooling = nn.MaxPool2d((2,2))
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):

        x = self.conv3d(x)
        x = x.squeeze(dim=2)

        x = self.layer_1(x)
        out_1 = x

        x = self.pooling(x)
        x = self.layer_2(x)
        out_2 = x

        x = self.pooling(x)
        x = self.layer_3(x)

        x = F.interpolate(x, scale_factor=(2,2),mode ='bilinear')
        x = torch.cat([out_2, x], dim=1)
        x = self.layer_4(x)

        x = F.interpolate(x, scale_factor=(2,2),mode ='bilinear')
        x = torch.cat([out_1, x], dim=1)
        x = self.layer_5(x)  

        x = self.bn(x)

        return x


class HSSR_Net(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.ext = MotionEnhance(5)
        self.ext.load_state_dict(torch.load('./ckpts/extractor.pth'))

        self.enc = Encoder()
        self.enc.load_state_dict(torch.load('./ckpts/encoder.pth'))

        self.dec = models.resnet34(pretrained=True)
        self.dec.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=0, bias=False)
        self.dec.fc = nn.Linear(512, num_classes)

    def forward(self, input, tgt_img):
        tgt_fea = self.ext(tgt_img) 
        pred_fea = self.enc(input)
        pred_label = self.dec(pred_fea)

        return pred_label, pred_fea, tgt_fea
