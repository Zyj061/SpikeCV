import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DefromConv(nn.Module):
    '''
    Deformable Convolution Implementation 
    '''
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


class Denoise(nn.Module):
    """
    Denoise Module
    """
    def __init__(self, T):
        super(Denoise, self).__init__()
        self.T = T
        self.dt = 3
        self.size = 7
        
        # Initialize convolution layers dynamically
        self.localconvs = nn.ModuleList()
        for i in range(T):
            # For the first and last layer, use dt-1 channels; for others, use dt channels
            channels = self.dt - 1 if i == 0 or i == T-1 else self.dt
            conv = nn.Conv2d(channels, 1, self.size, padding=self.size // 2)
            self.localconvs.append(conv)

    def forward(self, x):
        outputs = []
        # Apply each convolution to the appropriate slice of x
        for i, conv in enumerate(self.localconvs):
            if i == 0:
                slice = x[:, 0:self.dt-1, ...]
            elif i == self.T-1:
                slice = x[:, -self.dt+1:, ...]
            else:
                slice = x[:, i-1:i+self.dt-1, ...]
            outputs.append(conv(slice))
        # Concatenate all outputs along the channel dimension
        x = torch.cat(outputs, axis=1)
        return x


class MotionEnhance(nn.Module):
    '''
    Motion Enhancement Module 
    '''
    def __init__(self, T):
        super(MotionEnhance, self).__init__()
        k_size = 5
        atten_size = 7
        self.deform_conv = DefromConv(T, 64, k_size, k_size//2)
        self.conv = nn.Conv2d(2, 1, kernel_size=atten_size, padding=atten_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x) :
        x = self.deform_conv(x)
        max_result,_ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        attn = torch.cat([max_result, avg_result], 1)
        attn = self.conv(attn)
        map = self.sigmoid(attn)

        return x * map + x


class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.rand((self.shape)) * 0.001, requires_grad=True)

    def forward(self, x):
        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding)

        return y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.move0 = LearnableBias(inplanes)
        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.move1 = LearnableBias(planes)
        self.prelu = nn.PReLU(planes)
        self.move2 = LearnableBias(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.move0(x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.move1(out)
        out = self.prelu(out)
        out = self.move2(out)

        return out


class DMER_Net(nn.Module):
    '''
    the implementation of Binarized Neural Network is referred from:
    https://github.com/liuzechun/Bi-Real-net/blob/master/pytorch_implementation/BiReal18_34/birealnet.py
    '''
    def __init__(self, block, layers, num_classes=10):
        super(DMER_Net, self).__init__()
        self.inplanes = 64
        self.T = 7
        self.key = 1
        if self.key == 0:
            self.module = HardBinaryConv(self.T, 64, kernel_size=7, stride=2, padding=3)
        elif self.key == 1:
            self.module = nn.Sequential(
                Denoise(self.T), 
                MotionEnhance(self.T))

        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.module(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def net18(**kwargs):
    """Constructs a DMER-Net-18 model. """
    model = DMER_Net(BasicBlock, [4, 4, 4, 4], **kwargs)
    return model


def net34(**kwargs):
    """Constructs a DMER-Net-34 model. """
    model = DMER_Net(BasicBlock, [6, 8, 12, 6], **kwargs)
    return model
