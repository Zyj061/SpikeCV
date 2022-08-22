# -*- coding: utf-8 -*- 
# @Time : 2022/7/22
# @Author : Rui Zhao
# @File : reconstruction.py

import numpy as np
import torch
import torch.nn as nn
import torchvision
import cv2
import math

# ============================================
# Image Quality Assessment with Reference
# --------------------------------------------
# 1. PSNR
# 2. SSIM
# 3. LPIPS
# ============================================


# --------------------------------------------
# PSNR
# --------------------------------------------
def calculate_psnr(img1, img2, border=0):
    '''
    calculate PSNR
    img1 and img2 have range [0, 255]
    (H x W) or (H x W x 3) numpy array
    From https://github.com/cszn/KAIR/blob/master/utils/utils_image.py
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, border=0):
    '''
    calculate SSIM
    img1 and img2 have range [0, 255]
    (H x W) or (H x W x 3) numpy array
    From https://github.com/cszn/KAIR/blob/master/utils/utils_image.py
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# --------------------------------------------
# LPIPS
# --------------------------------------------
class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class LPIPS:
    def __init__(self, device='cuda'):
        self.device = device
        if not torch.cuda.is_available():
            self.device = 'cpu'
            print('There is no GPU available. The LPIPS will be calculated on CPU')

        self.vgg = VGG19().to(self.device).eval()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        # self.criterion = nn.L1Loss(size_average=True)
        self.criterion = nn.L1Loss(reduction='mean')
    
    def calculate_lpips_torch(self, pred, y):
        x_vgg, y_vgg = self.vgg(pred), self.vgg(y.detach())
        vgg_loss = 0.0
        for i in range(len(x_vgg)):
            vgg_loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return vgg_loss
    
    def calculate_lpips(self, img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        
        if img1.ndim == 3:
            img1 = torch.from_numpy(img1).unsqueeze_(dim=0).permute([0,3,1,2]).to(self.device).float() / 255.0
            img2 = torch.from_numpy(img2).unsqueeze_(dim=0).permute([0,3,1,2]).to(self.device).float() / 255.0
        elif img1.ndim == 2:
            img1 = torch.from_numpy(img1).unsqueeze_(dim=0).unsqueeze_(dim=0).repeat([1,3,1,1]).to(self.device).float() / 255.0
            img2 = torch.from_numpy(img2).unsqueeze_(dim=0).unsqueeze_(dim=0).repeat([1,3,1,1]).to(self.device).float() / 255.0

        lpips = self.calculate_lpips_torch(img1, img2)
        return lpips