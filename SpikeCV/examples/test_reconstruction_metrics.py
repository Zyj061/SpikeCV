# -*- coding: utf-8 -*- 
# @Time : 2022/7/22
# @Author : Rui Zhao
# @File : test_reconstruction_metrics.py

import sys
sys.path.append("..")

import os
import os.path as osp
import numpy as np
import cv2
from metrics.reconsturction import calculate_psnr, calculate_ssim, LPIPS


def to_np255uint8(im):
    return (im * 255).astype(np.uint8)


im_path = osp.join('test_data', 'LenaRGB.bmp')
np.random.seed(seed=2728)
noise_level_list = [5, 10, 15, 20]
lpips = LPIPS()

im_rgb = cv2.imread(im_path).astype(np.float32) / 255.0
im_gry = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

for noise_std in noise_level_list:
    im_rgb_n = im_rgb + np.random.normal(0, noise_std / 255.0, im_rgb.shape)
    im_gry_n = im_gry + np.random.normal(0, noise_std / 255.0, im_gry.shape)

    im_rgb = to_np255uint8(im_rgb)
    im_gry = to_np255uint8(im_gry)
    im_rgb_n = to_np255uint8(im_rgb_n)
    im_gry_n = to_np255uint8(im_gry_n)

    # calculate metrics
    psnr_rgb = calculate_psnr(im_rgb, im_rgb_n)
    psnr_gry = calculate_psnr(im_gry, im_gry_n)

    ssim_rgb = calculate_ssim(im_rgb, im_rgb_n)
    ssim_gry = calculate_ssim(im_gry, im_gry_n)

    lpips_rgb = lpips.calculate_lpips(im_rgb, im_rgb_n)
    lpips_gry = lpips.calculate_lpips(im_gry, im_gry_n)

    out_str1 = 'Gaussian Noise sigma {:2d}.    RGB  Image,  PSNR {:6.4f},  SSIM {:6.4f},  LPIPS {:6.4f}'.format(noise_std, psnr_rgb, ssim_rgb, lpips_rgb)
    out_str2 = 'Gaussian Noise sigma {:2d}.    Gray Image,  PSNR {:6.4f},  SSIM {:6.4f},  LPIPS {:6.4f}'.format(noise_std, psnr_gry, ssim_gry, lpips_gry)
    print(out_str1 + '\n' + out_str2 + '\n')