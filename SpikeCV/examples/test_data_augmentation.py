# -*- coding: utf-8 -*-
# @Time : 2022/8/6 15:19
# @Author Homepage : https://github.com/DingJianhao
# File : test_data_augmentation.py

import sys
sys.path.append('../../')
from PIL import Image
from SpikeCV.spkData.convert_img import img_to_spike
import SpikeCV.spkProc.augment as augments
from SpikeCV.spkData.data_transform import ToTorchTensor, ToNPYArray
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

for base_type in [torch.tensor, np.ndarray, ]:
    torch.manual_seed(10)
    np.random.seed(0)
    random.seed(34)

    gray_scale_img = np.array(Image.open('../examples/test_data/LenaRGB.bmp')).astype(np.float).mean(axis=2) / 255.
    spk_data = img_to_spike(gray_scale_img, gain_amp=0.5, v_th=1.0, n_timestep=10)

    if base_type == np.ndarray:
        spk_data = ToNPYArray()(spk_data)
    else:
        spk_data = ToTorchTensor()(spk_data)

    print('raw data size:', spk_data.shape)

    test_augments = {'Raw': augments.Assemble([]), # raw
                     'VerticalFlip': augments.RandomVerticalFlip(1.0),
                     'HorizontalFlip': augments.RandomHorizontalFlip(1.0),
                     'RandResize': augments.RandomResize(),
                     'Resize to 32': augments.Resize(32),
                     'CenterCrop': augments.CenterCrop(256),
                     'RandCrop': augments.RandomCrop(256),
                     'RandResizedCrop': augments.RandomResizedCrop(size=256),
                     'SpatialPad': augments.SpatialPad(padding=20, padding_mode='reflect'),
                     'TemporalPad': augments.TemporalPad(padding=(10,20), padding_mode='constant'),
                     'RandRotate': augments.RandomRotation(degrees=90),
                     'RandAffine': augments.RandomAffine(degrees=90),
                     'RandBlockErase': augments.RandomBlockErasing(),
                     'RandErase': augments.RandomSpikeErasing(p=0.5),
                     'RandAdd': augments.RandomSpikeAdding(p=0.5),
    }

    ncols = 5
    nrows = (len(test_augments) - 1) // ncols + 1

    fig, ax = plt.subplots(nrows = nrows, ncols=ncols)

    if not isinstance(ax[0], np.ndarray):
        ax = [ax]

    test_augments_keys = list(test_augments.keys())
    for i in range(len(test_augments_keys)):
        key = test_augments_keys[i]
        x = i // ncols
        y = i % ncols
        aug_spk_data = test_augments[key](spk_data)
        print(test_augments[key])
        print(aug_spk_data.shape, type(aug_spk_data), aug_spk_data.mean(axis=0).max(), aug_spk_data.mean(axis=0).min())
        ax[x][y].imshow(aug_spk_data.mean(axis=0), cmap='gray')
        # ax[x][y].imshow(aug_spk_data[2], cmap='gray')
        ax[x][y].set_title(key, fontsize=8)

    for x in range(nrows):
        for y in range(ncols):
            ax[x][y].axis('off')

    plt.show()
