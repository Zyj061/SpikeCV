# -*- coding: utf-8 -*-
# @Time : 2022/8/5 16:08
# @Author Homepage : https://github.com/DingJianhao
# File : img_sim_dataset.py

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets import MNIST
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
from SpikeCV.spkData.convert_img import img_to_spike
import numpy as np


class SpikeMNIST(VisionDataset):
    def __init__(self,
            root: str,
            gain_amp = 0.5,
            v_th = 1.0,
            timesteps = 100,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        '''
        图片转脉冲MNIST数据集

        :param root: 数据集目录
        :param gain_amp: 增益系数
        :param v_th: 阈值
        :param timesteps: 时间步
        :param train:
        :param transform:
        :param target_transform:
        :param download:
        '''
        super(SpikeMNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.mnist = MNIST(root=root,
                             train=train,
                             transform=None,
                             target_transform=target_transform,
                             download=download)
        self.gain_amp = gain_amp
        self.v_th = v_th
        self.n_timesteps = timesteps

    def __len__(self) -> int:
        return len(self.mnist.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.mnist[index]
        sim_spike_matrix = img_to_spike(np.array(img) / 255.0,
                                        gain_amp=self.gain_amp,
                                        v_th=self.v_th,
                                        n_timestep=self.n_timesteps)
        sim_spike_matrix = self.transform(sim_spike_matrix)
        return sim_spike_matrix.copy(), target

