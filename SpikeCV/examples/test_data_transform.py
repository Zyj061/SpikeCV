# -*- coding: utf-8 -*-
# @Time : 2022/8/6 14:12
# @Author Homepage : https://github.com/DingJianhao
# File : test_data_transform.py

import sys
sys.path.append('../../')
import SpikeCV.spkData.data_transform as transform
import numpy as np
import torch

ndarray_spike_matrix = np.random.randint(2, size=(100, 32, 32)) # 生成长度为100，宽高为32的numpy数组脉冲序列

# np.ndarray -> torch.tensor
tensor_spike_matrix = transform.ToTorchTensor(type=torch.FloatTensor)(ndarray_spike_matrix)
print(tensor_spike_matrix.shape, type(tensor_spike_matrix))

# torch.tensor -> np.ndarray
ndarray_spike_matrix = transform.ToNPYArray(type=np.float)(tensor_spike_matrix)
print(ndarray_spike_matrix.shape, type(ndarray_spike_matrix))
