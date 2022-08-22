# -*- coding: utf-8 -*- 
# @Time : 2022/7/21
# @Author : Rui Zhao
# @File : test_tfp.py
import os
import torch
import sys
sys.path.append("..")

import time

from spkData.load_dat import data_parameter_dict
from spkData.load_dat import VidarSpike
from spkProc.reconstruction.tfi import TFI
from visualization.get_video import obtain_reconstruction_video
from utils import path

# 指定数据序列及任务类型
data_filename = "recVidarReal2019/classA/car-100kmh"
label_type = 'raw'

# 加载数据集属性字典
paraDict = data_parameter_dict(data_filename, label_type)

#加载脉冲数据
vidarSpikes = VidarSpike(**paraDict)

block_len = 500
spikes = vidarSpikes.get_block_spikes(begin_idx=0, block_len=block_len)

device = torch.device('cpu')
reconstructor = TFI(paraDict.get('spike_h'), paraDict.get('spike_w'), device)

st = time.time()
recImg = reconstructor.spikes2images(spikes)
ed = time.time()
print('shape: ', recImg.shape, 'time: {:.6f}'.format(ed - st))

filename = path.split_path_into_pieces(data_filename)
if not os.path.exists('results'):
    os.makedirs('results')

result_filename = os.path.join('results', filename[-1] + '_tfi.avi')
obtain_reconstruction_video(recImg, result_filename, **paraDict)


