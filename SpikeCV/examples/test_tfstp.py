# -*- coding: utf-8 -*- 
# @Time : 2022/7/13 22:00 
# @Author : Yajing Zheng
# @File : test_tfstp.py
import os, sys
import torch
sys.path.append("..")

from spkData.load_dat import data_parameter_dict
from spkData.load_dat import SpikeStream
from spkProc.reconstruction.tfstp import TFSTP
from visualization.get_video import obtain_reconstruction_video
from utils import path
from pprint import pprint

data_filename = "recVidarReal2019/classA/car-100kmh"
label_type = 'raw'

paraDict = data_parameter_dict(data_filename, label_type)
pprint(paraDict)

# initial SpikeStream object for format input data
vidarSpikes = SpikeStream(**paraDict)

block_len = 1500
spikes = vidarSpikes.get_block_spikes(begin_idx=500, block_len=block_len)

device = torch.device('cuda')
reconstructor = TFSTP(paraDict.get('spike_h'), paraDict.get('spike_w'), device)

recImg = reconstructor.spikes2images_offline(spikes)

#recImg = reconstructor.spikes2images_online(spikes)

filename = path.split_path_into_pieces(data_filename)
if not os.path.exists('results'):
    os.makedirs('results')

result_filename = os.path.join('results', filename[-1] + '_tfstp.avi')
obtain_reconstruction_video(recImg, result_filename, **paraDict)


