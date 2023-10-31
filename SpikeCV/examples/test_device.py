'''
Descripttion: 
version: 
Author: Jiyuan Zhang
Date: 2022-12-05 10:53:27
LastEditors: Jiyuan Zhang
LastEditTime: 2022-12-06 11:28:09
'''
# -*- coding: utf-8 -*- 
# @Time : 2022/11/28 13:40 
# @Author : Yajing Zheng
# @Email: yj.zheng@pku.edu.cn
# @File : test_device.py
import os
import sys
sys.path.append('..')
from spkData.load_dat import device_parameters, SpikeStream
from visualization.get_video import obtain_spike_video
from sdk import spikelinkapi as link

type = 0 # dummy: 0, online camera: 1
filename = "F:\\datasets\\0824wave.bin"
# filename = 'F:\\Data\\wrj\\wrj2.bin'
decode_width = 1024
spike_width = 1000
height = 1000
paraDict = {'decode_width': decode_width, 'spike_width': spike_width, 'height': height}

params = link.SpikeLinkInitParams()
params_camera = link.SpikeLinkQSFPInitParams()
params_dummy = link.SpikeLinkDummyInitParams()

device_params = device_parameters(params, params_dummy, type, filename, decode_width, height)
paraDict['params'] = device_params
vidarSpikes = SpikeStream(offline=False, **paraDict)

spikes = vidarSpikes.get_device_matrix()
print(spikes.shape)

if not os.path.exists('./results'):
    os.makedirs('results')
spike_filename = os.path.join('results', 'test_device.avi')
save_paraDict = {'spike_h': height, 'spike_w': spike_width}

vidarSpikes.save_spikes("./results/test_device.dat")

obtain_spike_video(spikes, spike_filename, **save_paraDict)
