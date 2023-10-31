'''
Descripttion: 
version: 
Author: Jiyuan Zhang
Date: 2022-12-05 10:53:27
LastEditors: Jiyuan Zhang
LastEditTime: 2022-12-07 12:48:02
'''
# -*- coding: utf-8 -*- 
# @Time : 2022/11/28 13:40 
# @Author : Yajing Zheng
# @Email: yj.zheng@pku.edu.cn
# @File : test_device.py
import os
import sys
import ctypes
sys.path.append('..')
from spkData.load_dat_jy import device_parameters, SpikeStream
from visualization.get_video import obtain_spike_video
from sdk import spikelinkapi as link

# 全局变量 最先初始化的配置
params = link.SpikeLinkInitParams()
params_camera = link.SpikeLinkQSFPInitParams()
params_dummy = link.SpikeLinkDummyInitParams()

DEBUG_OUT = False
brunning = False
count = 0
pool_len = 500
input_c = link.spikelinkInput("../device/spikevision/m1k40/sdk/lib/Debug/spikelinkapi.dll")
input_c.linkinputlib.ReleaseFrame.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
framepool = link.spikeframepool()


def inputcallback(frame):
    global count

    if count > pool_len:
        return

    framepool.push(frame)
    if DEBUG_OUT:
        frame2 = ctypes.cast(frame, ctypes.POINTER(link.SpikeLinkVideoFrame))
        print("get frame:", frame2.contents.size, frame2.contents.width, frame2.contents.height,
                frame2.contents.pts)
    if count % 100 == 0:
        frame2 = ctypes.cast(frame, ctypes.POINTER(link.SpikeLinkVideoFrame))
        print("index:", frame2.contents.pts)
    count += 1


input_callback = link.LinkInputCallBack(inputcallback)
type = 0 # dummy: 0, online camera: 1
filename = "F:\\datasets\\0824wave.bin"
# filename = 'C:/Users/jyzhang/Work/research/SpikeCV/wrj2.bin'
decode_width = 1024
spike_width = 1000
height = 1000
paraDict = {'decode_width': decode_width, 'spike_width': spike_width, 'height': height}

device_params = device_parameters(params, params_dummy, type, filename, decode_width, height)
input_c.init(ctypes.byref(device_params))   
input_c.open()
input_c.setcallback(input_callback)
# paraDict['params'] = device_params

vidarSpikes = SpikeStream(offline=False, **paraDict)
spikes = vidarSpikes.get_device_matrix(_input=input_c, _framepool=framepool, block_len=pool_len)
print(spikes.shape)

input_c.close()

if not os.path.exists('./results'):
    os.makedirs('results')
spike_filename = os.path.join('results', 'test_device_jy.avi')
save_paraDict = {'spike_h': height, 'spike_w': spike_width}

obtain_spike_video(spikes, spike_filename, **save_paraDict)
