# -*- coding: utf-8 -*- 
import os
import sys

sys.path.append('..')
from spkData.load_dat import device_parameters, SpikeStream
from visualization.get_video import obtain_spike_video, obtain_reconstruction_video
# from sdk import spikelinkapi as link
from spkProc.reconstruction.tfi import TFI
import torch

type = 0  # dummy: 0, online camera: 1
filename = "G:\\高铁数据\\merged.dat"  # 制定加载百万像素相机的绝对路径

decode_width = 1024
spike_width = 1000
height = 1000
paraDict = {'decode_width': decode_width, 'spike_width': spike_width, 'height': height}
paraDict['filepath'] = filename

vidarSpikes = SpikeStream(offline=True, camera_type='PCIE', **paraDict)

block_len = 1000
spikes = vidarSpikes.get_file_matrix(begin_idx=100, block_len=block_len)
# spikes = vidarSpikes.get_device_matrix()
print(spikes.shape)
if not os.path.exists('./results'):
    os.makedirs('results')
spike_filename = os.path.join('results', 'test_device.avi')
save_paraDict = {'spike_h': height, 'spike_w': spike_width}

# obtain_spike_video(spikes, spike_filename, **save_paraDict) #为验证脉冲流解码是否正确可重构脉冲流视频

device = torch.device('cpu')  # 如果有GPU可以切换为cuda模式
reconstructor = TFI(paraDict.get('height'), paraDict.get('spike_width'), device)
recImg = reconstructor.spikes2images(spikes)

result_filename = os.path.join('results', 'train1111.avi')
obtain_reconstruction_video(recImg, result_filename, **save_paraDict)
