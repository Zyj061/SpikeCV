# -*- encoding: utf-8 -*-
'''
@File    :   test_load.py
@Time    :   2022/07/26 10:04:22
@Author  :   Jiyuan Zhang 
@Version :   0.1
@Contact :   jyzhang@stu.pku.edu.cn
'''

# here put the import lib
import sys
sys.path.append("..")

import numpy as np
from spkData import load_dat
from spkData.load_dat_jy import SpikeStream
from utils import path
from pprint import pprint

data_filename1 = 'motVidarReal2020/spike59/'
label_type = 'detection'
para_dict = load_dat.data_parameter_dict(data_filename1, label_type)
pprint(para_dict)
# RESULT：
# {'filepath': '..\\spkData\\datasets\\motVidarReal2020\\spike59\\spikes.dat',
#  'labeled_data_dir': '..\\spkData\\datasets\\motVidarReal2020\\spike59\\spikes_gt.txt',
#  'labeled_data_suffix': 'txt',
#  'labeled_data_type': [4, 5],
#  'spike_h': 250,
#  'spike_w': 400}
s = SpikeStream(offline=True, **para_dict)
s.get_block_spikes(0, 200)
print(isinstance(s.SpikeMatrix, np.ndarray))
s.to_tensor()
print(isinstance(s.SpikeMatrix, np.ndarray))
s.to_numpy()
print(isinstance(s.SpikeMatrix, np.ndarray))
print(s.SpikeMatrix.__class__)
print(s.SpikeMatrix.shape)
# RESULT:
# True
# False
# True
# <class 'numpy.ndarray'>
# (200, 250, 400)

data_filename2 = 'Spike-Stero/indoor/left/0000/0000/'
label_type = 'stero_depth_estimation'
para_dict = load_dat.data_parameter_dict(data_filename2, label_type)
pprint(para_dict)
# RESULT：
# {'filepath': '..\\spkData\\datasets\\Spike-Stero\\indoor\\left\\0000\\0000\\0000.dat',
#  'labeled_data_dir': '..\\spkData\\datasets\\Spike-Stero\\indoor\\left\\0000\\0000\\0000_gt.npy',
#  'labeled_data_suffix': 'npy',
#  'labeled_data_type': [3.2],
#  'spike_h': 250,
#  'spike_w': 400}

data_filename3 = 'PKU-Vidar-DVS/train/Vidar/00152_driving_outdoor3/1.dat'
label_type = 'detection'
para_dict = load_dat.data_parameter_dict(data_filename3, label_type)
pprint(para_dict)
# RESULT：
# {'filepath': '..\\spkData\\datasets\\PKU-Vidar-DVS\\train\\Vidar\\00152_driving_outdoor3\\1.dat',
#  'labeled_data_dir': '..\\spkData\\datasets\\PKU-Vidar-DVS\\train\\labels\\00152_driving_outdoor3\\1.txt',
#  'labeled_data_suffix': 'txt',
#  'labeled_data_type': [4],
#  'spike_h': 250,
#  'spike_w': 400}

data_filename3 = 'Spike-Stero/indoor/left/0000/0000/spikes.dat'
pieces = path.split_path_into_pieces(data_filename3)
print(pieces)
# RESULT：
# ['Spike-Stero', 'indoor', 'left', '0000', '0000', 'spikes.dat']

data_filename3 = r'Spike-Stero\indoor\left\0000\0000\spikes.dat'
pieces = path.split_path_into_pieces(data_filename3)
print(pieces)
# RESULT：
# ['Spike-Stero', 'indoor', 'left', '0000', '0000', 'spikes.dat']

data_filename3 = 'Spike-Stero\\indoor\\left\\0000\\0000\\spikes.dat'
pieces = path.split_path_into_pieces(data_filename3)
print(pieces)
# RESULT：
# ['Spike-Stero', 'indoor', 'left', '0000', '0000', 'spikes.dat']