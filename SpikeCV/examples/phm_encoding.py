# -*- coding: utf-8 -*- 
# @Time : 2022/7/22
# @Author : Rui Zhao
# @File : phm_encoding.py

import sys
sys.path.append("..")

import os
import os.path as osp
import argparse
import numpy as np
import cv2
import time

from spkData.load_dat import SpikeStream
from spkData.save_dat import SpikeToRaw

parser = argparse.ArgumentParser( )
parser.add_argument('-dr', '--data_root', type=str, default='../spkData/datasets/OpticalFlowPHM', help='Root path of the data')
parser.add_argument('-dn', '--data_name', type=str, default='test', help='Name of the data file')
parser.add_argument('-fl', '--flipud', action='store_true', help='Flip the raw spike')
parser.add_argument('-dt', '--dt', type=int, default=10, help='dt')
parser.add_argument('-l', '--data_length', type=int, default=25, help='length of spike sequence for each group')
args = parser.parse_args()


scene_list = os.listdir(args.data_root)

for scene in scene_list:
    data_path = os.path.join(args.data_root, scene, args.data_name)
    save_path = os.path.join(args.data_root, scene, 'spike_dt{:d}'.format(args.dt))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    paraDict = {}
    paraDict['filepath'] = data_path
    paraDict['spike_h'] = 500
    paraDict['spike_w'] = 800
    paraDict['print_dat_detail'] = False

    vidarspike = SpikeStream(**paraDict)
    raw_spike = vidarspike.get_spike_matrix()

    c, h, w = raw_spike.shape
    data_step = args.dt
    half_length = (args.data_length-1) // 2

    ii = 0
    while True:
        central_index = ii * data_step
        st_index = central_index - half_length
        ed_index = central_index + half_length + 1


        if (ed_index >= c - 40):
            break

        if (central_index < 40):
            ii += 1
            continue

        cur_save_path = osp.join(save_path, str(ii)+'.dat')
        cur_spike = raw_spike[st_index:ed_index, :, :]
        SpikeToRaw(cur_save_path, cur_spike, filpud=False)
        print('Finish process {:s} Scene {:s} #{:04d} sampling: length={:02d} dt{:d}'.format(args.data_root, scene, ii, args.data_length, args.dt))
        
        ii += 1