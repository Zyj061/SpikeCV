# -*- coding: utf-8 -*-
# @Time : 2024/12/05 20:17
# @Author : Yajing Zheng
# @Email: yj.zheng@pku.edu.cn
# @File : test_snntracker.py
import os, sys
sys.path.append("..")
import path

import numpy as np
from spkData.load_dat_snntrack import data_parameter_dict, SpikeStream
from pprint import pprint
import torch
from spkProc.tracking.snn_tracker import SNNTracker
from utils import vis_trajectory
from visualization.get_video import obtain_mot_video
import cv2
from tracking_mot import TrackingMetrics

from visualization.get_video import obtain_detection_video
import motmetrics as mm

# change the path to where you put the datasets
test_scene = ['spike59', 'rotTrans', 'cplCam', 'cpl1', 'badminton', 'ball']
# data_filename = 'motVidarReal2020/rotTrans'
scene_idx = 0
data_filename = '/root/autodl-fs/motVidarReal2020/' + test_scene[scene_idx]
label_type = 'tracking'
para_dict = data_parameter_dict(data_filename, label_type)
pprint(para_dict)
vidarSpikes = SpikeStream(**para_dict)

block_len = 1000
spikes = vidarSpikes.get_block_spikes(begin_idx=0, block_len=block_len)
# spikes = vidarSpikes.get_spike_matrix()
pprint(spikes.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

calibration_time = 150
filename = path.split_path_into_pieces(data_filename)
result_filename = filename[-1] + '_snn.txt'
if not os.path.exists('./results'):
    os.makedirs('./results')
tracking_file = os.path.join('./results', result_filename)

data_name = test_scene[scene_idx]

# measure the multi-object tracking performance
gt_file = os.path.join(data_filename, 'spikes_gt.txt')
gt = mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=0.5)
metrics = TrackingMetrics(tracking_file, **para_dict)
metrics.get_results()