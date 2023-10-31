# -*- coding: utf-8 -*- 
# @Time : 2023/2/15 13:48 
# @Author : Yajing Zheng
# @Email: yj.zheng@pku.edu.cn
# @File : test_pkuvidar_ssort.py
import os, sys
sys.path.append("..")

import numpy as np
from spkData.load_dat import data_parameter_dict, SpikeStream
from utils import path
from pprint import pprint
from spkProc.tracking.spike_sort import SpikeSORT
import torch
from visualization.get_video import obtain_detection_video, vis_trajectory
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

data_filename = 'PKU-Vidar-DVS/train/Vidar/00046_rotation_5000K_200r'
label_type = 'detection'
para_dict = data_parameter_dict(data_filename, label_type)
pprint(para_dict)
vidarSpikes = SpikeStream(**para_dict)

spikes = vidarSpikes.get_spike_matrix()
pprint(spikes.shape)

device = torch.device('cuda')

calibration_time = 150
filename = path.split_path_into_pieces(data_filename)
result_filename = filename[-2] + '_spikeSort.txt'
if not os.path.exists('results'):
    os.makedirs('results')
tracking_file = os.path.join('results', result_filename)
if os.path.exists(tracking_file):
    os.remove(tracking_file)

spike_tracker = SpikeSORT(para_dict.get('spike_h'), para_dict.get('spike_w'), device)
total_spikes = spikes

# using stp filter to filter out static spikes
spike_tracker.calibrate_motion(spikes, calibration_time)
# start tracking
spike_tracker.get_results(spikes[calibration_time:], tracking_file)
evaluate_seq_len = 5

if para_dict['filelist'] is not None:
    total_seq_len = len(para_dict['filelist'])
    for i_seq in range(evaluate_seq_len):

        para_dict['filepath'] = para_dict['filelist'][i_seq+1]
        vidarSpikes = SpikeStream(**para_dict)
        spikes = vidarSpikes.get_spike_matrix()
        spike_tracker.get_results(spikes, tracking_file)
        total_spikes = np.concatenate((total_spikes, spikes), axis=0)

trajectories_filename = os.path.join('results', filename[-2] + '.json')
visTraj_filename = os.path.join('results', filename[-2] + '.png')

spike_tracker.save_trajectory(trajectories_filename)

vis_trajectory(tracking_file, trajectories_filename, visTraj_filename, **para_dict)
block_len = total_spikes.shape[0]
# visualize the tracking results to a video
video_filename = os.path.join('results', filename[-2] + '_mot.avi')
#obtain_mot_video(spike_tracker.filterd_spikes, video_filename, tracking_file, **paraDict)
# obtain_detection_video(total_spikes, video_filename, tracking_file, **para_dict)
# obtain_detection_video(np.array(spike_tracker.filterd_spikes), video_filename, tracking_file, **para_dict)


