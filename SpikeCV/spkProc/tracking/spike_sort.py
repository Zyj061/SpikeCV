# -*- coding: utf-8 -*- 
# @Time : 2022/8/8 15:12 
# @Author : Yajing Zheng
# @Email: yj.zheng@pku.edu.cn
# @File : spike_sort.py
import os, sys
sys.path.append('../..')
import time
import numpy as np
import torch

from spkProc.filters.stp_filters_torch import STPFilter
from spkProc.detection.attention_select import SaccadeInput
from utils.encoder import NumpyEncoder
"""
The specific process of SORT processing is described in this article (https://arxiv.org/abs/1602.00763). 
Here we use the reproduced version of abewley. Please download sort.py from https://github.com/abewley/sort and place it in the same directory as this code. 
You can also use other implementations of SORT.
You can use other implementations of SORT.
"""
from sort import Sort
from collections import namedtuple
import json

trajectories = namedtuple('trajectories', ['id', 'x', 'y', 't', 'color'])

class SpikeSORT:

    def __init__(self, spike_h, spike_w, device):

        self.spike_h = spike_h
        self.spike_w = spike_w
        self.device = device

        self.stp_filter = STPFilter(spike_h, spike_w, device)
        self.object_detection = SaccadeInput(spike_h, spike_w, box_size=20, device=device)

        self.mot_tracker = Sort(max_age=2000, min_hits=3, iou_threshold=0.3)
        # self.timestamps = spikes.shape[0]
        # self.filterd_spikes = np.zeros([self.timestamps, self.spike_h, self.spike_w], np.uint8)
        self.calibration_time = 150
        self.timestamps = 0
        self.trajectories = {}
        self.filterd_spikes = []

    def calibrate_motion(self, spikes, calibration_time=None):

        if calibration_time is None:
            calibration_time = self.calibration_time
        else:
            self.calibration_time = calibration_time

        print('begin calibrate..')
        for t in range(calibration_time):
            input_spk = torch.from_numpy(spikes[t, :, :]).to(self.device)
            self.stp_filter.update_dynamics(t, input_spk)
            self.timestamps += 1

    def get_results(self, spikes, res_filepath):

        result_file = open(res_filepath, 'a+')
        timestamps = spikes.shape[0]
        total_time = 0
        for t in range(timestamps):
            try:
                input_spk = torch.from_numpy(spikes[t, :, :]).to(self.device)
                self.stp_filter.update_dynamics(self.timestamps, input_spk)

                self.stp_filter.local_connect(self.stp_filter.filter_spk)
                # self.filterd_spikes[t, :, :] = self.stp_filter.lif_spk.cpu().detach().numpy()

                self.filterd_spikes.append(self.stp_filter.lif_spk.cpu().detach().numpy())

                self.object_detection.update_dnf(self.stp_filter.lif_spk)
                attentionBox = self.object_detection.get_attention_location()
                num_box = attentionBox.shape[0]
                dets = torch.zeros((num_box, 6), dtype=torch.int)
                for i_box, bbox in enumerate(attentionBox):
                   dets[i_box, :] = torch.tensor([bbox[0], bbox[1], bbox[2], bbox[3], 1, 1])

                start_time = time.time()
                trackers = self.mot_tracker.update(dets)
                cycle_time = time.time() - start_time
                total_time += cycle_time
                track_ids = []
                for d in trackers:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                        self.timestamps, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]), file=result_file)
                    track_ids.append(d[4])

                    # update the trajectories
                    mid_x = (d[0] + d[2]) / 2
                    mid_y = (d[1] + d[3]) / 2
                    if d[4] not in self.trajectories:
                        self.trajectories[d[4]] = trajectories(int(d[4]), [], [], [], 255 * np.random.rand(1, 3))
                        self.trajectories[d[4]].x.append(mid_x)
                        self.trajectories[d[4]].y.append(mid_y)
                        self.trajectories[d[4]].t.append(self.timestamps)

                    else:
                        self.trajectories[d[4]].x.append(mid_x)
                        self.trajectories[d[4]].y.append(mid_y)
                        self.trajectories[d[4]].t.append(self.timestamps)

                self.timestamps += 1

            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print('WARNING: out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    else:
                        raise exception

        print('Total tracking took: %.3f seconds for %d timestamps spikes' %
              (total_time, self.timestamps - self.calibration_time))

        result_file.close()

    def save_trajectory(self, trajectories_filename):
        if os.path.exists(trajectories_filename):
            os.remove(trajectories_filename)

        num_len = len(self.trajectories)
        for i_traj in self.trajectories:
            traj_json_string = json.dumps(self.trajectories[i_traj]._asdict(), cls=NumpyEncoder)

            with open(trajectories_filename, 'a+') as f:
                f.write(traj_json_string)

                f.write('\n')
