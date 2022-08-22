# -*- coding: utf-8 -*- 
# @Time : 2022/7/20
# @Author : Rui Zhao
# @File : load_optical_flow.py

import os
import os.path as osp
import numpy as np
import torch
import glob
from spkData.load_dat import VidarSpike

class Dataset_SPIFT(torch.utils.data.Dataset):
    def __init__(self, **kwargs):

        self.filepath = kwargs.get('filepath')
        self.spike_h = kwargs.get('spike_h')
        self.spike_w = kwargs.get('spike_w')
        self.dt = kwargs.get('dt')
        # self.transform = kwargs.get('transform')
        self.kwargs_dat = kwargs
        self.kwargs_dat['print_dat_detail'] = False

        self.samples = self.collect_samples()
        print('SPIFT Length {:d}'.format(len(self.samples)))
    
    # get paths of all the (spike, flow_gt) pairs
    def collect_samples(self):
        scene_list = list(range(0, 100))
        samples = []

        for scene in scene_list:
            spike_dir = osp.join(self.filepath, str(scene), 'spike_dt{:d}'.format(self.dt))
            flowgt_dir = osp.join(self.filepath, str(scene), 'dt={:d}'.format(self.dt), 'flow')
            for st in range(0, len(glob.glob(spike_dir + '/*.dat')) - 1):
                seq1_path = osp.join(spike_dir, str(int(st)))
                seq2_path = osp.join(spike_dir, str(int(st+1)))
                flow_path = osp.join(flowgt_dir, '{:04d}.flo'.format(int(st)))
                
                if path_list_exist([seq1_path+'.dat', seq2_path+'.dat', flow_path]):
                    s = {}
                    s['seq1_path'], s['seq2_path'], s['flow_path'] = seq1_path, seq2_path, flow_path
                    samples.append(s)
        return samples
    
    def _load_sample(self, s):
        self.kwargs_dat['filepath'] = s['seq1_path']
        self.vidarSpikes = VidarSpike(**self.kwargs_dat)
        seq1 = self.vidarSpikes.get_spike_matrix(flipud=False).astype(np.float32)

        self.kwargs_dat['filepath'] = s['seq2_path']
        self.vidarSpikes = VidarSpike(**self.kwargs_dat)
        seq2 = self.vidarSpikes.get_spike_matrix(flipud=False).astype(np.float32)

        flow = readFlow(s['flow_path']).astype(np.float32)

        y0 = np.random.randint(0, 20)
        seq1 = seq1[:, y0:y0+480, :]
        seq2 = seq2[:, y0:y0+480, :]
        flow = flow[y0:y0+480, :, :]

        return seq1, seq2, flow

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        seq1, seq2, flow = self._load_sample(self.samples[index])
        return seq1, seq2, flow



class Dataset_PHM(torch.utils.data.Dataset):
    def __init__(self, **kwargs):

        self.filepath = kwargs.get('filepath')
        self.spike_h = kwargs.get('spike_h')
        self.spike_w = kwargs.get('spike_w')
        self.dt = kwargs.get('dt')
        self.scene = kwargs.get('scene')
        self.kwargs_dat = kwargs
        self.kwargs_dat['print_dat_detail'] = False

        self.samples = self.collect_samples()
        print('PHM Scene: {:s} Length: {:d}'.format(self.scene, len(self.samples)))
    
    # get paths of all the (spike, flow_gt) pairs
    def collect_samples(self):
        scene_list = [self.scene]
        samples = []

        for scene in scene_list:
            spike_dir = osp.join(self.filepath, str(scene), 'spike_dt{:d}'.format(self.dt))
            flowgt_dir = osp.join(self.filepath, str(scene), 'dt={:d}'.format(self.dt), 'flow')
            for st in range(0, len(glob.glob(spike_dir + '/*.dat')) - 1):
                seq1_path = osp.join(spike_dir, str(int(st)))
                seq2_path = osp.join(spike_dir, str(int(st+1)))
                flow_path = osp.join(flowgt_dir, '{:04d}.flo'.format(int(st)))
                
                if path_list_exist([seq1_path+'.dat', seq2_path+'.dat', flow_path]):
                    s = {}
                    s['seq1_path'], s['seq2_path'], s['flow_path'] = seq1_path, seq2_path, flow_path
                    samples.append(s)
        return samples
    
    def _load_sample(self, s):
        self.kwargs_dat['filepath'] = s['seq1_path']
        self.vidarSpikes = VidarSpike(**self.kwargs_dat)
        seq1 = self.vidarSpikes.get_spike_matrix().astype(np.float32)

        self.kwargs_dat['filepath'] = s['seq2_path']
        self.vidarSpikes = VidarSpike(**self.kwargs_dat)
        seq2 = self.vidarSpikes.get_spike_matrix().astype(np.float32)

        flow = readFlow(s['flow_path']).astype(np.float32)

        return seq1, seq2, flow

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        seq1, seq2, flow = self._load_sample(self.samples[index])
        return seq1, seq2, flow




################### utils ###################
def path_list_exist(path_list):
    for pp in path_list:
        if not osp.exists(pp):
            return False
    return True


def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))