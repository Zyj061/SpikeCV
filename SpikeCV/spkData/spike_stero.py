'''
Descripttion: 
version: 
Author: Jiyuan Zhang
Date: 2022-08-03 11:35:00
LastEditors: Jiyuan Zhang
LastEditTime: 2022-08-10 11:39:12
'''
# -*- encoding: utf-8 -*-
'''
@File    :   spike_stero.py
@Time    :   2022/07/26 20:31:27
@Author  :   Jiyuan Zhang 
@Version :   0.0
@Contact :   jyzhang@stu.pku.edu.cn
'''

# here put the import lib
import os
import torch
import numpy as np

from spkData.load_dat import data_parameter_dict, SpikeStream


class DatasetSpikeStero(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        super(DatasetSpikeStero, self).__init__()
        self.split = kwargs.get('split')
        assert 'train' in self.split or 'val' in self.split, "The \'split\' should be \'train\' of \'val\'"

        self.rootpath = kwargs.get('filepath')
        self.spike_h = kwargs.get('spike_h', 250)
        self.spike_w = kwargs.get('spike_w', 400)
        
        self.scence = kwargs.get('scence')
        assert self.scence in ['indoor', 'outdoor', 'both'], "invalid option, \'scence\' should be in [indoor, outdoor, both]."

        self.transform = kwargs.get('transform', None)

        self.path_list = self.__gen_data_list()

    def __gen_data_list(self):
        path_list = []

        if self.scence in ['indoor', 'outdoor']:
            rootfolders = [os.path.join(self.rootpath, self.scence)]
        else:
            rootfolders = [
                os.path.join(self.rootpath, 'indoor'),
                os.path.join(self.rootpath, 'outdoor')
            ]

        for rootfolder in rootfolders:
            rootfolder_left = os.path.join(rootfolder, 'left')
            rootfolder_right = os.path.join(rootfolder, 'right')
            l0_folders = sorted(os.listdir(rootfolder_left))
            folder_numbers = len(l0_folders)
            if self.split == 'train':
                l0_folders = l0_folders[:int(0.8*folder_numbers)]
            else:
                l0_folders = l0_folders[int(0.8*folder_numbers):]
            for l0_folder in l0_folders:
                l1_folders = os.listdir(os.path.join(rootfolder_left, l0_folder))
                for l1_folder in l1_folders:
                    sample = {}
                    sample['left'] = os.path.join(rootfolder_left, l0_folder, l1_folder)
                    sample['right'] = os.path.join(rootfolder_right, l0_folder, l1_folder)
                    path_list.append(sample)

        return path_list

    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, index):
        sample = self.path_list[index]
        left_path = sample['left']
        right_path = sample['right']

        left_dict = data_parameter_dict(left_path, 'stero_depth_estimation')
        right_dict = data_parameter_dict(right_path, 'stero_depth_estimation')

        label_path = left_dict['labeled_data_dir']
        label = np.load(label_path).astype(np.float32)
        if len(label.shape) == 2:  # [H x W] grayscale image -> [H x W x 1]
            label = np.expand_dims(label, -1)
        label = np.moveaxis(label, -1, 0)  # H x W x 1 -> 1 x H x W

        left_spike_obj = SpikeStream(filepath=left_dict['filepath'], spike_h=250, spike_w=400, print_dat_detail=False)
        right_spike_obj = SpikeStream(filepath=right_dict['filepath'], spike_h=250, spike_w=400, print_dat_detail=False)

        left_spike = left_spike_obj.get_spike_matrix(flipud=True).astype(np.float32)
        right_spike = right_spike_obj.get_spike_matrix(flipud=True).astype(np.float32)
        
        return {'left': left_spike, 'right': right_spike, 'depth': label}


class DatasetSpikeStero_Mono(DatasetSpikeStero):
    def __getitem__(self, index):
        sample = self.path_list[index]
        left_path = sample['left']

        left_dict = data_parameter_dict(left_path, 'stero_depth_estimation')
        
        label_path = left_dict['labeled_data_dir']
        label = np.load(label_path).astype(np.float32)
        if len(label.shape) == 2:  # [H x W] grayscale image -> [H x W x 1]
            label = np.expand_dims(label, -1)
        label = np.moveaxis(label, -1, 0)  # H x W x 1 -> 1 x H x W

        left_spike_obj = SpikeStream(filepath=left_dict['filepath'], spike_h=250, spike_w=400, print_dat_detail=False)
        
        left_spike = left_spike_obj.get_spike_matrix(flipud=True).astype(np.float32)
        
        return {'spike': left_spike, 'depth': label}