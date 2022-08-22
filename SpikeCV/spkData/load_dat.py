# -*- coding: utf-8 -*- 

from genericpath import isdir
from glob import glob
import os, sys
import warnings

import numpy as np
import yaml
import glob

sys.path.append("..")
from utils import path

# key-value for generate data loader according to the type of label data
LABEL_DATA_TYPE = {
    'raw': 0,
    'reconstruction': 1,
    'optical_flow': 2,
    'mono_depth_estimation': 3.1,
    'stero_depth_estimation': 3.2,
    'detection': 4,
    'tracking': 5,
    'recognition': 6
}


# generate parameters dictionary according to labeled or not
def data_parameter_dict(data_filename, label_type):
    filename = path.split_path_into_pieces(data_filename)
    
    file_root = os.path.join('..', 'spkData', 'datasets', *filename)
    config_filename = os.path.join('..', 'spkData', 'datasets', filename[0], 'config.yaml')
    with open(config_filename, 'r', encoding='utf-8') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    try:
        labeled_data_type = configs.get('labeled_data_type')
        if LABEL_DATA_TYPE[label_type] not in labeled_data_type:
            raise ValueError('there is no labeled data for this task in the %s dataset' % filename[0])

    except KeyError as exception:
        print('ERROR! Task name does not exist')
        print('Task name must be in %s' % LABEL_DATA_TYPE.keys())
        raise exception

    is_labeled = configs.get('is_labeled')

    paraDict = {'spike_h': configs.get('spike_h'), 'spike_w': configs.get('spike_w')}
    
    if is_labeled:
        paraDict['labeled_data_type'] = configs.get('labeled_data_type')
        paraDict['labeled_data_suffix'] = configs.get('labeled_data_suffix')

        if os.path.isdir(file_root):
            filepath = glob.glob(file_root+'/*.dat')[0]
            labelname = path.replace_identifier(filename, configs.get('data_field_identifier',''), configs.get('label_field_identifier',''))
            label_root = os.path.join('..', 'spkData', 'datasets', *labelname)
            paraDict['labeled_data_dir'] = glob.glob(label_root+'/*.'+paraDict['labeled_data_suffix'])[0]
        else:
            filepath = glob.glob(file_root)[0]
            rawname = filename[-1].replace('.dat', '')
            filename.pop(-1)
            filename.append(rawname)
            labelname = path.replace_identifier(filename, configs.get('data_field_identifier',''), configs.get('label_field_identifier',''))
            label_root = os.path.join('..', 'spkData', 'datasets', *labelname)
            paraDict['labeled_data_dir'] = glob.glob(label_root+'.'+paraDict['labeled_data_suffix'])[0]
    else:
        filepath = file_root

    paraDict['filepath'] = filepath

    return paraDict


# loading spikes from *.dat file
class VidarSpike:

    def __init__(self, **kwargs):

        self.filename = kwargs.get('filepath')
        if os.path.splitext(self.filename)[-1][1:] != 'dat':
            self.filename = self.filename + '.dat'
        self.spike_w = kwargs.get('spike_w')
        self.spike_h = kwargs.get('spike_h')
        if 'print_dat_detail' not in kwargs:
            self.print_dat_detail = True
        else:
            self.print_dat_detail = kwargs.get('print_dat_detail')

    # return all spikes from dat file
    def get_spike_matrix(self, flipud=True):

        file_reader = open(self.filename, 'rb')
        video_seq = file_reader.read()
        video_seq = np.frombuffer(video_seq, 'b')

        video_seq = np.array(video_seq).astype(np.byte)
        if self.print_dat_detail:
            print(video_seq)
        img_size = self.spike_h * self.spike_w
        img_num = len(video_seq) // (img_size // 8)

        if self.print_dat_detail:
            print('loading total spikes from dat file -- spatial resolution: %d x %d, total timestamp: %d' %
                (self.spike_w, self.spike_h, img_num))

        SpikeMatrix = np.zeros([img_num, self.spike_h, self.spike_w], np.byte)

        pix_id = np.arange(0, self.spike_h * self.spike_w)
        pix_id = np.reshape(pix_id, (self.spike_h, self.spike_w))
        comparator = np.left_shift(1, np.mod(pix_id, 8))
        byte_id = pix_id // 8

        for img_id in np.arange(img_num):
            id_start = img_id * img_size // 8
            id_end = id_start + img_size // 8
            cur_info = video_seq[id_start:id_end]
            data = cur_info[byte_id]
            result = np.bitwise_and(data, comparator)
            if flipud:
                SpikeMatrix[img_id, :, :] = np.flipud((result == comparator))
            else:
                SpikeMatrix[img_id, :, :] = (result == comparator)

        file_reader.close()

        return SpikeMatrix

    # return spikes with specified length and begin index
    def get_block_spikes(self, begin_idx, block_len=1, flipud=True):

        file_reader = open(self.filename, 'rb')
        video_seq = file_reader.read()
        video_seq = np.frombuffer(video_seq, 'b')

        video_seq = np.array(video_seq).astype(np.uint8)
        img_size = self.spike_h * self.spike_w
        img_num = len(video_seq) // (img_size // 8)

        end_idx = begin_idx + block_len
        if end_idx > img_num:
            warnings.warn("block_len exceeding upper limit! Zeros will be padded in the end. ", ResourceWarning)
            end_idx = img_num

        if self.print_dat_detail:
            print('loading total spikes from dat file -- spatial resolution: %d x %d, begin index: %d total timestamp: %d' %
                (self.spike_w, self.spike_h, begin_idx, block_len))

        SpikeMatrix = np.zeros([block_len, self.spike_h, self.spike_w], np.uint8)

        pix_id = np.arange(0, self.spike_h * self.spike_w)
        pix_id = np.reshape(pix_id, (self.spike_h, self.spike_w))
        comparator = np.left_shift(1, np.mod(pix_id, 8))
        byte_id = pix_id // 8

        for img_id in np.arange(begin_idx, end_idx):
            id_start = img_id * img_size // 8
            id_end = id_start + img_size // 8
            cur_info = video_seq[id_start:id_end]
            data = cur_info[byte_id]
            result = np.bitwise_and(data, comparator)

            if flipud:
                SpikeMatrix[img_id - begin_idx, :, :] = np.flipud((result == comparator))
            else:
                SpikeMatrix[img_id - begin_idx, :, :] = (result == comparator)

        return SpikeMatrix
