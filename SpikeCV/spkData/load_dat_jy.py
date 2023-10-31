# -*- coding: utf-8 -*- 

from genericpath import isdir
from glob import glob
import os, sys
import warnings

import ctypes

sys.path.append('../device/spikevision/m1k40')
from sdk import spikelinkapi as link

import numpy as np
import yaml
import glob
import torch

sys.path.append("..")
from utils import path
import time
import threading
import torch

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


# generate parameters for initial online device
def device_parameters(params, params_input, type, input_stream, decode_width, decode_height):

    picture = link.SVPicture()
    picture.width = decode_width
    picture.height = decode_height
    picture.format = 0x00010000
    picture.fps.num = 20000
    picture.fps.den = 1
    params.mode = 0x00002000
    params.format = 0x00010000
    params.picture = picture
    params.buff_size = 30
    params.cusum = 400

    if type == 0:
        ''' dummy camera source '''
        params.type = 0x00000024
        params_input.fps.num = 20000
        params_input.fps.den = 1
        params_input.duration = 0
        params_input.skip = 0
        params_input.start = 0
        params_input.end = 0
        params_input.repeat = 0
        params_input.fileName = bytes(input_stream, 'utf-8')
        params.opaque = ctypes.cast(ctypes.byref(params_input), ctypes.c_void_p)
    else:
        ''' Vidar camera source '''
        params.type = 0x00000020
        params_input.channels = 0
        params_input.channelMode = 0
        params_input.devName = bytes(input_stream, 'utf-8')
        params.opaque = ctypes.cast(ctypes.byref(params_input), ctypes.c_void_p)

    return params


# obtain spike matrix
class SpikeStream:

    def __init__(self, offline=True, camera_type=None, **kwargs):
        self.SpikeMatrix = None
        self.offline = offline
        if self.offline:
            self.filename = kwargs.get('filepath')
            if os.path.splitext(self.filename)[-1][1:] != 'dat':
                self.filename = self.filename + '.dat'
            self.spike_width = kwargs.get('spike_w')
            self.spike_height = kwargs.get('spike_h')
            if 'print_dat_detail' not in kwargs:
                self.print_dat_detail = True
            else:
                self.print_dat_detail = kwargs.get('print_dat_detail')
        elif camera_type == 'PCIE':
            ''' Spike camera PCIE source '''
            # self.dev_params = kwargs.get('params')
            self.decode_width = kwargs.get('decode_width')
            self.spike_width = kwargs.get('spike_width')
            self.spike_height = kwargs.get('height')
            self.camera_type = 0
        else:
            ''' Spike camera USB source '''
            self.spike_width = kwargs.get('spike_width')
            self.spike_height = kwargs.get('spike_height')
            self.camera_type = 1

    # get spike matrix from device
    def get_device_matrix(self, _input, _framepool, block_len=500):

        # self.input = _input
        self.count = 1
        self.brunning = True
        # self.block_len = 1000
        self.block_len = block_len
        self.SpikeMatrix = np.zeros([self.block_len, self.spike_height, self.spike_width], np.uint8)

        if self.camera_type == 0:
            readthrd = threading.Thread(target=self.get_pcie_spikes, kwargs={"_input": _input, "_framepool": _framepool})
        else:
            readthrd = threading.Thread(target=self.get_usb_spikes, kwargs={"_input": _input, "_framepool": _framepool})
        readthrd.start()

        _input.start()
        
        readthrd.join()
        self.brunning = False

        _input.stop()
        # _input.close()

        return self.SpikeMatrix

    # return all spikes from dat file
    def get_spike_matrix(self, flipud=True):

        file_reader = open(self.filename, 'rb')
        video_seq = file_reader.read()
        video_seq = np.frombuffer(video_seq, 'b')

        video_seq = np.array(video_seq).astype(np.byte)
        if self.print_dat_detail:
            print(video_seq)
        img_size = self.spike_height * self.spike_width
        img_num = len(video_seq) // (img_size // 8)

        if self.print_dat_detail:
            print('loading total spikes from dat file -- spatial resolution: %d x %d, total timestamp: %d' %
                (self.spike_width, self.spike_height, img_num))

        SpikeMatrix = np.zeros([img_num, self.spike_height, self.spike_width], np.byte)

        pix_id = np.arange(0, self.spike_height * self.spike_width)
        pix_id = np.reshape(pix_id, (self.spike_height, self.spike_width))
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

        self.SpikeMatrix = SpikeMatrix
        return SpikeMatrix

    # return spikes with specified length and begin index
    def get_block_spikes(self, begin_idx, block_len=1, flipud=True):

        file_reader = open(self.filename, 'rb')
        video_seq = file_reader.read()
        video_seq = np.frombuffer(video_seq, 'b')

        video_seq = np.array(video_seq).astype(np.uint8)
        img_size = self.spike_height * self.spike_width
        img_num = len(video_seq) // (img_size // 8)

        end_idx = begin_idx + block_len
        if end_idx > img_num:
            warnings.warn("block_len exceeding upper limit! Zeros will be padded in the end. ", ResourceWarning)
            end_idx = img_num

        if self.print_dat_detail:
            print('loading total spikes from dat file -- spatial resolution: %d x %d, begin index: %d total timestamp: %d' %
                (self.spike_width, self.spike_height, begin_idx, block_len))

        SpikeMatrix = np.zeros([block_len, self.spike_height, self.spike_width], np.uint8)

        pix_id = np.arange(0, self.spike_height * self.spike_width)
        pix_id = np.reshape(pix_id, (self.spike_height, self.spike_width))
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
        
        self.SpikeMatrix = SpikeMatrix
        return SpikeMatrix

    def get_pcie_spikes(self, _input, _framepool):
        pix_id = np.arange(0, self.spike_height * self.decode_width)
        pix_id = np.reshape(pix_id, (self.spike_height, self.decode_width))
        comparator = np.left_shift(1, np.mod(pix_id, 8))
        byte_id = pix_id // 8

        while self.brunning:
            if self.count > self.block_len:
                self.brunning = false
                break

            if _framepool.size() > 0:
                frame = _framepool.pop()
                frame2 = ctypes.cast(frame, ctypes.pointer(link.spikelinkvideoframe))
                # print("get frame:", frame2.contents.size, frame2.contents.width, frame2.contents.height,
                #       frame2.contents.pts)
                if 0 < frame2.contents.pts < self.block_len:
                    spkdata = frame2.contents.data[0]
                    chararr = ctypes.c_char * frame2.contents.size
                    char_arr = chararr(*spkdata[:frame2.contents.size])
                    data = np.frombuffer(char_arr, 'b')
                    data = np.array(data).astype(np.byte)
                    data_frame = data[byte_id]
                    result = np.bitwise_and(data_frame, comparator)
                    tmp_matrix = (result == comparator)
                    delete_ind = np.hstack(np.arange(500, 512),
                                           np.arange(1000, 1012))
                    tmp_matrix = np.delete(tmp_matrix, delete_ind, 1)

                    self.spikematrix[frame2.contents.pts, :, :] = tmp_matrix

                _input.releaseframe(frame)
            else:
                time.sleep(0.01)

    def get_usb_spikes(self, _input, _framepool):
        pix_id = np.arange(0, self.spike_height * self.spike_width)
        pix_id = np.reshape(pix_id, (self.spike_height, self.spike_width))
        comparator = np.left_shift(1, np.mod(pix_id, 8))
        byte_id = pix_id // 8

        while self.brunning:
            if self.count > self.block_len:
                self.brunning = False
                break

            if _framepool.size() > 0:
                frame = _framepool.pop()
                frame2 = ctypes.cast(frame, ctypes.pointer(link.spikelinkvideoframe))
                # print("get frame:", frame2.contents.size, frame2.contents.width, frame2.contents.height,
                #       frame2.contents.pts)
                if 0 < frame2.contents.pts < self.block_len:
                    spkdata = frame2.contents.data[0]
                    chararr = ctypes.c_char * frame2.contents.size
                    char_arr = chararr(*spkdata[:frame2.contents.size])
                    data = np.frombuffer(char_arr, 'b')
                    data = np.array(data).astype(np.byte)
                    data_frame = data[byte_id]
                    result = np.bitwise_and(data_frame, comparator)
                    tmp_matrix = (result == comparator)

                    self.spikematrix[frame2.contents.pts, :, :] = tmp_matrix

                _input.releaseframe(frame)
            else:
                time.sleep(0.01)

    def to_tensor(self):
        if not isinstance(self.SpikeMatrix, torch.Tensor):
            self.SpikeMatrix = torch.tensor(self.SpikeMatrix)
        return

    def to_numpy(self):
        if not isinstance(self.SpikeMatrix, np.ndarray):
            self.SpikeMatrix = self.SpikeMatrix.numpy()
        return