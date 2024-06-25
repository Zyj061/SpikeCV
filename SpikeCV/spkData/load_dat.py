# -*- coding: utf-8 -*- 

from genericpath import isdir
from glob import glob
import os, sys
import warnings

import ctypes

sys.path.append('../device/spikevision/m1k40')
from sdk import spikelinkapi as link

import numpy as np
import torch
import torchvision.transforms as transforms
import yaml
import glob

from .sps_parser import SPSParser

sys.path.append("..")
from utils import path
import time
import threading

# key-value for generate data loader according to the type of label data
LABEL_DATA_TYPE = {
    'raw': 0,
    'reconstruction': 1,
    'optical_flow': 2,
    'mono_depth_estimation': 3.1,
    'stero_depth_estimation': 3.2,
    'detection': 4,
    'tracking': 5,
    'recognition': 6,
    'bayer': 7
}


# generate parameters dictionary according to labeled or not
def data_parameter_dict(data_filename, label_type):
    filename = path.split_path_into_pieces(data_filename)

    if os.path.isabs(data_filename):
        file_root = data_filename
        if os.path.isdir(file_root):
            search_root = file_root
        else:
            search_root = '\\'.join(filename[0:-1])
        config_filename = path.seek_file(search_root, 'config.yaml')
    else:
        file_root = os.path.join('..', 'spkData', 'datasets', *filename)
        config_filename = os.path.join('..', 'spkData', 'datasets', filename[0], 'config.yaml')

    try:
        with open(config_filename, 'r', encoding='utf-8') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    except TypeError as err:
        print("Cannot find config file" + str(err))
        raise err

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
    paraDict['filelist'] = None

    if is_labeled:
        paraDict['labeled_data_type'] = configs.get('labeled_data_type')
        paraDict['labeled_data_suffix'] = configs.get('labeled_data_suffix')
        paraDict['label_root_list'] = None

        if os.path.isdir(file_root):
            filelist = sorted(glob.glob(file_root + '/*.dat'), key=os.path.getmtime)
            filepath = filelist[0]

            labelname = path.replace_identifier(filename, configs.get('data_field_identifier', ''),
                                                configs.get('label_field_identifier', ''))
            label_root_list = os.path.join('..', 'spkData', 'datasets', *labelname)
            paraDict['labeled_data_dir'] = sorted(glob.glob(label_root_list + '/*.' + paraDict['labeled_data_suffix']),
                                                  key=os.path.getmtime)

            paraDict['filelist'] = filelist
            paraDict['label_root_list'] = label_root_list
        else:
            filepath = glob.glob(file_root)[0]
            rawname = filename[-1].replace('.dat', '')
            filename.pop(-1)
            filename.append(rawname)
            labelname = path.replace_identifier(filename, configs.get('data_field_identifier', ''),
                                                configs.get('label_field_identifier', ''))
            label_root = os.path.join('..', 'spkData', 'datasets', *labelname)
            paraDict['labeled_data_dir'] = glob.glob(label_root + '.' + paraDict['labeled_data_suffix'])[0]
    else:
        filepath = file_root

    paraDict['filepath'] = filepath

    return paraDict


# generate parameters for initial online device
def device_parameters(params, params_input, type, input_stream, decode_width, decode_height, cusum=50):
    picture = link.SVPicture()
    picture.width = decode_width
    picture.height = decode_height
    picture.format = 0x00010000
    picture.fps.num = 20000
    picture.fps.den = 1
    params.mode = 0x00002000
    params.format = 0x00010000
    params.picture = picture
    params.buff_size = 500
    params.cusum = cusum

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
    elif type == 1:
        ''' Spike camera PCIE source '''
        params.type = 0x00000020
        params_input.channels = 0
        params_input.channelMode = 0
        params_input.devName = bytes(input_stream, 'utf-8')
        params.opaque = ctypes.cast(ctypes.byref(params_input), ctypes.c_void_p)
    else:
        ''' Spike camera USB source '''
        params.type = 0x00000021
        params_input.openByParam = bytes('000000000001', 'utf-8')
        params_input.type = 2
        params_input.packetSize = 262144
        params_input.queueSize = 16
        params.opaque = ctypes.cast(ctypes.byref(params_input), ctypes.c_void_p)
        print('read spikes from usb camera')

    return params


# obtain spike matrix
class SpikeStream:

    def __init__(self, offline=True, camera_type=None, **kwargs):
        self.SpikeMatrix = None
        self.offline = offline
        if self.offline and camera_type is None:
            self.filename = kwargs.get('filepath')
            if os.path.splitext(self.filename)[-1][1:] != 'dat':
                self.filename = self.filename + '.dat'
            self.spike_width = kwargs.get('spike_w')
            self.spike_height = kwargs.get('spike_h')
            if 'print_dat_detail' not in kwargs:
                self.print_dat_detail = True
            else:
                self.print_dat_detail = kwargs.get('print_dat_detail')

            self.camera_type = 1

        elif self.offline and camera_type == 'PCIE':
            self.filename = kwargs.get('filepath')
            self.decode_width = 1024
            self.spike_width = 1000
            self.spike_height = 1000
            self.camera_type = 0

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

    # get the row number index of the spike files
    def get_row_index(self):
        delete_row_index = None
        if self.camera_type == 0:
            # sps100 camera
            delete_row_index = np.hstack(np.arange(500, 512),
                                         np.arange(1000, 1012))
        elif self.camera_type == 1:
            # sps10 camera
            delete_row_index = np.arange(400, 416)

        return delete_row_index

    # get spike matrix from device
    def get_device_matrix(self, _input, _framepool, block_len=500, cusum=50):

        self.count = 1
        self.brunning = True
        self.block_len = 1000
        self.block_len = block_len
        self.SpikeMatrix = np.zeros([self.block_len, self.height, self.spike_width], np.uint8)

        readthrd = threading.Thread(target=self.get_spikes)
        readthrd.start()

        self.input.init(ctypes.byref(self.dev_params))
        self.input.setcallback(self.input_callback)
        self.input.open()
        self.input.start()

        readthrd.join()
        self.brunning = False

        self.input.stop()
        self.input.close()

        return self.SpikeMatrix

    def get_file_matrix(self, flipud=True, begin_idx=0, block_len=None, with_head=False):

        if self.camera_type == 1:
            if block_len is None:
                return self.get_spike_matrix(flipud, with_head)
            else:
                return self.get_block_spikes(begin_idx=begin_idx, block_len=block_len, flipud=flipud, with_head=with_head)
        else:
            return self.get_sps100_matrix(begin_idx=begin_idx, block_len=block_len)

    def get_sps100_matrix(self, begin_idx, block_len):
        file_reader = open(self.filename, 'rb')
        pix_id = np.arange(0, block_len * self.spike_height * self.decode_width)
        pix_id = np.reshape(pix_id, (block_len, self.spike_height, self.decode_width))
        comparator = np.left_shift(1, np.mod(pix_id, 8))
        byte_id = pix_id // 8

        # 1. find the head of frame
        first_frame_found = False
        spike_parser = SPSParser("test")
        one_line_data = bytearray([])
        frame_data = bytearray([])
        while not first_frame_found:
            data = file_reader.read(16)
            if len(data) == 0:
                print('file end')
                break
            if not spike_parser.is_one_frame_start(data):
                one_line_data += data
                if len(one_line_data) > 64:
                    one_line_data = one_line_data[-64:]
                continue

            first_frame_found = True
            one_line_data += data
            one_line_data = one_line_data[-64:]
            frame_data += one_line_data

        # 2. read 1024 * 1000 * block_len bit matrix
        if begin_idx == 0:
            tmp_len = 64 + 128 * (self.spike_height - 1) + 128 * self.spike_height * (block_len - 1)
            data = file_reader.read(tmp_len)
            frame_data += data
        else:
            # file_index = file_reader.tell()
            file_reader.read(64 + 128 * (self.spike_height - 1))
            img_size = self.spike_height * self.decode_width
            id_start = begin_idx * img_size // 8
            file_reader.seek(id_start, 1)  # move to the assigned begin frame from the current index
            matrix_len = img_size * block_len // 8
            frame_data = file_reader.read(matrix_len)

        data = np.frombuffer(frame_data, 'b')
        data_frames = data[byte_id]
        results = np.bitwise_and(data_frames, comparator)
        tmp_matrix = (results == comparator)
        delete_indx = self.get_row_index()

        # 3. delete no frame data
        tmp_matrix = np.delete(tmp_matrix, delete_indx, 2)
        self.SpikeMatrix = tmp_matrix[:, :, :]
        file_reader.close()
        return self.SpikeMatrix

    # return all spikes from dat file
    def get_spike_matrix(self, flipud=True, with_head=False):

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

        pix_id = np.arange(0, img_num * self.spike_height * self.spike_width)
        pix_id = np.reshape(pix_id, (img_num, self.spike_height, self.spike_width))
        comparator = np.left_shift(1, np.mod(pix_id, 8))
        byte_id = pix_id // 8

        data = video_seq[byte_id]
        result = np.bitwise_and(data, comparator)
        tmp_matrix = (result == comparator)
        # if with head, delete them
        if with_head:
            delete_indx = self.get_row_index()
            tmp_matrix = np.delete(tmp_matrix, delete_indx, 2)

        if flipud:
            self.SpikeMatrix = tmp_matrix[:, ::-1, :]
        else:
            self.SpikeMatrix = tmp_matrix

        file_reader.close()

        # self.SpikeMatrix = SpikeMatrix
        return self.SpikeMatrix

    # return spikes with specified length and begin index
    def get_block_spikes(self, begin_idx, block_len=1, flipud=True, with_head=False):

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
            print(
                'loading total spikes from dat file -- spatial resolution: %d x %d, begin index: %d total timestamp: %d' %
                (self.spike_width, self.spike_height, begin_idx, block_len))

        SpikeMatrix = np.zeros([block_len, self.spike_height, self.spike_width], np.uint8)

        pix_id = np.arange(0, block_len * self.spike_height * self.spike_width)
        pix_id = np.reshape(pix_id, (block_len, self.spike_height, self.spike_width))
        comparator = np.left_shift(1, np.mod(pix_id, 8))
        byte_id = pix_id // 8
        id_start = begin_idx * img_size // 8
        id_end = id_start + block_len * img_size // 8
        data = video_seq[id_start:id_end]
        data_frame = data[byte_id]
        result = np.bitwise_and(data_frame, comparator)
        tmp_matrix = (result == comparator)
        # if with head, delete them
        if with_head:
            delete_indx = self.get_row_index()
            tmp_matrix = np.delete(tmp_matrix, delete_indx, 2)

        if flipud:
            self.SpikeMatrix = tmp_matrix[:, ::-1, :]
        else:
            self.SpikeMatrix = tmp_matrix

        file_reader.close()
        return self.SpikeMatrix

    def get_pcie_spikes(self, _input, _framepool):
        """ Reading thread for obtain spikes from spike camera PCIE source """
        pix_id = np.arange(0, self.cusum * self.spike_height * self.decode_width)
        pix_id = np.reshape(pix_id, (self.cusum, self.spike_height, self.decode_width))
        comparator = np.left_shift(1, np.mod(pix_id, 8))
        byte_id = pix_id // 8

        if _framepool.size() > 0:
            frame = _framepool.pop()
            frame2 = ctypes.cast(frame, ctypes.POINTER(link.SpikeLinkVideoFrame))
            # print("get frame:", frame2.contents.size, frame2.contents.width, frame2.contents.height,
            #       frame2.contents.pts)

            spkdata = frame2.contents.data[0]
            CharArr = ctypes.c_char * frame2.contents.size
            char_arr = CharArr(*spkdata[:frame2.contents.size])
            data = np.frombuffer(char_arr, 'b')
            # data = np.array(data).astype(np.byte)
            data_frames = data[byte_id]
            results = np.bitwise_and(data_frames, comparator)
            tmp_matrix = (results == comparator)
            delete_index = self.get_row_index()
            tmp_matrix = np.delete(tmp_matrix, delete_index, 2)
            self.SpikeMatrix = tmp_matrix[:, :, :]

            _input.releaseFrame(frame)
        else:
            time.sleep(0.01)

    def get_usb_spikes(self, _input, _framepool):
        """ Reading thread for obtain spikes from spike camera USB source """
        pix_id = np.arange(0, self.cusum * self.spike_height * self.spike_width)
        pix_id = np.reshape(pix_id, (self.cusum, self.spike_height, self.spike_width))
        comparator = np.left_shift(1, np.mod(pix_id, 8))
        byte_id = pix_id // 8
        spatial_resolution = self.spike_height * self.spike_width

        if _framepool.size() > 0:
            frame = _framepool.pop()
            frame2 = ctypes.cast(frame, ctypes.POINTER(link.SpikeLinkVideoFrame))
            # print("get frame:", frame2.contents.size, frame2.contents.width, frame2.contents.height,
            #       frame2.contents.pts)
            timestamp = frame2.contents.pts * self.cusum
            print('process spikes at timestamp: ', timestamp)

            # if 0 < frame2.contents.pts < self.block_len:
            decode_st = time.time()

            spkdata = frame2.contents.data[0]
            chararr = ctypes.c_char * frame2.contents.size
            char_arr = chararr(*spkdata[:frame2.contents.size])
            data = np.frombuffer(char_arr, 'b')
            # data = np.array(data, dtype=np.byte)

            data_frames = data[byte_id]
            results = np.bitwise_and(data_frames, comparator)
            tmp_matrix = (results == comparator)
            self.SpikeMatrix = tmp_matrix[:, ::-1, :]

            print('decode one block take %.3f seconds' % (time.time() - decode_st))

            _input.releaseFrame(frame)
        else:
            time.sleep(0.01)


class Dataset_RealSpike(torch.utils.data.Dataset):
    '''数据集组织格式为

        mydataset
        |____input
             |-- xx.dat
             |-- xx.dat
             |-- xx.dat
            ...

        加载一堆dat文件, 用于真实场景序列, 无GroundTruth。
    '''

    def __init__(self, root_dir, crop_size):
        super(Dataset_RealSpike, self).__init__()
        self.spikefolder = root_dir + "/input/"

        self.spike_list = os.listdir(self.spikefolder)

        self.crop_size = crop_size

        self.transform = transforms.Compose([
            transforms.RandomCrop(self.crop_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
        ])

    def __getitem__(self, index: int):
        item_name = self.spike_list[index][:-4]
        spike_path = os.path.join(self.spikefolder, item_name + '.dat')

        paraDict = data_parameter_dict(spike_path, 'raw')
        vidarSpikes = SpikeStream(**paraDict)
        spikes = vidarSpikes.get_block_spikes(begin_idx=0, block_len=41)  # T H W numpy
        spikes = torch.from_numpy(spikes.astype(np.float32))

        if self.crop_size != 0:
            spikes = self.transform(spikes)

        item = {}
        item['spikes'] = spikes

        return item

    def __len__(self) -> int:
        return len(self.spike_list)
