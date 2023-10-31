from base import BaseModel
import torch.nn as nn
import torch
from os.path import join
from model.submodules import \
    ConvLSTM, ResidualBlock, ConvLayer, \
    UpsampleConvLayer, TransposedConvLayer


def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    return x1 + x2


def identity(x1, x2=None):
    return x1


class BaseERGB2Depth(BaseModel):
    def __init__(self, config):
        super().__init__(config)

        assert ('num_bins_rgb' in config)
        self.num_bins_rgb = int(config['num_bins_rgb'])  
        assert ('num_bins_events' in config)
        self.num_bins_events = int(config['num_bins_events'])  

        try:
            self.skip_type = str(config['skip_type'])   # 'sum'
        except KeyError:
            self.skip_type = 'sum'

        try:
            self.state_combination = str(config['state_combination'])   # none
        except KeyError:
            self.state_combination = 'sum'

        try:
            self.num_encoders = int(config['num_encoders']) # 3
        except KeyError:
            self.num_encoders = 4

        try:
            self.base_num_channels = int(config['base_num_channels'])   # 32
        except KeyError:
            self.base_num_channels = 32

        try:
            self.num_residual_blocks = int(config['num_residual_blocks'])   # 2
        except KeyError:
            self.num_residual_blocks = 2

        try:
            self.recurrent_block_type = str(config['recurrent_block_type']) # none
        except KeyError:
            self.recurrent_block_type = 'convlstm'

        try:
            self.norm = str(config['norm']) # 'none'
        except KeyError:
            self.norm = None

        try:
            self.use_upsample_conv = bool(config['use_upsample_conv'])  # True
        except KeyError:
            self.use_upsample_conv = True

        try:
            self.every_x_rgb_frame = config['every_x_rgb_frame']    # 1
        except KeyError:
            self.every_x_rgb_frame = 1

        try:
            self.baseline = config['baseline']  # e
        except KeyError:
            self.baseline = False

        try:
            self.loss_composition = config['loss_composition']  # 'image'
        except KeyError:
            self.loss_composition = False

        self.kernel_size = int(config.get('kernel_size', 5))    # 5
        self.gpu = torch.device('cuda:' + str(config['gpu']))


