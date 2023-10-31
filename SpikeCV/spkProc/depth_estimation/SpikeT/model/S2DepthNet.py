import torch.nn as nn
import torch
import torch.nn.functional as F
from model.model import BaseERGB2Depth
from model.encoder_transformer import LongSpikeStreamEncoderConv
from model.submodules import ResidualBlock, ConvLayer, UpsampleConvLayer


def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    return x1 + x2


def identity(x1, x2=None):
    return x1


class S2DepthTransformerUNetConv(BaseERGB2Depth):
    def __init__(self, config):
        super(S2DepthTransformerUNetConv, self).__init__(config)
        assert self.base_num_channels % 48 == 0

        self.depths=[int(i) for i in config["swin_depths"]]
        self.num_encoders = len(self.depths)
        self.num_heads=[int(i) for i in config["swin_num_heads"]]
        self.patch_size=[int(i) for i in config["swin_patch_size"]]
        self.out_indices=[int(i) for i in config["swin_out_indices"]]
        self.ape=config["ape"]
        try:
            self.num_v = config["new_v"]
        except KeyError:
            self.num_v = 0

        self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders-1)
        self.activation = getattr(torch, 'sigmoid')
        # self.num_channel_spikes = config["num_channel_spikes"]
        self.num_output_channels = 1

        self.encoder = LongSpikeStreamEncoderConv(
            patch_size=self.patch_size,
            in_chans=self.num_bins_rgb,
            embed_dim=self.base_num_channels,
            depths=self.depths,
            num_heads=self.num_heads,
            out_indices=self.out_indices,
            new_version=self.num_v,
        )

        self.UpsampleLayer = UpsampleConvLayer

        if self.skip_type == 'sum':
            self.apply_skip_connection = skip_sum
        elif self.skip_type == 'concat':
            self.apply_skip_connection = skip_concat
        elif self.skip_type == 'no_skip' or self.skip_type is None:
            self.apply_skip_connection = identity
        else:
            raise KeyError('Could not identify skip_type, please add "skip_type":'
                           ' "sum", "concat" or "no_skip" to config["model"]')

        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()
    
    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))
    

    def build_decoders(self):
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i) for i in range(self.num_encoders)]))

        self.decoders = nn.ModuleList()
        for input_size in decoder_input_sizes:
            self.decoders.append(self.UpsampleLayer(input_size if self.skip_type == 'sum' else 2 * input_size,
                                                    input_size // 2,
                                                    kernel_size=5, padding=2, norm=self.norm))

    def build_prediction_layer(self):
        self.pred = ConvLayer(self.base_num_channels // 2 if self.skip_type == 'sum' else 2 * self.base_num_channels,
                              self.num_output_channels, 1, activation=None, norm=self.norm)

    def forward_decoder(self, super_states):
        # last superstate is taken as input for decoder.
        if not bool(self.baseline) and self.state_combination == "convlstm":
            x = super_states[-1][0]
        else:
            x = super_states[-1]
        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            if i == 0:
                x = decoder(x)
                # print(x.shape)
            else:
                if not bool(self.baseline) and self.state_combination == "convlstm":
                    x = decoder(self.apply_skip_connection(x, super_states[self.num_encoders - i - 1][0]))
                else:
                    x = decoder(self.apply_skip_connection(x, super_states[self.num_encoders - i - 1]))
                    # print(x.shape)
            # x = decoder(x)

        # tail
        # img = self.activation(self.pred(self.apply_skip_connection(x, head)))
        img = self.activation(self.pred(x))

        return img

    def forward(self, item, prev_super_states, prev_states_lstm):
        #def forward(self, spike_tensor, prev_states=None):
        """
        :param spike_tensor: N x C x H x W
        :return: a predicted image of size N x 1 x H x W, taking values in [0,1].
        """
        predictions_dict = {}
        print(item.keys())
        spike_tensor = item["image"].to(self.gpu)
        # if "image" in list(item.keys()):
        #     spike_tensor = item["image"].to(self.gpu)
        # else:
        #     spike_tensor = item["image"].to(self.gpu)
        encoded_xs = self.encoder(spike_tensor)
        # for x in encoded_xs:
            # print(x.shape)
        prediction = self.forward_decoder(encoded_xs)
        predictions_dict["image"] = prediction

        return predictions_dict, {'image': None}, prev_states_lstm

