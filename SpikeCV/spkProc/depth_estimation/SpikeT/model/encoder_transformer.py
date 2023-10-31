import torch.nn as nn
import torch
import torch.nn.functional as F

from model.swin_transformer_3d import SwinTransformer3D


class LongSpikeStreamEncoderConv(nn.Module):
    def __init__(
        self,
        # num_blocks,
        # block_channel,
        patch_size=(32,2,2), 
        in_chans=128, 
        embed_dim=96, 
        depths=[2,2,6],
        num_heads=[3,6,12],
        patch_norm=False,
        out_indices=(0,1,2),
        frozen_stages=-1,
        new_version=3,
        ):
        super(LongSpikeStreamEncoderConv, self).__init__()

        
        self.num_blocks = in_chans // patch_size[0]
        # self.out_num_depths = self.num_blocks - 1
        # self.block_channel = block_channel
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        self.num_encoders = len(self.depths)
        self.out_channels = [self.embed_dim*(2**i) for i in range(self.num_encoders)]

        self.swin3d = SwinTransformer3D(
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            depths=self.depths,
            num_heads=self.num_heads,
            out_indices=self.out_indices,
            frozen_stages=self.frozen_stages,
            new_version=new_version,
        )

        self.patches_T = self.num_blocks
        # self.patch_T = self.patches_T // self.num_blocks  # 1

        self.conv_layers = nn.ModuleList()
        for i in range(self.num_encoders):
            conv_layer_i = nn.ModuleList()
            for ti in range(self.num_blocks):
                conv_layer_i.append(nn.Conv2d(self.out_channels[i], self.out_channels[i] // self.num_blocks, 1))
            self.conv_layers.append(conv_layer_i)

    def forward(self, inputs):
        B, C, H, W = inputs.shape

        features = self.swin3d(inputs)

        outs = []
        for i in range(self.num_encoders):
            out_layer_i = []
            features_i = features[i].chunk(self.num_blocks, 2)
            B, C, T, H, W = features_i[0].shape
            # features_i = features_i.reshape(B, -1, H, W)
            for k in range(self.num_blocks):
                feature_k = features_i[k].reshape(B, -1, H, W) # B,C,H,W
                out_k = self.conv_layers[i][k](feature_k)
                out_layer_i.append(out_k)
            out_i = torch.cat(out_layer_i, dim=1)
            outs.append(out_i)

        return outs
