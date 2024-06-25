import torch
import torch.nn as nn

from . import basicblocks as B


class ColorPrior(nn.Module):
    def __init__(self, in_channels=1,  out_channels=3, nc=[16, 32, 64, 128],
                nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(ColorPrior, self).__init__()
        self.head_r = B.conv(in_channels+1, nc[0], mode='C'+act_mode[-1])
        self.head_g = B.conv(in_channels+1, nc[0], mode='C'+act_mode[-1])
        self.head_b = B.conv(in_channels+1, nc[0], mode='C'+act_mode[-1])

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.down_r_1 = B.sequential(*[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[0], nc[1], mode='2'+act_mode))
        self.down_r_2 = B.sequential(*[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[1], nc[2], mode='2'+act_mode))
        self.down_r_3 = B.sequential(*[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[2], nc[3], mode='2'+act_mode))

        self.down_g_1 = B.sequential(*[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[0], nc[1], mode='2'+act_mode))
        self.down_g_2 = B.sequential(*[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[1], nc[2], mode='2'+act_mode))
        self.down_g_3 = B.sequential(*[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[2], nc[3], mode='2'+act_mode))

        self.down_b_1 = B.sequential(*[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[0], nc[1], mode='2'+act_mode))
        self.down_b_2 = B.sequential(*[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[1], nc[2], mode='2'+act_mode))
        self.down_b_3 = B.sequential(*[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)], downsample_block(nc[2], nc[3], mode='2'+act_mode))

        self.body  = B.sequential(*[B.conv(nc[3], nc[3], mode='C'+act_mode) for _ in range(nb+1)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.up_3= B.sequential(upsample_block(nc[3], nc[2], mode='2'+act_mode), *[B.conv(nc[2], nc[2], mode='C'+act_mode) for _ in range(nb)])
        self.up_2 = B.sequential(upsample_block(nc[2], nc[1], mode='2'+act_mode), *[B.conv(nc[1], nc[1], mode='C'+act_mode) for _ in range(nb)])
        self.up_1 = B.sequential(upsample_block(nc[1], nc[0], mode='2'+act_mode), *[B.conv(nc[0], nc[0], mode='C'+act_mode) for _ in range(nb)])
        self.tail = B.conv(nc[0], out_channels, bias=True, mode='C')

    def forward(self, rgb, noise_map):
        x_r = torch.cat((rgb[:,0:1,:,:], noise_map), axis=1)
        x_g = torch.cat((rgb[:,1:2,:,:], noise_map), axis=1)
        x_b = torch.cat((rgb[:,2:3,:,:], noise_map), axis=1)

        x_r_1 = self.head_r(x_r)
        x_r_2 = self.down_r_1(x_r_1)
        x_r_3 = self.down_r_2(x_r_2)
        x_r_4 = self.down_r_3(x_r_3)

        x_g_1 = self.head_g(x_g)
        x_g_2 = self.down_g_1(x_g_1)
        x_g_3 = self.down_g_2(x_g_2)
        x_g_4 = self.down_g_3(x_g_3)

        x_b_1 = self.head_b(x_b)
        x_b_2 = self.down_b_1(x_b_1)
        x_b_3 = self.down_b_2(x_b_2)
        x_b_4 = self.down_b_3(x_b_3)

        x_5 = self.body(x_r_4 + x_g_4 + x_b_4)

        x = self.up_3(x_5 + x_r_4 + x_g_4 +  x_b_4)
        x = self.up_2(x + x_r_3 + x_g_3 +  x_b_3)
        x = self.up_1(x + x_r_2 + x_g_2 +  x_b_2)
        x = self.tail(x + x_r_1 + x_g_1 +  x_b_1)
        return x + rgb
