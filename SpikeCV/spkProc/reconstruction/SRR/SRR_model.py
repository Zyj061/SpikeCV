import torch.nn as nn
from base_networks import *
from dbpns import Net as DBPNS


class SRR_Net(nn.Module):
    def __init__(self, num_channels, base_filter, feat, num_stages, n_resblock, nFrames, scale_factor):
        super(SRR_Net, self).__init__()
        # base_filter=256
        # feat=64
        self.nFrames = nFrames

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2

        # Initial Feature Extraction

        # ～～～
        self.t0 = ConvBlock(240, 128, 1, 1, 0, activation='prelu', norm=None)
        self.t1 = ConvBlock(128, base_filter, 1, 1, 0, activation='prelu', norm=None)
        # ～～～

        # self.t = ConvBlock(240, base_filter, 3, 1, 1, activation='prelu', norm=None)

        self.feat1 = ConvBlock(8, base_filter, 3, 1, 1, activation='prelu', norm=None)

        ###DBPNS
        self.DBPN = DBPNS(base_filter, feat, num_stages, scale_factor)

        # Res-Block1
        modules_body1 = [
            ResnetBlock(base_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body1.append(DeconvBlock(base_filter, feat, kernel, stride, padding, activation='prelu', norm=None))
        self.res_feat1 = nn.Sequential(*modules_body1)

        # Res-Block2
        modules_body2 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body2.append(ConvBlock(feat, feat, 3, 1, 1, activation='prelu', norm=None))
        self.res_feat2 = nn.Sequential(*modules_body2)

        # Res-Block3
        modules_body3 = [
            ResnetBlock(feat, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None) \
            for _ in range(n_resblock)]
        modules_body3.append(ConvBlock(feat, base_filter, kernel, stride, padding, activation='prelu', norm=None))
        self.res_feat3 = nn.Sequential(*modules_body3)

        # Reconstruction
        self.output = ConvBlock((nFrames - 1) * feat, num_channels, 3, 1, 1, activation=None, norm=None)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    # def forward(self, x, neigbor, flow): # ~~~
    def forward(self, x, neigbor, flow, spike):
        ### initial feature extraction
        # feat_input = self.t(spike)

        # ～～～
        feat_input = self.t0(spike)
        feat_input = self.t1(feat_input)
        # ~~~

        feat_frame = []
        for j in range(len(neigbor)):
            feat_frame.append(self.feat1(torch.cat((x, neigbor[j], flow[j]), 1)))  # 6 * (1, 8, 120, 180) -> conv(8, 256, k=3, s=1, p=1) -> 6 * (1, 256, 120, 180)
        #             feat_frame.append(self.feat1(torch.cat((spikes[j], spikes[-1], flow[j]), 1)))

        ####Projection
        Ht = []
        for j in range(len(neigbor)):
            h0 = self.DBPN(feat_input)  # SISR：DBPN上采样 (1, 256, 120, 180) -> (1, 64, 480, 720)
            h1 = self.res_feat1(feat_frame[j])  # MISR：Resnet上采样 (1, 256, 120, 180) -> (1, 64, 480, 720)

            e = h0 - h1
            e = self.res_feat2(e)
            h = h0 + e
            Ht.append(h)
            feat_input = self.res_feat3(h)

        ####Reconstruction
        out = torch.cat(Ht, 1)  # 6 * (1, 64, 480, 720) -> (1, 384, 480, 720)
        output = self.output(out)  # (1, 384, 480, 720) -> conv(384, 3, k=3, s=1) -> (1, 3, 480, 720)

        return output
