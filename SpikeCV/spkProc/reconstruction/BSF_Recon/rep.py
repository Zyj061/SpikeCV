import torch
import torch.nn as nn

class MODF(nn.Module):
    def __init__(self, base_dim=64, act=nn.ReLU()):
        super().__init__()
        self.base_dim = base_dim

        self.conv1 = self._make_layer(input_dim=21, hidden_dim=self.base_dim, output_dim=self.base_dim, act=act)
        self.conv_for_others = nn.ModuleList([
            self._make_layer(input_dim=self.base_dim, hidden_dim=self.base_dim, output_dim=self.base_dim, act=act) for ii in range(3)
        ])
        self.conv_fuse = self._make_layer(input_dim=self.base_dim*3, hidden_dim=self.base_dim, output_dim=self.base_dim, act=act)

    def _make_layer(self, input_dim, hidden_dim, output_dim, act):
        layer = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            act,
            nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1),
        )
        return layer

    def forward(self, dsft_dict):
        d11 = 1.0 / dsft_dict['dsft11']
        d12 = 2.0 / dsft_dict['dsft12']
        d21 = 2.0 / dsft_dict['dsft21']
        d22 = 3.0 / dsft_dict['dsft22']

        d_list = [d11, d12, d21, d22]
        feat_batch_cat = self.conv1(torch.cat(d_list, dim=0))
        feat_list = feat_batch_cat.chunk(4, dim=0)

        feat_11 = feat_list[0]
        feat_others_list = feat_list[1:]
        feat_others_list_processed = []
        for ii in range(3):
            feat_others_list_processed.append(self.conv_for_others[ii](feat_others_list[ii]))


        other_feat = torch.cat(feat_others_list_processed, dim=1)
        other_feat_res = self.conv_fuse(other_feat)

        return feat_11 + other_feat_res

