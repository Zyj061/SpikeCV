import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class CAPA(nn.Module):
    def __init__(self, chnn, sc=11):
        super().__init__()
        self.sc = sc
        self.unfold = nn.Unfold(kernel_size=3*self.sc, dilation=1, padding=self.sc, stride=self.sc)
        self.scale = chnn ** -0.5
        self.to_q = nn.Conv2d(chnn, chnn, 1, bias=False)
        self.to_k = nn.Conv2d(chnn, chnn, 1, bias=False)
        self.to_v = nn.Conv2d(chnn, chnn, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.mask_k = True

    def forward(self, x_key, x_ref):
        b, c, h_in, w_in = x_key.shape
        x_pad = self.sc - w_in % self.sc
        y_pad = self.sc - h_in % self.sc
        feat_key = F.pad(x_key, (0, x_pad, 0, y_pad)) 
        feat_ref = F.pad(x_ref, (0, x_pad, 0, y_pad)) 
        b, c, h, w = feat_key.shape
        h_sc = h // self.sc
        w_sc = w // self.sc 

        fm = torch.ones(1, 1, h_in, w_in).to(feat_key.device)
        fm = F.pad(fm, (0, x_pad, 0, y_pad))
        fm_k = self.unfold(fm).view(1, 1, -1, h_sc*w_sc)
        fm_q = fm.view(1, 1, h_sc, self.sc, w_sc, self.sc).permute(0, 1, 2, 4, 3, 5).contiguous().view(1, 1, h_sc*w_sc, self.sc**2)
        am = torch.einsum('b c k n, b c n s -> b k n s', fm_k, fm_q)
        am = (am - 1) * 99.
        am = am.repeat(b, 1, 1, 1)

        feat_q = self.to_q(feat_key)
        feat_k = self.to_k(feat_ref)
        feat_k = self.unfold(feat_k).view(b, c, -1, h_sc*w_sc)
        feat_k = self.scale * feat_k
        feat_q = feat_q.view(b, c, h_sc, self.sc, w_sc, self.sc).permute(0, 1, 2, 4, 3, 5).contiguous().view(b, c, h_sc*w_sc, self.sc**2)
        attn = torch.einsum('b c k n, b c n s -> b k n s', feat_k, feat_q)


        attn = attn + am
        self.attn = F.softmax(attn, dim=1)

        feat_v = self.to_v(feat_ref)
        feat_v = self.unfold(feat_v).view(b, c, -1, h_sc*w_sc)
        feat_r = torch.einsum('b k n s, b c k n -> b c n s', self.attn, feat_v)
        feat_r = feat_r.view(b, c, h_sc, w_sc, self.sc, self.sc).permute(0, 1, 2, 4, 3, 5).contiguous().view(b, c, h, w)
        feat_r = feat_r[:,:,:h_in,:w_in]
        feat_o = x_ref + feat_r * self.gamma
        return feat_o


class Multi_Granularity_Align_One_Level(nn.Module):
    def __init__(self, base_dim=64, offset_groups=4, act=nn.ReLU(), memory=True):
        super().__init__()
        self.offset_groups = offset_groups
        self.memory = memory

        if self.memory:
            first_output_dim = base_dim
        else:
            first_output_dim = 3*self.offset_groups*3*3

        self.offset_conv_1 = self._make_two_conv_layer(input_dim=base_dim*2, hidden_dim=base_dim, output_dim=first_output_dim, kernel_size=3, stride=1, padding=1, act=act)
        if self.memory:
            self.offset_conv2_1 = self._make_two_conv_layer(input_dim=base_dim + 3*self.offset_groups*3*3, hidden_dim=base_dim, output_dim=3*self.offset_groups*3*3, kernel_size=3, stride=1, padding=1, act=act)
            self.fuse_feat = self._make_two_conv_layer(input_dim=base_dim*2, hidden_dim=base_dim, output_dim=base_dim, kernel_size=3, stride=1, padding=1, act=act)

    def _make_two_conv_layer(self, input_dim, hidden_dim, output_dim, kernel_size, stride, padding, act):
        layer = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            act,
            nn.Conv2d(hidden_dim, output_dim, kernel_size=kernel_size, stride=1, padding=padding),
        )
        return layer

    def forward(self, feat_list, kpa, deform_conv, prev_offset_feat=None, prev_feat=None):
        xa, xb, xc, xd, xe = feat_list

        xa_kpa, xb_kpa, xd_kpa, xe_kpa = [kpa(x_key=xc, x_ref=xxx) for xxx in [xa, xb, xd, xe]]
        feat_for_conv_offset1 = [torch.cat([xxx, xc], dim=1) for xxx in [xa_kpa, xb_kpa, xd_kpa, xe_kpa]]

        offset_feat_list1 = [self.offset_conv_1(f) for f in feat_for_conv_offset1]
        if self.memory:
            prev_offset_upsample_list1 = [F.interpolate(offset_feat, scale_factor=2, mode='bilinear') for offset_feat in prev_offset_feat]
            offset_feat_list1 = [self.offset_conv2_1(torch.cat((f1, f2), dim=1)) for f1, f2 in zip(offset_feat_list1, prev_offset_upsample_list1)]
        
        o1o2m_abde_list1 = [f.chunk(3, dim=1) for f in offset_feat_list1]
        offset_abde_list1 = [torch.cat((o1o2m[0], o1o2m[1]), dim=1) for o1o2m in o1o2m_abde_list1]
        mask_abde_list1 = [torch.sigmoid(o1o2m[2]) for o1o2m in o1o2m_abde_list1]
        
        x_align_abde = [deform_conv(input=xx, offset=offset, mask=mask) for xx,offset,mask in zip([xa_kpa, xb_kpa, xd_kpa, xe_kpa], offset_abde_list1, mask_abde_list1)]

        if self.memory:
            prev_x_abde_align_upasmple_list = [F.interpolate(xxx, scale_factor=2, mode='bilinear') for xxx in prev_feat]
            x_align_abde = [self.fuse_feat(torch.cat((x_align, prev_x_align_upsample), dim=1)) for x_align, prev_x_align_upsample in zip(x_align_abde, prev_x_abde_align_upasmple_list)]

        xa_align, xb_align, xd_align, xe_align = x_align_abde
        x_align = (xa_align, xb_align, xc, xd_align, xe_align)
        return x_align, offset_feat_list1


class Multi_Granularity_Align(nn.Module):
    def __init__(self, base_dim=64, groups=4, act=nn.ReLU(), sc=11):
        super().__init__()
        self.offset_groups = groups
        self.sc = sc

        self.kpa = CAPA(chnn=base_dim, sc=self.sc)
        self.deform_conv = DeformConv2d(in_channels=base_dim, out_channels=base_dim, kernel_size=3, stride=1, padding=1, groups=1)

        ## Downsample
        self.conv_ds_L2 = self._make_two_conv_layer(input_dim=base_dim, hidden_dim=base_dim, output_dim=base_dim, kernel_size=3, stride=2, padding=1, act=act)
        self.conv_ds_L3 = self._make_two_conv_layer(input_dim=base_dim, hidden_dim=base_dim, output_dim=base_dim, kernel_size=3, stride=2, padding=1, act=act)

        self.align_L3 = Multi_Granularity_Align_One_Level(base_dim=base_dim, offset_groups=self.offset_groups, act=act, memory=False)
        self.align_L2 = Multi_Granularity_Align_One_Level(base_dim=base_dim, offset_groups=self.offset_groups, act=act, memory=True)
        self.align_L1 = Multi_Granularity_Align_One_Level(base_dim=base_dim, offset_groups=self.offset_groups, act=act, memory=True)

    def _make_two_conv_layer(self, input_dim, hidden_dim, output_dim, kernel_size, stride, padding, act):
        layer = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            act,
            nn.Conv2d(hidden_dim, output_dim, kernel_size=kernel_size, stride=1, padding=padding),
        )
        return layer

    def forward(self, feat_list):
        '''
        feat_list: xa, xb, xc, xd, xe
        '''

        xa, xb, xc, xd, xe = feat_list
        ## Downsample
        feat_batch_cat_L1 = torch.cat(feat_list, dim=0)
        feat_batch_cat_L2 = self.conv_ds_L2(feat_batch_cat_L1)
        feat_batch_cat_L3 = self.conv_ds_L3(feat_batch_cat_L2)

        L3_align_feat_list, L3_offset_feat_list = self.align_L3(feat_list=feat_batch_cat_L3.chunk(5, dim=0), kpa=self.kpa, deform_conv=self.deform_conv,
                                                                prev_offset_feat=None, prev_feat=None)

        L2_align_feat_list, L2_offset_feat_list = self.align_L2(feat_list=feat_batch_cat_L2.chunk(5, dim=0), kpa=self.kpa, deform_conv=self.deform_conv, 
                                                                prev_offset_feat=L3_offset_feat_list, prev_feat=L3_align_feat_list)

        L1_align_feat_list, L1_offset_feat_list = self.align_L1(feat_list=feat_batch_cat_L1.chunk(5, dim=0), kpa=self.kpa, deform_conv=self.deform_conv,
                                                                prev_offset_feat=L2_offset_feat_list, prev_feat=L2_align_feat_list)

        return L1_align_feat_list
    

