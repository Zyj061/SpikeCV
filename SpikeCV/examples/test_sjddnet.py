# -*- coding: utf-8 -*- 
# @Time : 2024/06/25
# @Author : Yanchen Dong
# @File : test_sjddnet.py
import torch
import torch.nn as nn
import numpy as np
import sys, time
sys.path.append("..")
from spkProc.reconstruction.SJDDNet.sjddnet_model import SJDDNet
import os
from spkData.load_dat import data_parameter_dict
from spkData.load_dat import SpikeStream
from utils import path
from PIL import Image

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.modules.container')


def generate_bayer_mask(h, w, color_mode):
    num = []
    flag = 0
    for c in color_mode:
        if c == 'r':
            num.append(0)
        elif c == 'g' and flag == 0:
            num.append(1)
            flag = 1
        elif c == 'g' and flag == 1:
            num.append(2)
        elif c == 'b':
            num.append(3)
    mask = np.zeros((4, h, w))
    rows_1 = slice(0, h, 2)
    rows_2 = slice(1, h, 2)
    cols_1 = slice(0, w, 2)
    cols_2 = slice(1, w, 2)
    mask[num[0], rows_1, cols_1] = 1
    mask[num[1], rows_1, cols_2] = 1
    mask[num[2], rows_2, cols_1] = 1
    mask[num[3], rows_2, cols_2] = 1
    return mask


if __name__ == '__main__':
    # 参数
    block_len = 39
    gamma = 2.2
    h = 1000
    w = 1000
    color_mode = 'bggr'

    # 加载模型
    model = SJDDNet(n=block_len)
    model = torch.nn.DataParallel(model)
    model.cuda()
    model_path = os.path.join("..", "spkProc", "reconstruction", "SJDDNet", "pretrained", "gp", "pretrained_sjddnet.pth")
    checkpoint = torch.load(model_path, map_location=None)
    model.load_state_dict(checkpoint['model_state'], strict=False)
    del checkpoint
    model.eval()

    # 指定数据序列及任务类型
    data_filename = "BSS/data/water.dat"
    label_type = 'bayer'

    # 加载数据集属性字典
    paraDict = data_parameter_dict(data_filename, label_type)

    #加载脉冲数据
    vidarSpikes = SpikeStream(**paraDict)

    spikes = vidarSpikes.get_block_spikes(begin_idx=0, block_len=block_len) # T H W numpy

    spikes = torch.from_numpy(spikes.astype(np.float32)).unsqueeze(0).cuda()
    mask = torch.from_numpy(generate_bayer_mask(h, w, color_mode).astype(np.float32)).unsqueeze(0).cuda()
    st = time.time()
    with torch.no_grad():
        res, _ = model(spikes, mask)
    ed = time.time()
    print('shape: ', res.shape, 'time: {:.6f}'.format(ed - st))
    res = ((res.clamp(0., 1.))**(1/gamma)).permute(0, 2, 3, 1).squeeze(0).cpu().numpy() *255.0 
    filename = path.split_path_into_pieces(data_filename)
    if not os.path.exists('results'):
        os.makedirs('results')

    res_path = os.path.join('results', filename[-1] + '_sjddnet_res.png')
    img = Image.fromarray(np.uint8(res))
    img.convert('RGB').save(res_path)

    print("done.")
