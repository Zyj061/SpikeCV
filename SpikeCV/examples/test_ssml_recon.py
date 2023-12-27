# -*- coding: utf-8 -*- 
# @Time : 2022/12/06
# @Author : Shiyan Chen
# @File : test_ssml_recon.py
import torch
import torch.nn as nn
import numpy as np
import sys, time
sys.path.append("..")
from spkProc.reconstruction.SSML_Recon.ssml_model import SSML_ReconNet
import cv2, os
from spkData.load_dat import data_parameter_dict
from spkData.load_dat import SpikeStream
from utils import path

if __name__ == '__main__':
    model = SSML_ReconNet()
    model_path = os.path.join("..", "spkProc", "reconstruction", "SSML_Recon", "pretrained", "pretrained_ssml_recon.pt")
    model.load_state_dict(torch.load(model_path)) 
    model = model.cuda()

    # 指定数据序列及任务类型
    data_filename = "recVidarReal2019/classA/car-100kmh"
    label_type = 'raw'

    # 加载数据集属性字典
    paraDict = data_parameter_dict(data_filename, label_type)

    #加载脉冲数据
    vidarSpikes = SpikeStream(**paraDict)
    block_len = 41
    spikes = vidarSpikes.get_block_spikes(begin_idx=0, block_len=block_len) # T H W numpy

    spikes = torch.from_numpy(spikes.astype(np.float32)).unsqueeze(0).cuda()

    st = time.time()
    res = model(spikes, train=False)
    ed = time.time()
    print('shape: ', res.shape, 'time: {:.6f}'.format(ed - st))
    res = res[0].detach().cpu().permute(1,2,0).numpy()*255

    filename = path.split_path_into_pieces(data_filename)
    if not os.path.exists('results'):
        os.makedirs('results')

    res_path = os.path.join('results', filename[-1] + '_ssml_recon_res.png')
    cv2.imwrite(res_path,res)

    print("done.")
