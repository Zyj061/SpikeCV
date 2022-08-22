# -*- coding: utf-8 -*- 
# @Time : 2022/7/21
# @Author : Rui Zhao
# @File : test_scflow.py

import sys
sys.path.append("..")

import argparse
import time
import torch
import torch.optim
import torchvision.transforms as transforms
import datetime
from tensorboardX import SummaryWriter
import cv2
import os
import os.path as osp
import numpy as np
import random
import glob
import warnings
warnings.filterwarnings('ignore')

from spkData.load_dat import data_parameter_dict
from spkData.load_optical_flow import Dataset_PHM
from spkProc.optical_flow.SCFlow.scflow import get_scflow
from spkProc.optical_flow.SCFlow.utils import InputPadder, flow_warp
from metrics.optical_flow import compute_aee
from visualization.optical_flow_visualization import flow_visualization

data_filename = "OpticalFlowPHM"
label_type = "optical_flow"
scene = "poker"
result_path = osp.join('results', 'scflow', scene)

dt = 10

# 每隔多少个样例输出一次光流颜色编码图
vis_interval = 5

paraDict = data_parameter_dict(data_filename, label_type)
paraDict['dt'] = dt
paraDict['scene'] = scene

phm_dataset = Dataset_PHM(**paraDict)
phm_loader = torch.utils.data.DataLoader(dataset=phm_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=1)


pretrained_path = osp.join("..", "spkProc", "optical_flow", "SCFlow", "pretrained", "pretrained_dt{:d}.pth.tar".format(paraDict.get('dt')))
network_data = torch.load(pretrained_path)
model = get_scflow(data=network_data).cuda()
model = torch.nn.DataParallel(model).cuda()

model.eval()
AEE_sum = 0.
eval_time_sum = 0.
iters = 0.

if not osp.exists(result_path):
    os.makedirs(result_path)

for i, data in enumerate(phm_loader, 0):
    seq1_raw, seq2_raw, flowgt_raw = data

    # compute output
    seq1_raw = seq1_raw.cuda().type(torch.cuda.FloatTensor)
    seq2_raw = seq2_raw.cuda().type(torch.cuda.FloatTensor)
    flowgt = flowgt_raw.cuda().type(torch.cuda.FloatTensor).permute([0, 3, 1, 2])

    padder = InputPadder(seq1_raw.shape)
    seq1, seq2 = padder.pad(seq1_raw, seq2_raw)

    st_time = time.time()
    if i == 0:
        B, C, H, W = seq1.shape
        flow_init = torch.zeros([B, 2, H, W])
    with torch.no_grad():
        flows, model_res_dict = model(seq1=seq1, seq2=seq2, flow=flow_init, dt=dt)
    eval_time = time.time() - st_time

    flow_init = flows[0].clone().detach()
    flow_init = flow_warp(flow_init, -flow_init)

    pred_flow = padder.unpad(flows[0]).detach().permute([0, 2, 3, 1]).squeeze().cpu().numpy()
    flowgt = flowgt.detach().permute([0, 2, 3, 1]).squeeze().cpu().numpy()

    if i % vis_interval == 0:
        pred_flow_vis = flow_visualization(pred_flow, mode='scflow', use_cv2=True)
        pred_flow_vis_path = osp.join(result_path, '{:03d}.png'.format(i))
        cv2.imwrite(pred_flow_vis_path, pred_flow_vis)

    AEE = compute_aee(flowgt, pred_flow)
    AEE_sum += AEE
    eval_time_sum += eval_time
    iters += 1
    
print('-------------------------------------------------------')
print('Scene: {:s}, Mean AEE: {:6.4f}, Mean Eval Time: {:6.4f}'.format(scene, AEE_sum / iters, eval_time_sum / iters))
print('-------------------------------------------------------')