# Learning Super-Resolution Reconstruction for High Temporal Resolution Spike Stream (TCSVT 21, Xijie Xiang)
from __future__ import print_function

import os
import cv2
import time
import numpy as np
import math
from PIL import Image, ImageOps

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from spkProc.reconstruction.SRR.SRR_model import SRR_Net as SRR

from spkData.load_dat import data_parameter_dict
from spkData.load_dat import SpikeStream

import pyflow

def TFI(seq: np.ndarray, mid: int) -> np.ndarray:
    '''
    Snapshot an image using interval method.
    '''
    length, height, width = seq.shape
    result = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            up, down = mid, mid-1
            for up in range(mid, length):
                if seq[up, i, j] == 1:
                    break
            for down in range(mid-1, -1, -1):
                if seq[down, i, j] == 1:
                    break
            result[i, j] = math.pow(255 / (up - down),1/1)
    result = (255*result/(np.max(result))).astype(np.uint8)
    # result = np.repeat(np.expand_dims(result, axis=-1), 3, axis=-1)
    result = Image.fromarray(result).convert('RGB')
    result.save('tfi.png')
    return result

def spike2img(spike, spike_num):

    i = int(spike_num / 6)
    recon_idx = [i, 2*i, 4*i, 5*i]
    middle = TFI(spike, 3 * i)
    recon_frames = [TFI(spike, i) for i in recon_idx]

    return middle, recon_frames

def get_flow(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.

    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
                                         nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    return flow

model_path = 'SpikeCV/spkProc/reconstruction/SRR/pretrained/dark.pth'
output_path = 'results'

if not os.path.exists(output_path):
    os.makedirs(output_path)

gpus_list = [0]

cuda = True
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(123)
if cuda:
    torch.cuda.manual_seed(123)

print('===> Loading datasets')
# 指定数据序列及任务类型
data_filename = "recVidarReal2019/classA/car-100kmh"
label_type = 'raw'
# 加载数据集属性字典
paraDict = data_parameter_dict(data_filename, label_type)
#加载脉冲数据
vidarSpikes = SpikeStream(**paraDict)
block_len = 240
spikes = vidarSpikes.get_block_spikes(begin_idx=0, block_len=block_len) # T H W numpy
#预处理数据
middle, neigbor = spike2img(spikes, spikes.shape[0])
flow = [get_flow(input, j) for j in neigbor]  # 分别计算当前帧和-3、-2、-1、1、2、3帧的光流：(120, 180, 3) (120, 180, 3) -> (120, 180, 2)
input = ToTensor()(input)  # PIL.Image (180, 120) -> Tensor (3, 120, 180)
neigbor = [ToTensor()(j) for j in neigbor]  # 6 * PIL.Image (180, 120) -> 6 * Tensor (3, 120, 180)
flow = [torch.from_numpy(j.transpose(2, 0, 1)) for j in flow]  # 6 * ndarray (120, 180, 2) -> 6 * ndarray (2, 120, 180) -> 6 * Tensor (2, 120, 180)
spike = torch.from_numpy(spikes.copy()).float()
test_set = [[input, neigbor, flow, spike]]

print('===> Building model')
model = SRR(num_channels=3, base_filter=256, feat=64, num_stages=3, n_resblock=5, nFrames=5,
            scale_factor=4)

if cuda:
    model = torch.nn.DataParallel(model, device_ids=gpus_list)

model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])

def eval():
    model.eval()
    count = 1
    for batch in test_set:
        input, neigbor, flow, spike = batch[0], batch[1], batch[2], batch[3]

        with torch.no_grad():
            input = Variable(input).cuda(gpus_list[0])  # (1, 3, 120, 180)
            neigbor = [Variable(j).cuda(gpus_list[0]) for j in neigbor]  # 6 * (1, 3, 120, 180)
            flow = [Variable(j).cuda(gpus_list[0]).float() for j in flow]  # 6 * (1, 2, 120, 180)
            spike = Variable(spike).cuda(gpus_list[0])

            t0 = time.time()
            prediction = model(input, neigbor, flow, spike)
            t1 = time.time()
            print("===> Processing: %s || Timer: %.4f sec." % (str(count), (t1 - t0)))

            srr = prediction.cpu().squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)
            cv2.imwrite(output_path + '/output.png', cv2.cvtColor(srr * 255, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])
            count += 1

if __name__ == "__main__":
    eval()
