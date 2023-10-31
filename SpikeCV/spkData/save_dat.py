# -*- coding: utf-8 -*- 
# @Time : 2022/7/22
# @Author : Rui Zhao
# @File : save_dat.py

import os
import os.path as osp
import numpy as np

# saving .dat file from spikes
def SpikeToRaw(save_path, SpikeSeq, filpud=True, delete_if_exists=True):
    """
        save spike sequence to .dat file
        save_path: full saving path (string)
        SpikeSeq: Numpy array (T x H x W)
        Rui Zhao
    """
    if delete_if_exists:
        if osp.exists(save_path):
            os.remove(save_path)

    sfn, h, w = SpikeSeq.shape
    remainder = int((h * w) % 8)
    # assert (h * w) % 8 == 0
    base = np.power(2, np.linspace(0, 7, 8))
    fid = open(save_path, 'ab')
    for img_id in range(sfn):
        if filpud:
            # 模拟相机的倒像
            spike = np.flipud(SpikeSeq[img_id, :, :])
        else:
            spike = SpikeSeq[img_id, :, :]
        # numpy按自动按行排，数据也是按行存的
        # spike = spike.flatten()
        spike = np.concatenate([spike.flatten(), np.array([0]*(8-remainder))])
        spike = spike.reshape([int(h*w/8), 8])
        data = spike * base
        data = np.sum(data, axis=1).astype(np.uint8)
        fid.write(data.tobytes())

    fid.close()

    return