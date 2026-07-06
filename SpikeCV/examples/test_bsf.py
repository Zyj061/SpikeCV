# -*- coding: utf-8 -*-
# @Time : 2026/07/06
# @File : test_bsf.py

import os
import sys

import torch
from pprint import pprint

sys.path.append("..")

from spkData.load_dat import SpikeStream, data_parameter_dict
from spkProc.reconstruction.BSF_Recon.bsf_recon import BSFRecon, default_weight_path
from visualization.get_video import obtain_reconstruction_video

if __name__ == '__main__':
    data_filename = "recVidarReal2019/classA/car-100kmh"
    label_type = "raw"
    begin_idx = 500
    block_len = 300
    step = 3

    paraDict = data_parameter_dict(data_filename, label_type)
    pprint(paraDict)

    vidarSpikes = SpikeStream(**paraDict)
    spikes = vidarSpikes.get_block_spikes(begin_idx=begin_idx, block_len=block_len)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Using device: {device_name}")
    print(f"Expected weight path: {default_weight_path()}")

    reconstructor = BSFRecon(
        spike_h=paraDict["spike_h"],
        spike_w=paraDict["spike_w"],
        device=device,
        step=step,
    )

    print("Running BSF reconstruction...")
    rec_img = reconstructor.spikes2images(spikes)
    print(f"Reconstructed frames shape: {rec_img.shape}")

    if not os.path.exists("results"):
        os.makedirs("results")

    result_video = os.path.join("results", "car-100kmh_bsf.avi")
    obtain_reconstruction_video(rec_img, result_video, **paraDict)
    print(f"Reconstruction video saved to: {result_video}")
