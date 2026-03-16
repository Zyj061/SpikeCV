# -*- coding: utf-8 -*-
# @Time : 2024/12/05 20:17
# @Author : Yajing Zheng
# @Email: yj.zheng@pku.edu.cn
# @File : test_snntracker.py

'''
A program for applying snnTracker on tracking task with different parameters setting
Download the motVidarReal2025 spike dataset before running this program: https://github.com/Zyj061/snnTracker

example:
    python test_snntracker.py
    python test_snntracker.py --scene_idx 0 --attention_size 15 --data_path motVidarReal2020/ --label_type tracking --metrics
'''


import argparse
import cv2
import os
from utils import path
import sys
import torch
import numpy as np
from pprint import pprint
from spkProc.tracking.SNN_Tracker.snn_tracker import SNNTracker
from spkData.load_dat import data_parameter_dict, SpikeStream
from utils.utils import vis_trajectory, downscale_input
from visualization.get_video import obtain_mot_video
from metrics.tracking_mot_v2 import TrackingMetrics
import pathlib
from pathlib import Path

def main():
    try:
        print("REMOTE_HOST_RUN:", os.uname().nodename)
    except AttributeError:
        print("REMOTE_HOST_RUN: Windows (os.uname not available)")
    print("CWD:", os.getcwd())
    print("FILE:", pathlib.Path(__file__).resolve())
    # 以脚本所在目录为根，保证稳定
    SCRIPT_DIR = Path(__file__).resolve().parent
    RESULTS_DIR = SCRIPT_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description=__doc__, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene_idx", "-s", type=int, default=0,help="Index of the test scene")
    parser.add_argument("--attention_size", "-attn_size", type=int, default=15,help="Size of attention window")

    parser.add_argument("--data_path", "-d", type=str, default="motVidarReal2020/",help="Path to dataset root")
    parser.add_argument("--label_type", "-l", type=str, default="tracking",help="Label type")
    parser.add_argument("--metrics", "-m", action="store_true",help="Enable quantitative metrics (requires GT)")
    args = parser.parse_args()

    # TODO: 测试 no ground truth 的算法能否正确运行、可视化
    # change the path to where you put the datasets
    test_scene = ['spike59', 'rotTrans', 'cplCam', 'cpl1', 'badminton', 'ball', 'badminton-l1', 'badminton-l2', 'pingpong']
    # data_filename = 'motVidarReal2020/rotTrans'
    attention_size = args.attention_size
    data_name = test_scene[args.scene_idx]
    # 修复：直接指定数据文件路径，而不是目录路径
    data_filename = os.path.join(args.data_path, data_name)
    para_dict = data_parameter_dict(data_filename, args.label_type)
    pprint(para_dict)
    vidarSpikes = SpikeStream(**para_dict)

    block_len = 1000
    spikes = vidarSpikes.get_block_spikes(begin_idx=0, block_len=block_len)
    # spikes = vidarSpikes.get_spike_matrix()

    # downscale the input
    # Turn scaling off by default
    # FIXME: 缩放后无法输出预期结果，可能是attention_size也要scaledown，以及后面输出的result要upscale回去
    scale_w = 1 #2
    scale_h = 1 #2
    spikes = downscale_input(spikes, scale_w, scale_h)
    para_dict['spike_w'] = para_dict.get('spike_w') // scale_w
    para_dict['spike_h'] = para_dict.get('spike_h') // scale_h
    pprint(spikes.shape)

    # TODO: 尚未测试在CUDA上是否正常运行
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    calibration_time = 150
    filename = path.split_path_into_pieces(data_filename)
    result_filename = filename[-1] + '_snn.txt'
    tracking_file = RESULTS_DIR / result_filename

    spike_tracker = SNNTracker(para_dict.get('spike_h'), para_dict.get('spike_w'), device, attention_size=attention_size)
    # spike_tracker.object_cluster.K2 = 4
    total_spikes = spikes

    # using stp filter to filter out static spikes
    spike_tracker.calibrate_motion(spikes, calibration_time)
    # start tracking
    track_videoName = (RESULTS_DIR / (Path(data_filename).name + "_snn.avi")).as_posix()
    trajectories_filename = (RESULTS_DIR / (data_name + "_py.json")).as_posix()
    visTraj_filename = (RESULTS_DIR / (data_name + ".png")).as_posix()

    mov = cv2.VideoWriter(track_videoName, cv2.VideoWriter_fourcc(*'MJPG'), 30, (para_dict.get('spike_w'), para_dict.get('spike_h')))
    spike_tracker.get_results(spikes[calibration_time:], tracking_file, mov, save_video=True)

    spike_tracker.save_trajectory(RESULTS_DIR.as_posix(), data_name)
    vis_trajectory(trajectories_filename, visTraj_filename, **para_dict)

    if args.metrics:
        # measure the multi-object tracking performance
        print("Calculating metrics...")
        metrics = TrackingMetrics(tracking_file, **para_dict)
        metrics.get_results()

    # block_len = total_spikes.shape[0]
    mov.release()
    cv2.destroyAllWindows()
    print("RESULTS_DIR:", RESULTS_DIR)
    print("tracking_file:", tracking_file)

    # visualize the tracking results to a video
    # TODO: 使用 tracker.filtered_spikes 来生成结果还没有测试
    video_filename = os.path.join('results', filename[-1] + '_mot.avi')
    obtain_mot_video(spikes, video_filename, tracking_file, **para_dict)
    

if __name__ == "__main__":
    main()
