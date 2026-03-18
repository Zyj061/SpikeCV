# -*- coding: utf-8 -*- 
# @Time : 2023/7/16 20:23 
# @Author : Yajing Zheng
# @Email: yj.zheng@pku.edu.cn
# @File : snn_tracker.py
import os, sys
sys.path.append('../..')
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from spkProc.filters.stp_filters_torch import STPFilter
# from filters import stpFilter
from spkProc.detection.attention_select_v2 import SaccadeInput
from spkProc.motion.motion_detection import motion_estimation
from spkProc.detection.stdp_clustering import stdp_cluster
# from spkProc.detection.stdp_clustering_V2 import stdp_cluster
from utils.utils import NumpyEncoder
from collections import namedtuple
import json
import cv2
from tqdm import tqdm

trajectories = namedtuple('trajectories', ['id', 'x', 'y', 't', 'color'])

# TODO: docstring
class SNNTracker:
    def __init__(self, spike_h, spike_w, device, attention_size=20, diff_time=1, **STPargs):

        self.spike_h = spike_h
        self.spike_w = spike_w
        self.device = device

        # self.stp_filter = STPFilter(spike_h, spike_w, device)
        if STPargs is not None:
            self.stp_filter = STPFilter(spike_h, spike_w, device, diff_time, **STPargs)
        else:
            self.stp_filter = STPFilter(spike_h, spike_w, device, diff_time)
        # self.stp_filter = stpFilter()
        self.attention_size = attention_size
        self.object_detection = SaccadeInput(spike_h, spike_w, box_size=self.attention_size, device=device)
        from tensorboardX import SummaryWriter
        logger = SummaryWriter(log_dir='data/log_pkuvidar')
        self.motion_estimator = motion_estimation(spike_h, spike_w, device, logger=logger)
        # gpu_tracker.track()  # run function between the code line where uses GPU

        self.object_cluster = stdp_cluster(spike_h, spike_w, box_size=self.attention_size, device=device)

        # self.timestamps = spikes.shape[0]
        # self.filterd_spikes = np.zeros([self.timestamps, self.spike_h, self.spike_w], np.uint8)
        self.calibration_time = 150
        self.timestamps = 0
        self.trajectories = {}
        self.filterd_spikes = []

    def calibrate_motion(self, spikes, calibration_time=None):

        if calibration_time is None:
            calibration_time = self.calibration_time
        else:
            self.calibration_time = calibration_time

        print('begin calibrate..')
        for t in range(calibration_time):
            input_spk = torch.from_numpy(spikes[t, :, :].copy()).to(self.device)
            self.stp_filter.update_dynamics(t, input_spk)
            self.timestamps += 1

    def get_results(self, spikes, res_filepath, mov_writer=None, save_video=False):

        result_file = open(res_filepath, 'w')  # HACK: 改为写入模式，避免累积重复数据

        timestamps = spikes.shape[0]
        total_time = 0
        predict_kwargs = {'spike_h': self.spike_h, 'spike_w': self.spike_w, 'device': self.device}

        # Timing stats for stdp_tracking
        stdp_count = 0
        stdp_sum_ms = 0.0
        stdp_min_ms = float('inf')
        stdp_max_ms = 0.0
        stdp_timing_data = []  # 记录每次调用的耗时数据

        for t in tqdm(range(timestamps), desc=f'tracking'):
            try:
                input_spk = torch.from_numpy(spikes[t, :, :].copy()).to(self.device)
                self.stp_filter.update_dynamics(self.timestamps, input_spk)

                self.stp_filter.local_connect(self.stp_filter.filter_spk)
                # self.filterd_spikes[t, :, :] = self.stp_filter.lif_spk.cpu().detach().numpy()

                self.object_detection.update_dnf(self.stp_filter.lif_spk)
                attentionBox, attentionInput = self.object_detection.get_attention_location(self.stp_filter.lif_spk)
                # attentionInput = attentionInput.to(self.device)
                num_box = attentionBox.shape[0]

                # Perform motion tracking using STDP
                _t00 = time.perf_counter()
                self.motion_estimator.stdp_tracking(self.stp_filter.lif_spk)
                _t01 = time.perf_counter()
                elapsed_ms = (_t01 - _t00) * 1000.0
                stdp_count += 1
                stdp_sum_ms += elapsed_ms
                stdp_timing_data.append(elapsed_ms)  # 记录每次调用的耗时
                if elapsed_ms < stdp_min_ms:
                    stdp_min_ms = elapsed_ms
                if elapsed_ms > stdp_max_ms:
                    stdp_max_ms = elapsed_ms

                motion_id, motion_vector, _ = self.motion_estimator.local_wta(self.stp_filter.lif_spk, self.timestamps, visualize=False)
                # gpu_tracker.track()  # run function between the code line where uses GPU

                predict_fire, sw, bw = self.object_cluster.update_weight(attentionInput)

                predict_object = self.object_cluster.detect_object(predict_fire, attentionBox, motion_id, motion_vector, **predict_kwargs)

                # visualize_weights(sw, 'before update tracks', t)

                sw, bw = self.object_cluster.update_tracks(predict_object, sw, bw, self.timestamps)

                self.object_cluster.synaptic_weight = sw.detach().clone()
                self.object_cluster.bias_weight = bw.detach().clone()

                dets = torch.zeros((num_box, 6), dtype=torch.int)
                for i_box, bbox in enumerate(attentionBox):
                    dets[i_box, :] = torch.tensor([bbox[0], bbox[1], bbox[2], bbox[3], 1, 1])

                track_ids = []
                if save_video:
                    track_frame = self.stp_filter.lif_spk.cpu().numpy()
                    track_frame = (track_frame * 255).astype(np.uint8)
                    # track_frame = np.transpose(track_frame, (1, 2, 0))

                    # track_frame = np.tile(track_frame, (3, 1, 1))
                    # track_frame = np.squeeze(track_frame)
                    track_frame = cv2.cvtColor(track_frame, cv2.COLOR_GRAY2BGR)

                    for i_box in range(attentionBox.shape[0]):
                        tmp_box = attentionBox[i_box, :]
                        cv2.rectangle(track_frame, (int(tmp_box[1]), int(tmp_box[0])), (int(tmp_box[3]), int(tmp_box[2])),
                                           (int(0), int(0), int(255)), 2)

                for i_box in range(self.object_cluster.K2):
                    if self.object_cluster.tracks[i_box].visible == 1:
                        tmp_box = self.object_cluster.tracks[i_box].bbox.numpy()
                        pred_box = self.object_cluster.tracks[i_box].predbox.numpy()
                        id = self.object_cluster.tracks[i_box].id
                        color = self.object_cluster.tracks[i_box].color

                        # update the trajectories
                        mid_y = (tmp_box[0, 0] + tmp_box[0, 2]) / 2  # height
                        mid_x = (tmp_box[0, 1] + tmp_box[0, 3]) / 2  # width
                        box_w = int(tmp_box[0, 3] - tmp_box[0, 1])
                        box_h = int(tmp_box[0,2] - tmp_box[0, 0])
                        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                            self.timestamps, id, tmp_box[0, 1], tmp_box[0, 0], box_w, box_h), file=result_file)

                        if id not in self.trajectories:
                            self.trajectories[id] = trajectories(int(id), [], [], [], 255 * np.random.rand(1, 3))
                            self.trajectories[id].x.append(mid_x)
                            self.trajectories[id].y.append(mid_y)
                            self.trajectories[id].t.append(self.timestamps)

                        else:
                            self.trajectories[id].x.append(mid_x)
                            self.trajectories[id].y.append(mid_y)
                            self.trajectories[id].t.append(self.timestamps)
                            # the detection results

                        if save_video:
                            cv2.rectangle(track_frame, (int(tmp_box[0, 1]), int(tmp_box[0, 0])),
                                          (int(tmp_box[0, 3]), int(tmp_box[0, 2])),
                                          (int(color[0, 0]), int(color[0, 1]), int(color[0, 2])), 2)

                            # # the predicted results
                            # cv2.rectangle(track_frame, (int(pred_box[0, 1]), int(pred_box[0, 0])),
                            #               (int(pred_box[0, 3]), int(pred_box[0, 2])), (int(0), int(0), int(255)), 2)

                            # the label box
                            cv2.rectangle(track_frame, (int(tmp_box[0, 1]), int(tmp_box[0, 0] - 35)),
                                          (int(tmp_box[0, 1] + 60), int(tmp_box[0, 0])),
                                          (int(color[0, 0]), int(color[0, 1]), int(color[0, 2])), -1)
                            if self.object_cluster.tracks[i_box].unvisible_count > 0:
                                show_text = 'predict' + str(id)
                            else:
                                show_text = 'object' + str(id)
                            cv2.putText(track_frame, show_text, (int(tmp_box[0, 1]), int(tmp_box[0, 0] - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (255, 255, 255), 2)

                if save_video:
                    cv2.putText(track_frame, str(int(self.timestamps)),
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    mov_writer.write(track_frame)
                self.timestamps += 1

            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print('WARNING: out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    else:
                        raise exception

        # Print stdp_tracking timing summary
        if stdp_count > 0:
            stdp_avg_ms = stdp_sum_ms / stdp_count
            print('stdp_tracking avg/min/max over %d frames: %.3f / %.3f / %.3f ms' % (
                stdp_count, stdp_avg_ms, stdp_min_ms, stdp_max_ms))
            
            # 绘制耗时曲线图
            self._plot_timing_curve(stdp_timing_data, res_filepath)

        print('Total tracking took: %.3f seconds for %d timestamps spikes' %
              (total_time, self.timestamps - self.calibration_time))

        # if save_video:
        #     mov_writer.release()
        #     cv2.destroyAllWindows()

        result_file.close()

    def _plot_timing_curve(self, timing_data, res_filepath):
        """
        绘制stdp_tracking耗时曲线图
        Args:
            timing_data: 耗时数据列表
            res_filepath: 结果文件路径，用于生成图片文件名
        """
        if not timing_data:
            return
            
        # 生成调用ID（从1开始）
        call_ids = list(range(1, len(timing_data) + 1))
        
        # 创建图形
        plt.figure(figsize=(12, 6))
        plt.plot(call_ids, timing_data, 'b-', linewidth=1, alpha=0.7)
        plt.scatter(call_ids, timing_data, c='red', s=10, alpha=0.6)
        
        # 设置图形属性
        plt.xlabel('Call ID', fontsize=12)
        plt.ylabel('Time (ms)', fontsize=12)
        plt.title('STDP Tracking Time Curve', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        avg_time = np.mean(timing_data)
        min_time = np.min(timing_data)
        max_time = np.max(timing_data)
        
        plt.axhline(y=avg_time, color='green', linestyle='--', alpha=0.7, 
                   label=f'Average: {avg_time:.3f} ms')
        plt.axhline(y=min_time, color='blue', linestyle=':', alpha=0.7, 
                   label=f'Minimum: {min_time:.3f} ms')
        plt.axhline(y=max_time, color='red', linestyle=':', alpha=0.7, 
                   label=f'Maximum: {max_time:.3f} ms')
        
        plt.legend()
        
        # 生成图片文件名
        base_name = os.path.splitext(os.path.basename(res_filepath))[0]
        plot_filename = f"{base_name}_stdp_timing_curve.png"
        plot_path = os.path.join(os.path.dirname(res_filepath), plot_filename)
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'Timing curve plot saved to: {plot_path}')
        
        # 保存耗时数据到CSV文件
        csv_filename = f"{base_name}_stdp_timing_data.csv"
        csv_path = os.path.join(os.path.dirname(res_filepath), csv_filename)
        
        with open(csv_path, 'w') as f:
            f.write('Call ID,Time(ms)\n')
            for i, time_ms in enumerate(timing_data):
                f.write(f'{i+1},{time_ms:.6f}\n')
        
        print(f'Timing data saved to: {csv_path}')

    def save_trajectory(self, results_dir, data_name):
        trajectories_filename = os.path.join(results_dir, data_name + '_py.json')
        mat_trajectories_filename = 'results/' + data_name + '.json'
        track_box_filename = 'results/' + data_name + '_bbox.json'

        if os.path.exists(trajectories_filename):
            os.remove(trajectories_filename)

        if os.path.exists(mat_trajectories_filename):
            os.remove(mat_trajectories_filename)

        if os.path.exists(track_box_filename):
            os.remove(track_box_filename)

        for i_traj in range(self.object_cluster.K2):
            tmp_traj = self.object_cluster.trajectories[i_traj]
            tmp_bbox = self.object_cluster.tracks_bbox[i_traj]

            traj_json_string = json.dumps(tmp_traj._asdict(), cls=NumpyEncoder)
            bbox_json_string = json.dumps(tmp_bbox._asdict(), cls=NumpyEncoder)

            with open(mat_trajectories_filename, 'a+') as f:
                f.write(traj_json_string)

            with open(track_box_filename, 'a+') as f:
                f.write(bbox_json_string)

        num_len = len(self.trajectories)
        for i_traj in self.trajectories:
            traj_json_string = json.dumps(self.trajectories[i_traj]._asdict(), cls=NumpyEncoder)

            with open(trajectories_filename, 'a+') as f:
                f.write(traj_json_string)

                f.write('\n')
