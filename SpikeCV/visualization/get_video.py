# -*- coding: utf-8 -*- 
# @Time : 2022/6/12 15:21 
# @Author : Yajing Zheng
# @File : visualize.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from pprint import pprint
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator

def obtain_spike_video(spikes, video_filename, **dataDict):
    spike_h = dataDict.get('spike_h')
    spike_w = dataDict.get('spike_w')
    timestamps = spikes.shape[0]

    mov = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'MJPG'), 30, (spike_w, spike_h))

    for iSpk in range(timestamps):
        tmpSpk = spikes[iSpk, :, :] * 255
        tmpSpk = cv2.cvtColor(tmpSpk.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        mov.write(tmpSpk)

    mov.release()


def obtain_reconstruction_video(images, video_filename, **dataDict):
    spike_h = dataDict.get('spike_h')
    spike_w = dataDict.get('spike_w')

    img_num = images.shape[0]
    mov = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'MJPG'), 30, (spike_w, spike_h))
    for iImg in range(img_num):
        tmp_img = images[iImg, :, :]
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_GRAY2BGR)
        mov.write(tmp_img)

    mov.release()


def obtain_mot_video(spikes, video_filename, res_filepath, **dataDict):
    spike_h = dataDict.get('spike_h')
    spike_w = dataDict.get('spike_w')

    gt_file = dataDict.get('labeled_data_dir')
    gt_boxes = {}
    if gt_file is not None:
        gt_f = open(gt_file, 'r')
        gt_lines = gt_f.readlines()
        for line in gt_lines:
            gt_term = line.split(',')
            time_step = gt_term[0]
            box_id = gt_term[1]
            x = float(gt_term[2])
            y = float(gt_term[3])
            w = float(gt_term[4])
            h = float(gt_term[5])

            if str(time_step) not in gt_boxes:
                gt_boxes[str(time_step)] = []
            bbox = [box_id, x, y, w, h]
            gt_boxes[str(time_step)].append(bbox)

        gt_f.close()

    result_file = res_filepath
    test_boxes = {}
    result_f = open(result_file, 'r')
    result_lines = result_f.readlines()
    color_dict = {}

    for line in result_lines:
        res_box = line.split(',')
        time_step = res_box[0]
        track_id = res_box[1]
        if track_id not in color_dict.keys():
            colors = (np.random.rand(1, 3) * 255).astype(np.uint8)
            color_dict[track_id] = np.squeeze(colors)

        x = float(res_box[2])
        y = float(res_box[3])
        w = float(res_box[4])
        h = float(res_box[5])

        if str(time_step) not in test_boxes:
            test_boxes[str(time_step)] = []

        test_box = [track_id, x, y, w, h]
        test_boxes[str(time_step)].append(test_box)

    result_f.close()

    mov = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'MJPG'), 30, (spike_w, spike_h))

    timestamps = spikes.shape[0]
    for t in range(151, timestamps):
    # for t in range(160, 1000):
        tmp_ivs = spikes[t, :, :] * 255
        tmp_ivs = cv2.cvtColor(tmp_ivs.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        if len(gt_boxes) > 0:
            if str(t) in gt_boxes:
                gts = gt_boxes[str(t)]
                gt_num = len(gts)
                for i in range(gt_num):
                    box = gts[i]
                    box_id = box[0]
                    cv2.rectangle(tmp_ivs, (int(box[2]), int(box[1])),
                                  (int(box[2] + box[4]), int(box[1] + box[3])),
                                  (int(255), int(255), int(255)), 2)

        if str(t) in test_boxes:
            test = test_boxes[str(t)]
            test_num = len(test)
            for i in range(test_num):
                box = test[i]
                box_id = box[0]
                colors = color_dict[box_id]
                cv2.rectangle(tmp_ivs, (int(box[2]), int(box[1])),
                              (int(box[2] + box[4]), int(box[1] + box[3])),
                              (int(colors[0]), int(colors[1]), int(colors[2])), 2)

        mov.write(tmp_ivs)

    mov.release()


def obtain_detection_video(spikes, video_filename, res_filepath, **dataDict):
    spike_h = dataDict.get('spike_h')
    spike_w = dataDict.get('spike_w')

    result_file = res_filepath
    test_boxes = {}
    result_f = open(result_file, 'r')
    result_lines = result_f.readlines()
    color_dict = {}

    for line in result_lines:
        res_box = line.split(',')
        time_step = res_box[0]
        track_id = res_box[1]
        if track_id not in color_dict.keys():
            colors = (np.random.rand(1, 3) * 255).astype(np.uint8)
            color_dict[track_id] = np.squeeze(colors)

        x = float(res_box[2])
        y = float(res_box[3])
        w = float(res_box[4])
        h = float(res_box[5])

        if str(time_step) not in test_boxes:
            test_boxes[str(time_step)] = []

        test_box = [track_id, x, y, w, h]
        test_boxes[str(time_step)].append(test_box)

    result_f.close()

    mov = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'MJPG'), 30, (spike_w, spike_h))

    block_len = spikes.shape[0]
    for t in range(150, block_len):
        tmp_ivs = spikes[t, :, :] * 255
        tmp_ivs = cv2.cvtColor(tmp_ivs.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        if str(t) in test_boxes:
            test = test_boxes[str(t)]
            test_num = len(test)
            for i in range(test_num):
                box = test[i]
                box_id = box[0]
                colors = color_dict[box_id]
                cv2.rectangle(tmp_ivs, (int(box[2]), int(box[1])),
                              (int(box[2] + box[4]), int(box[1] + box[3])),
                              (int(colors[0]), int(colors[1]), int(colors[2])), 2)

        mov.write(tmp_ivs)

    mov.release()


def vis_trajectory(box_file, json_file, filename, **dataDict):

    spike_h = dataDict.get('spike_h')
    spike_w = dataDict.get('spike_w')
    traj_dict = []
    with open(json_file, 'r') as f:
        for line in f.readlines():
            traj_dict.append(json.loads(line))

    box_file = open(box_file, 'r')
    result_lines = box_file.readlines()
    num_traj = len(traj_dict)

    fig = plt.figure(figsize=[10, 6])
    ax = fig.add_subplot(111, projection='3d')
    min_t = 1000
    max_t = 0

    for tmp_traj in traj_dict:
        tmp_t = np.array(tmp_traj['t'])
        if np.min(tmp_t) < min_t:
            min_t = np.min(tmp_t)
        if np.max(tmp_t) > max_t:
           max_t = np.max(tmp_t)

        tmp_x = spike_w - np.array(tmp_traj['x'])
        tmp_y = np.array(tmp_traj['y'])
        tmp_color = np.array(tmp_traj['color']) / 255.
        ax.plot(tmp_t, tmp_x, tmp_y, color=tmp_color, linewidth=2, label='traj ' + str(tmp_traj['id']))

    ax.legend(loc='best', bbox_to_anchor=(0.7, 0., 0.4, 0.8))
    zoom = [2.2, 0.8, 0.5, 1]
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([zoom[0], zoom[1], zoom[2], zoom[3]]))
    ax.set_xlim(min_t, max_t)
    ax.set_ylim(0, spike_w)
    ax.set_zlim(0, spike_h)

    ax.set_xlabel('time', fontsize=15)
    ax.set_ylabel('width', fontsize=15)
    ax.set_zlabel('height', fontsize=15)

    ax.view_init(elev=16, azim=135)
    # ax.view_init(elev=2, azim=27)
    ax.yaxis.set_major_locator(MultipleLocator(100))
    fig.subplots_adjust(top=1., bottom=0., left=0.2, right=1.)
    plt.show()






