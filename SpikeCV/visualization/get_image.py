# -*- coding: utf-8 -*- 
# @Time : 2023/8/20 16:06 
# @Author : Yajing Zheng
# @Email: yj.zheng@pku.edu.cn
# @File : get_image.py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator
from matplotlib.patches import Rectangle
import torch
import copy

def get_spike_raster(data):
    num_neuron, timesteps = data.shape
    colors = [f'C{i}' for i in range(num_neuron)]
    # set different line properties for each set of positions
    # note that some overlap
    lineoffsets1 = np.array(range(1, num_neuron*2+1, 2))
    linelengthts1 = np.ones((num_neuron, )) * 1.5

    plt.figure(figsize=(8, 6))
    plt.eventplot(data, colors=colors, lineoffsets=lineoffsets1, linelengths=linelengthts1)
    return plt.gcf()


def get_heatmap_handle(data, marker=None, bounding_box=None):

    if torch.is_tensor(data):
        data = copy.deepcopy(data.cpu().detach().numpy())

    fig, ax = plt.subplots(figsize=(8, 6))
    h, w = data.shape
    if marker is not None:
        num_points = marker.shape[1]
        colors = [f'C{i}' for i in range(num_points)]
        for i_point in range(num_points):
            ax.plot(marker[1, i_point], h-marker[0, i_point], 'o', color=colors[i_point], markersize=10)
            ax.annotate('P{}'.format(i_point), (marker[1, i_point], h-marker[0, i_point]))

    if bounding_box is not None:
        for i_box, bbox in enumerate(bounding_box):
            ax.add_patch(Rectangle((bbox[1], bbox[0]), bbox[3]-bbox[1], bbox[2] - bbox[0],
                                   edgecolor='red', facecolor='none', lw=2))

    ax.imshow(data, cmap='Blues', interpolation='nearest')

    # plt.colorbar()
    plt.axis('off')  # 可选，关闭坐标轴
    plt.title('Heatmap')

    return plt.gcf()


def get_histogram_handle(data, marker=None, bounding_box=None):

    if torch.is_tensor(data):
        data = copy.deepcopy(data.cpu().detach().numpy())

    fig, ax = plt.subplots(figsize=(8, 6))
    h, w = data.shape
    ax.hist(data.reshape((-1, 1)), bins=20)

    # plt.colorbar()
    # plt.axis('off')  # 可选，关闭坐标轴
    plt.title('Heatmap')

    return plt.gcf()
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
    # fig.tight_layout()
    # plt.savefig(filename, dpi=500, transparent=True)
    # filename = filename.replace('png', 'eps')
    # plt.savefig(filename, dpi=500, transparent=True)
    plt.show()
