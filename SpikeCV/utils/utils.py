import sys
import numpy as np
import torch
import threading
import cv2
import json

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator


class dataReader(threading.Thread):
    def __init__(self, file_reader, device, q, is_dat=True, is_npy=False, filedir=None):
        super(dataReader, self).__init__()
        self.file_reader = file_reader
        self.device = device
        self.q = q
        self.is_dat = is_dat
        self.is_npy = is_npy
        self.filedir = filedir
        self.stream = torch.cuda.Stream()

    def run(self):
        with torch.cuda.stream(self.stream):
            for t in range(tnum):
                if self.is_dat:
                    ibuffer = self.file_reader.read(int(ivs_w * ivs_h / 8))
                    a = bin(int.from_bytes(ibuffer, byteorder=sys.byteorder))
                    a = a[2:].zfill(ivs_w * ivs_h)

                    a = list(a)
                    a = np.array(a, dtype=np.byte)
                    a = np.reshape(a, [ivs_h, ivs_w])
                    if ivs_h == 600:
                        a = np.flip(a, 0)
                    if ivs_h == 250:
                        a = np.flip(a, 1)
                    input_spk = torch.from_numpy(a != 0).to(device)
                elif self.is_npy:
                    npy_filename = self.filedir + str(t + 442) + '.npy'
                    tmp_data = np.load(npy_filename)
                    superResolution_rate = tmp_data.shape[2]
                    for i_data in range(superResolution_rate):
                        tmp_spk = tmp_data[:, :, i_data]
                        input_spk = torch.from_numpy(tmp_spk).to(device)
                        self.q.put(input_spk)

                else:
                    # img_filename = self.filedir + str(t + 4200) + '.png'
                    img_filename = self.filedir + 'spike_' + str(t + 1) + '.png'
                    # print('reading %d frames' % (t+1))
                    # print('reading %d frames' % (t+5000))
                    a = cv2.imread(img_filename)
                    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
                    a = a / 255
                    a = np.array(a, dtype=np.byte)
                    input_spk = torch.from_numpy(a != 0).to(device)

                self.q.put(input_spk)


# obtain 2D gaussian filter
def get_kernel(filter_size, sigma):
    assert (filter_size + 1) % 2 == 0, '2D filter size must be odd number!'
    g = np.zeros((filter_size, filter_size), dtype=np.float32)
    half_width = int((filter_size - 1) / 2)
    # center location

    xc = (filter_size + 1) / 2
    yc = (filter_size + 1) / 2
    for i in range(-half_width, half_width + 1, 1):
        for j in range(-half_width, half_width + 1, 1):
            x = int(xc + i)
            y = int(yc + j)
            g[y - 1, x - 1] = np.exp(- (i ** 2 + j ** 2) / 2 / sigma / sigma)

    g = (g - g.min()) / (g.max() - g.min())
    return g


def get_transform_matrix(ori, speed):
    ori_num = len(ori)
    speed_num = len(speed)
    transform_matrix = torch.zeros(ori_num * speed_num, 2, 3)
    cnt = 0
    for iOri in range(ori_num):
        for iSpeed in range(speed_num):
            transform_matrix[cnt, 0, 0] = 1
            transform_matrix[cnt, 1, 1] = 1

            transform_matrix[cnt, 0, 2] = - float(ori[iOri, 1] * speed[iSpeed] / ivs_w)
            transform_matrix[cnt, 1, 2] = - float(ori[iOri, 0] * speed[iSpeed] / ivs_h)

            cnt += 1

    transform_matrix = transform_matrix.to(device)
    return transform_matrix


def get_transform_matrix_new(ori, speed, dvs_w, dvs_h, device):
    ori_num = len(ori)
    speed_num = len(speed)
    transform_matrix = torch.zeros(ori_num * speed_num, 2, 3)
    cnt = 0
    for iOri in range(ori_num):
        for iSpeed in range(speed_num):
            transform_matrix[cnt, 0, 0] = 1
            transform_matrix[cnt, 1, 1] = 1

            transform_matrix[cnt, 0, 2] = - float(ori[iOri, 1] * speed[iSpeed] / dvs_w)
            transform_matrix[cnt, 1, 2] = - float(ori[iOri, 0] * speed[iSpeed] / dvs_h)

            cnt += 1

    transform_matrix = transform_matrix.to(device)
    return transform_matrix


# monitor the inference process
def visualize_img(gray_img, tag, curT):
    gray_img = gray_img.float32()
    img = torch.unsqueeze(gray_img, 0)
    logger.add_image(tag, img, global_step=curT)


def visualize_images(images, tag, curT):
    if images.shape[0] < 1:
        return
    images = torch.squeeze(images)
    img_num = images.shape[-1]
    for iImg in range(img_num):
        tmp_img = images[:, :, iImg]
        tmp_img = torch.squeeze(tmp_img)
        tmp_img = torch.unsqueeze(tmp_img, 0)
        logger.add_image(tag + str(iImg), tmp_img, global_step=curT)


def visualize_weights(weights, tag, curT):
    if weights.shape[0] < 1:
        return
    weights = torch.squeeze(weights)
    weights_num = weights.shape[0]
    input_size = weights.shape[1]
    stim_size = int(np.sqrt(input_size))
    for iw in range(weights_num):
        tmp_w = weights[iw, :]
        tmp_w = torch.squeeze(tmp_w)
        tmp_w = (tmp_w - torch.min(tmp_w)) / (torch.max(tmp_w) - torch.min(tmp_w))
        tmp_w = torch.reshape(tmp_w, (stim_size, stim_size))
        tmp_w = torch.unsqueeze(tmp_w, 0)
        logger.add_image(tag + str(iw), tmp_w, global_step=curT)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def vis_trajectory(json_file, filename, **dataDict):
    spike_h = dataDict.get('spike_h')
    spike_w = dataDict.get('spike_w')
    traj_dict = []
    with open(json_file, 'r') as f:
        for line in f.readlines():
            traj_dict.append(json.loads(line))

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
    ax.yaxis.set_major_locator(MultipleLocator(100))
    fig.subplots_adjust(top=1., bottom=0., left=0.2, right=1.)
    # fig.tight_layout()
    
    # 修复：在非交互式环境中不调用 plt.show()
    # plt.show()
    plt.savefig(filename, dpi=500, transparent=True)
    plt.close()
