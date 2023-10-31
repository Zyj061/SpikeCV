import sys

sys.path.append("../spkProc/depth_estimation/SpikeT")
import argparse
import json
import logging
import os
from os.path import join

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch
from model.S2DepthNet import S2DepthTransformerUNetConv
from utils.data_augmentation import CenterCrop, RandomCrop

logging.basicConfig(level=logging.INFO, format='')


def RawToSpike(video_seq, h, w, flipud=True):

    video_seq = np.array(video_seq).astype(np.uint8)
    img_size = h*w
    img_num = len(video_seq)//(img_size//8)
    SpikeMatrix = np.zeros([img_num, h, w], np.uint8)
    pix_id = np.arange(0,h*w)
    pix_id = np.reshape(pix_id, (h, w))
    comparator = np.left_shift(1, np.mod(pix_id, 8))
    byte_id = pix_id // 8

    for img_id in np.arange(img_num):
        id_start = int(img_id)*int(img_size)//8
        id_end = int(id_start) + int(img_size)//8
        cur_info = video_seq[id_start:id_end]
        data = cur_info[byte_id]
        result = np.bitwise_and(data, comparator)
        if flipud:
            SpikeMatrix[img_id, :, :] = np.flipud((result == comparator))
        else:
            SpikeMatrix[img_id, :, :] = (result == comparator)

    return SpikeMatrix


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def make_colormap(img, color_mapper):
    color_map_inv = np.ones_like(img[0]) * np.amax(img[0]) - img[0]
    color_map_inv = np.nan_to_num(color_map_inv, nan=1)
    color_map_inv = color_map_inv / np.amax(color_map_inv)
    color_map_inv = np.nan_to_num(color_map_inv)
    color_map_inv = color_mapper.to_rgba(color_map_inv)
    color_map_inv[:, :, 0:3] = color_map_inv[:, :, 0:3][..., ::-1]
    return color_map_inv


def main(config, initial_checkpoint):
    use_phased_arch = config['use_phased_arch']
    realdata_folders = 'D:/Datasets/PKU-Vidar-DVS-Dataset/test/Vidar/00232_driving_outdoor3'
    realdata_result = '../spkProc/depth_estimation/SpikeT/s2d_weights/debug_A100_SpikeTransformerUNetConv_LocalGlobal-Swin3D-T/00232_driving_outdoor3'
    ensure_dir(realdata_result)
    realdatas = os.listdir(realdata_folders)
    
    config['model']['gpu'] = config['gpu']
    config['model']['every_x_rgb_frame'] = config['data_loader']['train']['every_x_rgb_frame']
    config['model']['baseline'] = config['data_loader']['train']['baseline']
    config['model']['loss_composition'] = config['trainer']['loss_composition']
    
    model = eval(config['arch'])(config['model'])
    if initial_checkpoint is not None:
        print('Loading initial model weights from: {}'.format(initial_checkpoint))
        checkpoint = torch.load(initial_checkpoint)
        print(checkpoint['state_dict'])
        model = torch.nn.DataParallel(model).cuda()
        if use_phased_arch:
            C, (H, W) = config["model"]["num_bins_events"], config["model"]["spatial_resolution"]
            dummy_input = torch.Tensor(1, C, H, W)
            times = torch.Tensor(1)
            _ = model.forward(dummy_input, times=times, prev_states=None)
        print(model.state_dict)
        model.load_state_dict(checkpoint['state_dict'])

    gpu = torch.device('cuda:' + str(config['gpu']))
    model.to(gpu)
    model.eval()

    data_tranfsorm = CenterCrop(224)

    # construct color mapper, such that same color map is used for all outputs.
    spike_path = os.path.join(realdata_folders, realdatas[0])
    f = open(spike_path, 'rb')
    spike_seq = f.read()
    spike_seq = np.frombuffer(spike_seq, 'b')
    spikes = RawToSpike(spike_seq, 250, 400)
    spikes = spikes.astype(np.float32)
    spikes = torch.from_numpy(spikes)
    f.close()
    data = data_tranfsorm(spikes)
    dT, dH, dW = data.shape
    item = {}
    item['image'] = data
    input = {}
    input['image'] = data[None, dT//2-64:dT//2+64]
    prev_super_states = {'image': None}
    prev_states_lstm = {}
    new_predicted_targets, _, _ = model(input, prev_super_states['image'], prev_states_lstm)
    
    frame = new_predicted_targets['image'][0].detach().cpu().numpy()
    color_map_inv = np.ones_like(frame[0]) * np.amax(frame[0]) - frame[0]
    color_map_inv = np.nan_to_num(color_map_inv, nan=1)
    color_map_inv = color_map_inv / np.amax(color_map_inv)
    color_map_inv = np.nan_to_num(color_map_inv)
    vmax = np.percentile(color_map_inv, 95)
    normalizer = mpl.colors.Normalize(vmin=color_map_inv.min(), vmax=vmax)
    color_mapper_overall = cm.ScalarMappable(norm=normalizer, cmap='magma')

    with torch.no_grad():
        for datapaths in realdatas:
            spike_path = os.path.join(realdata_folders, datapaths)
            f = open(spike_path, 'rb')
            spike_seq = f.read()
            spike_seq = np.frombuffer(spike_seq, 'b')
            spikes = RawToSpike(spike_seq, 250, 400)
            spikes = spikes.astype(np.float32)
            spikes = torch.from_numpy(spikes)
            f.close()
            data = data_tranfsorm(spikes)
            
            print(data.shape)
            dT, dH, dW = data.shape
            item = {}
            item['image'] = data
            input = {}
            input['image'] = data[None, dT//2-64:dT//2+64]
            prev_super_states = {'image': None}
            prev_states_lstm = {}
            
            new_predicted_targets, _, _ = model(input, prev_super_states['image'], prev_states_lstm)
            
            predict_depth = new_predicted_targets['image']
            print(predict_depth.shape)
            predict_depth = predict_depth[0].cpu().numpy()
            ensure_dir(realdata_result)
            img = predict_depth
            cv2.imwrite(join(realdata_result, 'frame_%s.png' % datapaths[:-4]), img[0][:, :, None] * 255.0)

            spikes = data.permute(1,2,0).cpu().numpy()
            input_spikes = np.mean(spikes, axis=2).astype(np.float32)
            cv2.imwrite(join(realdata_result, 'input_%s.png' % datapaths[:-4]), input_spikes[:, :, None] * 255.0)

            # save color map
            color_map = make_colormap(img, color_mapper_overall)
            cv2.imwrite(join(realdata_result, 'color_%s.png' % datapaths[:-4]), color_map * 255.0)


if __name__ == '__main__':
    logger = logging.getLogger()
    parser = argparse.ArgumentParser(
        description='Inference depth map from monocular spike stream')
    parser.add_argument('--path_to_model', type=str,
                        help='path to the model weights',
                        default='')
    parser.add_argument('--config', type=str,
                        help='path to config. If not specified, config from model folder is taken',
                        default=None)
    parser.add_argument('--data_folder', type=str,
                        help='path to folder of data to be tested',
                        default=None)
    args = parser.parse_args()

    if args.config is None:
        head_tail = os.path.split(args.path_to_model)
        config = json.load(open(os.path.join(head_tail[0], 'config.json')))
    else:
        config = json.load(open(args.config))

    main(config, args.path_to_model)
