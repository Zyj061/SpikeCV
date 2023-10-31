# -*- coding: utf-8 -*- 
# @Time : 2022/12/06
# @Author : Shiyan Chen
# @File : train_ssml_recon.py
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
from spkProc.reconstruction.SSML_Recon.ssml_model import SSML_ReconNet
import os
from utils import path
import argparse
from datetime import datetime
import torch.backends.cudnn as cudnn
import random
from torch.optim import Adam, lr_scheduler
from spkData.load_dat import Dataset_RealSpike


def progress_bar(batch_idx, num_batches, report_interval, train_loss):
    """Neat progress bar to track training."""

    dec = int(np.ceil(np.log10(num_batches)))
    bar_size = 21 + dec
    progress = (batch_idx % report_interval) / report_interval
    fill = int(progress * bar_size) + 1
    print('\rBatch {:>{dec}d} [{}{}] Train loss: {:>1.5f}'.format(batch_idx + 1, '=' * fill + '>', ' ' * (bar_size - fill), train_loss, dec=str(dec)), end='')

def time_elapsed_since(start):
    """Computes elapsed time since start."""

    timedelta = datetime.now() - start
    string = str(timedelta)[:-7]
    ms = int(timedelta.total_seconds() * 1000)

    return string, ms

def save_network(network, save_path):
    if isinstance(network, nn.DataParallel):
        network = network.module
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', default='./bubble', help='path to dataset')
    parser.add_argument('--valid-dir', default='./bubble', help='path to dataset')

    parser.add_argument('--ckpt-dir', default='./ckpts/expr1', help='folder to output images and model checkpoints')

    parser.add_argument('--batch-size', type=int, default=4, help='input batch size')
    parser.add_argument('--crop-size', type=int, default=256, help='the height / width of the input image to network')

    parser.add_argument('--nb_epochs', type=int, default=50000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate, default=0.0002')

    parser.add_argument('--manualSeed', type=int, default=42, help='manual seed')

    opt = parser.parse_args()
    print(opt)

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    
    crop_size = opt.crop_size
    train_dir = opt.train_dir
    valid_dir = opt.valid_dir
    batch_size = opt.batch_size
    nb_epochs = opt.nb_epochs
    ckpt_dir = opt.ckpt_dir
    if not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)

    train_dataset = Dataset_RealSpike(train_dir,crop_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=20,pin_memory=True)

    valid_dataset = Dataset_RealSpike(valid_dir,0)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False,num_workers=20,pin_memory=True)

    model = SSML_ReconNet()
    optim = Adam(model.parameters(), lr=opt.lr,betas=[0.9,0.99])
    scheduler = lr_scheduler.MultiStepLR(optim, milestones=[10000], gamma=0.5)   
    criterion = nn.MSELoss()

    model = model.cuda()
    model = nn.DataParallel(model)

    max_psnr = 0

    print("Start training!")
    train_start = datetime.now()
    for epoch in range(nb_epochs):
        print('EPOCH {:d} / {:d}'.format(epoch + 1, nb_epochs))
        epoch_start = datetime.now()
        
        model.train(True)
        for batch_idx, item in enumerate(train_loader):
            progress_bar(batch_idx, len(train_loader), 50, 0)

            source = item['spikes'].cuda()

            optim.zero_grad()

            nbsn_pred, bsn_pred,tfi,tfp = model(source, train=True)

            loss1 = criterion(bsn_pred,tfi)
            loss2 = criterion(bsn_pred,nbsn_pred)
            loss = loss1 + 0.01 * loss2
            
            loss.backward()
            optim.step()

        scheduler.step()

        '''
        SSML用于真实场景重建, 没有Ground Truth, 无需且无法计算PSNR.
        '''
        ####### eval #######
        # model.eval()
        # psnr = 0
        # for batch_idx, item in enumerate(valid_loader):
        #     source = item['spikes'].cuda()
        #     optim.zero_grad()
            
        #     with torch.set_grad_enabled(False):
        #         out = model(source, train=False)
        #         psnr += 0
        # psnr /= len(valid_dataset)
        # print("PSNR: {:.4f}".format(psnr))

        path = "{}/ssml_ckpt_{}.pth".format(ckpt_dir, epoch)
        save_network(model, path)

        epoch_time = time_elapsed_since(epoch_start)[0]
        print('epoch_time: {}\n'.format(epoch_time))

    train_elapsed = time_elapsed_since(train_start)[0]
    print('Training done! Total elapsed time: {}\n'.format(train_elapsed))