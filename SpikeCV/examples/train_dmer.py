# SpiReco:https://github.com/Evin-X/SpiReco
import os
import torch
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from spkProc.recognition.recog_dmer import net18

import os
from torch.utils.data import Dataset
import numpy as np


def TFI(spike, middle, window=50):
    '''
    Modified from the original codes of Rui Zhao (https://github.com/ruizhao26)
    '''
    C, H, W = spike.shape
    lindex, rindex = np.zeros([H, W]), np.zeros([H, W])
    l, r = middle+1, middle+1
    for r in range(middle+1, middle + window+1): 
        l = l - 1
        if l>=0:
            newpos = spike[l, :, :]*(1 - np.sign(lindex)) 
            distance = l*newpos
            lindex += distance
        if r<C:
            newpos = spike[r, :, :]*(1 - np.sign(rindex))
            distance = r*newpos
            rindex += distance
        if l<0 and r>=C:
            break

    rindex[rindex==0] = window+middle
    lindex[lindex==0] = middle-window
    interval = rindex - lindex
    tfi = 1.0 / interval 

    return tfi.astype(np.float32) 


class Spikedata(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir 
        self.T = 7

    def __getitem__(self, index):
        item = {}
        data = np.load(self.root_dir + '/{}.npz'.format(index))
        item['label'] = int(data['label'])
        spkmap = []
        dur = 50

        for t in range(self.T):
            tmp = TFI(data['spk'][0:100,13:237,13:237], 50+t, dur) # input size T*224*224
            tmp = np.where(tmp > 0.5, 0.5, tmp) / 0.5 # benefit for training
            spkmap.append(tmp)
        item['spkmap'] = np.array(spkmap)

        return item

    def __len__(self) -> int:
        return len(os.listdir(self.root_dir))


def build_dataset(path):
    train_path = path + '/trainset'
    val_path = path + '/testset'
    train_dataset = Spikedata(root_dir=train_path)
    val_dataset = Spikedata(root_dir=val_path)

    return train_dataset, val_dataset



parser = argparse.ArgumentParser("DMER_Net")
parser.add_argument('-dataset-dir', default='/home/Dataset/S-CALTECH', help='path to dataset')
parser.add_argument('-momentum', type=float, default=0.9, help='momentum')
parser.add_argument('-weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('-epochs', default=128, type=int, help='number of total epochs to run')
parser.add_argument('-w', default=12, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('-b', default=16, type=int, help='mini-batch size (default: 16)')
parser.add_argument('-lr', default=1e-3, type=float, help='init learning rate')
parser.add_argument('-gpu', type=str, default='0', help="device id to run")
parser.add_argument('-classes', type=int, default=101, help='number of category')


def main():
    global args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cudnn.benchmark = True
    cudnn.enabled=True

    train_dataset, val_dataset = build_dataset(args.dataset_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=args.w, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.b, shuffle=False, num_workers=args.w, pin_memory=True)

    criterion = nn.CrossEntropyLoss()

    model = net18(num_classes=args.classes)
    model = model.cuda()

    all_parameters = model.parameters()
    weight_parameters = []
    for pname, p in model.named_parameters():
        if p.ndimension() == 4 or 'conv' in pname:
            weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))
    optimizer = torch.optim.Adam(
            [{'params' : other_parameters},
            {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
            lr=args.lr,)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/args.epochs), last_epoch=-1)


    for epoch in range(args.epochs):

        # train 
        train(train_loader, model, criterion, optimizer, scheduler, epoch)

        # evaluate 
        validate(val_loader, model)


def train(train_loader, model, criterion, optimizer, scheduler, epoch):

    model.train()
    scheduler.step()
    print('Training Epoch: {}'.format(epoch))

    for batch_id, item in enumerate(train_loader):
        tgt_label = item['label'].cuda()
        input = item['spkmap'].cuda()
        pred_label = model(input)
        loss = criterion(pred_label, tgt_label)

        # compute gradient 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(val_loader, model):

    # switch to evaluation mode
    model.eval()
    total = 0
    correct = 0
    for batch_id, item in enumerate(val_loader):

        # compute output
        with torch.no_grad():
            tgt_label = item['label'].cuda()
            input = item['spkmap'].cuda()
            pred_label = model(input)
        _, predicts = torch.max(pred_label.data, 1)

        # measure accuracy
        total += tgt_label.size(0)
        correct += (predicts == tgt_label).sum().item()
        
    acc_test = 100 * correct / total
    print('Testing Accuracy: %.3f' % (acc_test))


 
if __name__ == '__main__':
    main()
