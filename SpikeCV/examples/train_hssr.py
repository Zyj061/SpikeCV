# HSSR:https://github.com/Evin-X/HSSR
from __future__ import print_function
import argparse
import numpy as np
import os, sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.utils.data import Dataset
from spkProc.recognition.recon_hssr import HSSR_Net


def progress_bar(batch, total):
    """
    Displays or updates a console progress bar.

    Original source: https://stackoverflow.com/a/15860757/1391441
    """
    barLength, status = 20, ""
    progress = batch
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\rBatch {:.0f}  [{}] {:.0f}% {}".format(batch + 1,
        "#" * block + "-" * (barLength - block), round(progress * 100, 0),
        status)
    sys.stdout.write(text)
    sys.stdout.flush()


class Dataset_loader(Dataset):
    def __init__(self, root_dir, key, cls):
        super(Dataset_loader, self).__init__()
        self.root_dir = root_dir 
        self.key = key
        self.cls = cls
 
    def __getitem__(self, index: int):
        item = {}
        data = np.load(self.root_dir + '/{}.npz'.format(index))
        tag = int(np.random.random() * 80)
        item['input'] = np.expand_dims(data['spk'][tag:tag+20], axis=0).astype(np.float32)
        item['image'] = data['frame'].astype(np.float32)
        if self.cls == 28:
            item['label'] = data['label_28']
        else:
            item['label'] = data['label']
    
        return item

    def __len__(self) -> int:
        return len(os.listdir(self.root_dir))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train-dir', default='./input/train', help='path to dataset')
    parser.add_argument('-valid-dir', default='./input/test', help='path to dataset')
    parser.add_argument('-w', default=20, type=int, help='number of data loading workers')
    parser.add_argument('-b', default=16, type=int, help='batch size')
    parser.add_argument('-lr', default=1e-4, type=float, help='initial learning rate') 
    parser.add_argument('-weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('-epochs', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('-gpu', type=str, default='0', help="device id to run")
    parser.add_argument('-classes', default=100, type=int)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cudnn.benchmark = True
    model = HSSR_Net(args.classes)
    model = model.cuda()

    train_dataset = Dataset_loader(args.train_dir, args.key, args.classes)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=args.w, pin_memory=True)
    valid_dataset = Dataset_loader(args.valid_dir, args.key, args.classes)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.b, shuffle=False, num_workers=args.w, pin_memory=True)

    optim = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr = args.lr, momentum = 0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda step : (1.0-step/args.epochs), last_epoch=-1)
 
    cri_mse = nn.MSELoss()
    cri_ce = nn.CrossEntropyLoss()

    # Training
    print("Start training!")
    for epoch in range(args.epochs):
        print('EPOCH {:d} / {:d}'.format(epoch + 1, args.epochs))

        model.train()
        for param_group in optim.param_groups:
            cur_lr = param_group['lr']

        for batch_idx, item in enumerate(train_loader):
            progress_bar(batch_idx, len(train_loader))

            tgt_label = item['label'].cuda()
            input = item['input'].cuda()
            tgt_img = item['image'].cuda()
            pred_label, pred_fea, tgt_fea = model(input, tgt_img)
            loss_1 = cri_mse(pred_fea, tgt_fea)
            loss_2 = cri_ce(pred_label, tgt_label)

            loss = loss_2 + 0.1*loss_1
            
            optim.zero_grad()
            loss.backward()
            optim.step()
        scheduler.step()


        # validation
        model.eval()
        total_item = 0
        correct_item = 0
        for batch_idx, item in enumerate(valid_loader):
            # compute output
            with torch.no_grad():
                tgt_label = item['label'].cuda()
                input = item['input'].cuda()
                tgt_img = item['image'].cuda()
                outputs, _, _ = model(input, tgt_img)
                _, predicts = torch.max(outputs.data, 1)
            # measure accuracy
            total_item += tgt_label.size(0)
            correct_item += (predicts == tgt_label).sum().item()
        
        acc = 100 * correct_item / total_item
        print('\naccuracy: %.4f\n' % (acc))
