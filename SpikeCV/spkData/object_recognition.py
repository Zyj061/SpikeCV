import os
import torch
import numpy as np
from SpikeCV.spkData.load_dat import SpikeStream
from torchvision.datasets.folder import make_dataset

class DatasetRecognition(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.rootpath = kwargs.get('filepath')
        self.spike_h = kwargs.get('spike_h', 250)
        self.spike_w = kwargs.get('spike_w', 400)
        self.train = kwargs.get('train', True)

        if self.train:
            self.path = os.path.join(self.rootpath, 'train')
        else:
            self.path = os.path.join(self.rootpath, 'test')
        self.classes = sorted(e.name for e in os.scandir(self.path) if e.is_dir())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []

        for k in self.class_to_idx.keys():
            filelist = os.listdir(os.path.join(self.path, k))
            self.samples.extend([(os.path.join(self.path, k, f), self.class_to_idx[k]) for f in filelist])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        filepath, target = self.samples[index]
        spike_obj = SpikeStream(filepath=filepath, spike_h=self.spike_h, spike_w=self.spike_w, print_dat_detail=False)
        spike = spike_obj.get_spike_matrix(flipud=True).astype(np.float32)
        return spike, target