import os.path
import torchvision.transforms as transforms
import numpy as np
from spkProc.reconstruction.NeurImg_HDR.data.base_dataset import BaseDataset
from spkProc.reconstruction.NeurImg_HDR.data.image_folder import make_dataset
from PIL import Image
import cv2

class TestDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_ldr = os.path.join(opt.dataroot, 'LDR/' + opt.phase)
        self.dir_im = os.path.join(opt.dataroot, 'IM/' + opt.phase)

        self.ldr_paths = sorted(make_dataset(self.dir_ldr, opt.max_dataset_size))
        self.im_paths = sorted(make_dataset(self.dir_im, opt.max_dataset_size))
        self.ldr_size = len(self.ldr_paths)
        self.im_size = len(self.im_paths)
        self.im_type = opt.im_type
        self.pad_flag = True

    def __getitem__(self, index):  
        # ----------- Load Intensity Map -------------
        index_im = index % self.im_size
        im_path = self.im_paths[index_im]
        
        if self.im_type == 'event':
            im = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
            im = (im / 255.0).astype(np.float32)
        elif self.im_type == 'spike':
            if im_path[-3:] == 'npy':
                im = np.load(im_path).astype(np.float32)

        orig_h, orig_w = im.shape[:2]
        im_h_pad, im_w_pad = 0, 0
        if (orig_w % 32 == 0) and (orig_h % 32 == 0):
            self.pad_flag == False
        else:
            self.pad_flag == True

        if orig_h & 1 != 0:
            im = np.pad(im, ((0, 1), (0, 0)), mode='reflect') 
        if orig_w & 1 != 0:
            im = np.pad(im, ((0, 0), (0, 1)), mode='reflect') 
        im_h, im_w = im.shape[:2]
        if self.pad_flag:
            if im_h % 32 == 0:
                im_h_pad = 0
            else:
                im_h_pad = int(((im_h//32 + 1)*32-im_h)/2)
            if im_w % 32 == 0:
                im_w_pad = 0
            else:
                im_w_pad = int(((im_w//32 + 1)*32-im_w)/2)
            im_crop = np.pad(im, ((im_h_pad, im_h_pad), (im_w_pad, im_w_pad)), mode='reflect') 
        else:
            im_crop = self.__scale(im, self.opt.resolution//2)
            im_crop = cv2.resize(im_crop, (256, 256), interpolation=cv2.INTER_LINEAR)
            
        im_crop = (im_crop * 2.0 - 1.0).astype(np.float32)
        im_tensor = transforms.ToTensor()(im_crop)
        
        # ----------- Load LDR Image -------------
        index_ldr = index % self.ldr_size
        ldr_path = self.ldr_paths[index_ldr]
        ldr_img = Image.open(ldr_path).convert('RGB')       
        ldr_img = np.array(ldr_img).astype(np.float32)

        if self.pad_flag:
            ldr_h_pad = im_h_pad * self.opt.up_scale
            ldr_w_pad = im_w_pad * self.opt.up_scale
            ldr_crop = np.pad(ldr_img, ((ldr_h_pad, ldr_h_pad), (ldr_w_pad, ldr_w_pad), (0, 0)), mode='reflect')
        else:
            ldr_crop = ldr_img

        ldr_crop = ldr_crop / 255.0
        # ldr_crop = ((ldr_crop)**2.2)
        ldr_norm = (ldr_crop*2.0-1.0).astype(np.float32)
        ldr_rgb = transforms.ToTensor()(ldr_norm)

        ldr_yuv = cv2.cvtColor(ldr_crop, cv2.COLOR_RGB2YUV)
        ldr_y = ldr_yuv[:,:,0] # [0.0, 1.0]
        ldr_u = ldr_yuv[:,:,1] # [-0.5, 0.5]
        ldr_v = ldr_yuv[:,:,2] # [-0.5, 0.5]
        ldr_y = ldr_y * 2.0 - 1.0
        data_ldr_y = transforms.ToTensor()(ldr_y.astype(np.float32))
        
        data_ldr_u = transforms.ToTensor()(ldr_u)
        data_ldr_v = transforms.ToTensor()(ldr_v)
        
        return {'input_ldr_y': data_ldr_y, 'input_ldr_u': data_ldr_u, 'input_ldr_v': data_ldr_v, 'input_ldr_rgb': ldr_rgb, 'input_im': im_tensor, 'paths': ldr_path,
                'im_h_pad': im_h_pad, 'im_w_pad': im_w_pad}


    def __len__(self):
        return self.ldr_size

    def __scale(self, img, target_shorter):
        oh, ow = img.shape[:2]
        if ow >= oh:
            h = target_shorter
            w = int(target_shorter * ow / oh)
        else:
            w = target_shorter
            h = int(target_shorter * oh / ow)
        img_scaled = cv2.resize(img, (w, h), cv2.INTER_LINEAR)

        return img_scaled
