import random
import numpy as np
import cv2 
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)



# -----------get pair-wise transform----------------
def get_pairwise_transform(opt, params=None, convert=True):
    transform_list = []

    if 'resize' in opt.preprocess:
        osize = (opt.load_size, opt.load_size)
        transform_list.append(transforms.Lambda(lambda img: cv2.resize(img, osize)))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, opt.crop_size, isTrain=opt.isTrain)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_22(img, base=4)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.Lambda(lambda img: __random_flip(img)))

    if convert:
        transform_list.append(transforms.ToTensor())
        
    return transforms.Compose(transform_list)


def __random_flip(img, prob=0.5):
    if(random.random() > prob):
        return cv2.flip(img, 1)
    return img

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)

def __make_power_22(img, base):
    ow, oh = img.shape[0], img.shape[1]
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return cv2.resize(img, (w, h))


def __scale_width(img, scale_range, method=cv2.INTER_LINEAR):
    range_low, range_high = scale_range[0], scale_range[-1]
    target_shorter = random.randint(range_low, range_high)
        
    oh, ow = img.shape[:2]
    if (min(oh, ow) == target_shorter):
        return img
    if ow >= oh:
        h = target_shorter
        w = int(target_shorter * ow / oh)
    else:
        w = target_shorter
        h = int(target_shorter * oh / ow)
        
    img_scaled = cv2.resize(img, (w, h), method)
    return img_scaled


def __crop(img, crop_size, isTrain=True):
    oh, ow = img.shape[:2]
    if isTrain:
        low_x = random.randint(0, oh-crop_size[0])
        low_y = random.randint(0, ow-crop_size[1])
    else:
        low_x = int((oh-crop_size[0]) / 2)
        low_y = int((ow-crop_size[1]) / 2)
    crop_img = img[low_x:low_x+crop_size[0], low_y:low_y+crop_size[1], :]
    return crop_img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
