import torch
import torchvision.transforms.functional as F_tensor
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, List, Optional, Sequence
import numbers
import SpikeCV.spkData.data_transform as transform


_interpolations = ['NEAREST', 'BILINEAR', 'BICUBIC', 'LANCZOS']

_str_to_pil_interpolation = {    
    'NEAREST': Image.NEAREST,    
    'BILINEAR': Image.BILINEAR,    
    'BICUBIC': Image.BICUBIC,    
    'LANCZOS': Image.LANCZOS,
    }

_str_to_cv2_interpolation = {    
    'NEAREST': cv2.INTER_NEAREST,    
    'BILINEAR': cv2.INTER_LINEAR,    
    'BICUBIC': cv2.INTER_CUBIC,    
    'LANCZOS': cv2.INTER_LANCZOS4,
    }

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size

def spk_fliplr(spk_data):
    # [..., H, W]
    len_shape = len(spk_data.shape)
    if isinstance(spk_data, np.ndarray):
        return np.flip(spk_data, axis=len_shape-1)
    elif isinstance(spk_data, torch.Tensor):
        return F_tensor.hflip(spk_data)

def spk_flipud(spk_data):
    # [..., H, W]
    len_shape = len(spk_data.shape)
    if isinstance(spk_data, np.ndarray):
        return np.flip(spk_data, axis=len_shape-2)
    elif isinstance(spk_data, torch.Tensor):
        return F_tensor.vflip(spk_data)

def resize(spk_data, size: List[int], interpolation: str = 'BILINEAR'):
    raw_shape = spk_data.shape
    len_shape = len(raw_shape)
    h, w = spk_data.shape[len_shape-2], spk_data.shape[len_shape-1]

    if isinstance(size, numbers.Number):
        if (w <= h and w == size) or (h <= w and h == size):
            return spk_data
        
        if w < h:
            new_w = size
            new_h = int(size * h / w)
        else:
            new_h = size
            new_w = int(size * w / h)

        size = (new_w, new_h)
    else:
        size = size[1], size[0]

    if isinstance(spk_data, np.ndarray):
        spk_data = spk_data.reshape(-1, h, w).transpose((1,2,0))
        spk_data = cv2.resize(spk_data, size, interpolation=_str_to_cv2_interpolation[interpolation])
        return spk_data.transpose((2, 0, 1)).reshape((*raw_shape[:-2], size[1], size[0]))
    elif isinstance(spk_data, torch.Tensor):
        return F_tensor.resize(spk_data, (size[1], size[0]), interpolation=_str_to_pil_interpolation[interpolation])

def bit_quant(distorted_data, mode):
    if isinstance(distorted_data, np.ndarray):
        if mode.lower() == 'round':
            return np.clip(np.round(distorted_data), 0, 1)
        elif mode.lower() == 'floor':
            return np.clip(np.floor(distorted_data), 0, 1)
        elif mode.lower() == 'ceil':
            return np.clip(np.ceil(distorted_data), 0, 1)
    elif isinstance(distorted_data, torch.Tensor):
        if mode.lower() == 'round':
            return torch.clip(torch.round(distorted_data), 0, 1)
        elif mode.lower() == 'floor':
            return torch.clip(torch.floor(distorted_data), 0, 1)
        elif mode.lower() == 'ceil':
            return torch.clip(torch.ceil(distorted_data), 0, 1)

def crop(spk_data, top, left, height, width):
    if isinstance(spk_data, torch.Tensor):
        cropped = F_tensor.crop(spk_data, top, left, height, width)
    elif isinstance(spk_data, np.ndarray):
        cropped = spk_data[..., top:top + height, left:left+width]
    return cropped

def resized_crop(spk_data, top: int, left: int, height: int, width: int, size: List[int],
                 interpolation: str = 'BILINEAR'):
    spk_data = crop(spk_data, top, left, height, width)
    spk_data = resize(spk_data, size, interpolation)
    return spk_data

def pad(spk_data, pad_width, mode='constant', value=0):
    tensor_flag = False
    if isinstance(spk_data, torch.Tensor):
        tensor_flag = True
        spk_data = transform.ToNPYArray()(spk_data)
    if mode == 'constant':
        spk_data = np.pad(spk_data, pad_width, mode=mode, constant_values=value)
    else:
        spk_data = np.pad(spk_data, pad_width, mode=mode)

    if tensor_flag:
        spk_data = transform.ToTorchTensor()(spk_data) # TODO
    return spk_data

def rotate(spk_data, angle: float, resample: int = 0, expand: bool = False,
        center: Optional[List[int]] = None, fill: Optional[int] = None):
    ndarray_flag = False
    if isinstance(spk_data, np.ndarray):
        ndarray_flag = True
        spk_data = transform.ToTorchTensor()(spk_data)

    spk_data = F_tensor.rotate(spk_data, angle=angle,
                    resample=resample, expand=expand, center=center, fill=fill)

    if ndarray_flag:
        spk_data = transform.ToNPYArray()(spk_data)

    return spk_data

def affine(spk_data, angle: float, translate: List[int], scale: float, shear: List[float],
        resample: int = 0, fillcolor: Optional[int] = None):
    ndarray_flag = False
    if isinstance(spk_data, np.ndarray):
        ndarray_flag = True
        spk_data = transform.ToTorchTensor()(spk_data)

    spk_data = F_tensor.affine(spk_data, angle=angle, translate=translate, scale=scale, shear=shear,
                               resample=resample, fillcolor=fillcolor)

    if ndarray_flag:
        spk_data = transform.ToNPYArray()(spk_data)

    return spk_data

def check_sequence_input(x, name, req_sizes):
    # code from torchvision.transforms
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError("{} should be a sequence of length {}.".format(name, msg))
    if len(x) not in req_sizes:
        raise ValueError("{} should be sequence of length {}.".format(name, msg))

def setup_angle(x, name, req_sizes=(2, )):
    # code from torchvision.transforms
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError("If {} is a single number, it must be positive.".format(name))
        x = [-x, x]
    else:
        check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]

def block_erase(spk_data, i: int, j: int, h: int, w: int, value, inplace=False):
    if isinstance(spk_data, np.ndarray):
        if inplace:
            spk_data[..., i:i + h, j:j + w] = value
            return spk_data
        else:
            x = spk_data.copy()
            x[..., i:i + h, j:j + w] = value
            return x
    elif isinstance(spk_data, torch.Tensor):
        if inplace:
            spk_data[..., i:i + h, j:j + w] = value
            return spk_data
        else:
            x = spk_data.clone()
            x[..., i:i + h, j:j + w] = value
            return x

def random_erase(spk_data, p, value=0, inplace=False):
    if isinstance(spk_data, np.ndarray):
        nonzero = np.array(spk_data.nonzero())
        n_nonzero = nonzero.shape[1]
        chosed = nonzero[:, np.random.choice(n_nonzero, int(p * n_nonzero))]
        idx_chosed = tuple([chosed[i] for i in range(nonzero.shape[0])])
        if inplace:
            spk_data[idx_chosed] = value
            return spk_data
        else:
            x = spk_data.copy()
            x[idx_chosed] = value
            return x
    elif isinstance(spk_data, torch.Tensor):
        nonzero = torch.nonzero(spk_data)
        n_nonzero = nonzero.shape[0]
        chosed = nonzero[np.random.choice(n_nonzero, int(p * n_nonzero)),:]
        idx_chosed = tuple([chosed[:,i] for i in range(nonzero.shape[1])])
        if inplace:
            spk_data[idx_chosed] = value
            return spk_data
        else:
            x = spk_data.clone()
            x[idx_chosed] = value
            return x
    return spk_data

def random_add(spk_data, p, value=1.0, inplace=False):
    inv_spk_data = 1.0 - spk_data
    if isinstance(spk_data, np.ndarray):
        nonzero = np.array(inv_spk_data.nonzero())
        n_nonzero = nonzero.shape[1]
        chosed = nonzero[:, np.random.choice(n_nonzero, int(p * n_nonzero))]
        idx_chosed = tuple([chosed[i] for i in range(nonzero.shape[0])])
        if inplace:
            spk_data[idx_chosed] = value
            return spk_data
        else:
            x = spk_data.copy()
            x[idx_chosed] = value
            return x
    elif isinstance(spk_data, torch.Tensor):
        nonzero = torch.nonzero(inv_spk_data)
        n_nonzero = nonzero.shape[0]
        chosed = nonzero[np.random.choice(n_nonzero, int(p * n_nonzero)),:]
        idx_chosed = tuple([chosed[:,i] for i in range(nonzero.shape[1])])
        if inplace:
            spk_data[idx_chosed] = value
            return spk_data
        else:
            x = spk_data.clone()
            x[idx_chosed] = value
            return x
    return spk_data