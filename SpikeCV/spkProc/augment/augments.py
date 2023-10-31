# Part of the scripts are adapted from torchvision.transforms

import numpy as np
import torch
from PIL import Image
import PIL
import skimage.transform
import numbers
import random
import math
import cv2
from typing import Sequence, Optional, List, Tuple
import warnings

from . import func as F

__all__ = ["Assemble", "RandomHorizontalFlip", "RandomVerticalFlip", "Resize", "SpikeQuant", "RandomResize",
    "CenterCrop", "RandomCrop", "RandomResizedCrop",
    "SpatialPad", "TemporalPad",
    "RandomRotation", "RandomAffine",
    "RandomBlockErasing", "RandomSpikeErasing", "RandomSpikeAdding"
    # TODO "Normalize"
    ]

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


class Assemble:
    '''
    Assembling the spike augmentation
    '''
    def __init__(self, arguments):
        self.arguments = arguments

    def __call__(self, spk_data):
        for a in self.arguments:
            spk_data = a(spk_data)
        return spk_data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for a in self.arguments:
            format_string += '\n'
            format_string += f'    {a}'
        format_string += '\n)'
        return format_string

class RandomHorizontalFlip:
    """Horizontally flip the list of given images randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, spk_data):
        """
        Args:
        The data can be a numpy ndarray or a torch Tensor, in which case it is expected
        to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions


        Returns:
        torch.Tensor or numpy.ndarray: Randomly flipped clip
        """
        if random.random() < self.p:
            if isinstance(spk_data, (np.ndarray, torch.Tensor)):
                return F.spk_fliplr(spk_data)
            else:
                raise TypeError('Expected numpy.ndarray or torch.Tensor' +
                                ' but got {0}'.format(type(spk_data)))
        return spk_data

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomVerticalFlip:
    """Vertically flip the list of given images randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, spk_data):
        """
        Args:
        The data can be a numpy ndarray or a torch Tensor, in which case it is expected
        to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions


        Returns:
        torch.Tensor or numpy.ndarray: Randomly flipped clip
        """
        if random.random() < self.p:
            if isinstance(spk_data, (np.ndarray, torch.Tensor)):
                return F.spk_flipud(spk_data)
            else:
                raise TypeError('Expected numpy.ndarray or torch.Tensor' +
                                ' but got {0}'.format(type(spk_data)))
        return spk_data

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class Resize:
    """Resizes Spike data to the final size

    Args:
    interpolation (str): 
    """

    def __init__(self, size, interpolation='BILINEAR'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, spk_data):
        resized = F.resize(
            spk_data, self.size, interpolation=self.interpolation)
        return resized

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, self.interpolation)

class SpikeQuant:
    '''
    Quantize distorted spike data

    Args:
    mode (str): `floor`, `round`, `ceil`

    '''
    def __init__(self, mode='round'):
        self.mode = mode

    def __call__(self, distorted_data):
        return F.bit_quant(distorted_data, mode=self.mode)

    def __repr__(self):
        return self.__class__.__name__ + f'({self.mode})'

class RandomResize:
    """Resizes spike data to the final size randomly
    """

    def __init__(self, ratio=(3. / 4., 4. / 3.), interpolation='BILINEAR'):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, spk_data):
        scaling_factor = random.uniform(self.ratio[0], self.ratio[1])
        h, w = spk_data.shape[-2], spk_data.shape[-1]
        new_w = int(w * scaling_factor)
        new_h = int(h * scaling_factor)
        new_size = (new_w, new_h)
        resized = F.resize(
            spk_data, new_size, interpolation=self.interpolation)
        return resized
    
    def __repr__(self):
        return self.__class__.__name__ + f'(ratio=[{self.ratio[0]:.2f},{self.ratio[1]:.2f}])'

class CenterCrop:
    """Extract center crop at the same location

    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, spk_data):
        size_h, size_w = self.size
        raw_shape = spk_data.shape
        len_shape = len(raw_shape)
        h, w = spk_data.shape[len_shape - 2], spk_data.shape[len_shape - 1]

        if size_w > w or size_h > h:
            error_msg = (
                'Initial spike data size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial spike data is ({im_w}, {im_h})'.format(
                    im_w=w, im_h=h, w=size_w, h=size_h))
            raise ValueError(error_msg)

        left = int(round((w - size_w) / 2.))
        top = int(round((h - size_h) / 2.))

        cropped = F.crop(spk_data, top, left, size_h, size_w)
        return cropped

    def __repr__(self):
        return self.__class__.__name__ + '(size={})'.format(self.size)

class RandomCrop:
    """Extract random crop at the same location for spike data
        """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, spk_data):
        size_h, size_w = self.size
        raw_shape = spk_data.shape
        len_shape = len(raw_shape)
        h, w = spk_data.shape[len_shape - 2], spk_data.shape[len_shape - 1]

        if size_w > w or size_h > h:
            error_msg = (
                'Initial spike data size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial spike data is ({im_w}, {im_h})'.format(
                    im_w=w, im_h=h, w=size_w, h=size_h))
            raise ValueError(error_msg)

        left = random.randint(0, w - size_w)
        top = random.randint(0, h - size_h)

        cropped = F.crop(spk_data, top, left, size_h, size_w)
        return cropped

    def __repr__(self):
        return self.__class__.__name__ + '(size={})'.format(self.size)

class RandomResizedCrop:
    """Crop the given image to random size and aspect ratio.
        The spike data is expected
        to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

        A crop of random size (default: of 0.08 to 1.0) of the original size and a random
        aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
        is finally resized to given size.

    Args:
        size (int or sequence): expected output size of each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a tuple or list of length 1, it will be interpreted as (size[0], size[0]).
        scale (tuple of float): range of size of the origin size cropped
        ratio (tuple of float): range of aspect ratio of the origin aspect ratio cropped.
        interpolation (str): Desired interpolation.
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='BILINEAR'):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size
        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(spk_data, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (list of PIL Image): spike data to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        raw_shape = spk_data.shape
        len_shape = len(raw_shape)
        height, width = spk_data.shape[len_shape - 2], spk_data.shape[len_shape - 1]
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, spk_data):
        i, j, h, w = self.get_params(spk_data, self.scale, self.ratio)
        return F.resized_crop(spk_data, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(self.interpolation)
        return format_string

class SpatialPad:
    """Pad the given data on all spatial sides with the given "pad" value.
        The spike data is expected
        to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

        Args:
            padding (int or tuple or list): Padding on each border. If a single int is provided this
                is used to pad all borders. If tuple of length 2 is provided this is the padding
                on left/right and top/bottom respectively. If a tuple of length 4 is provided
                this is the padding for the left, top, right and bottom borders respectively.
            fill (int): Pixel fill value for constant fill. Default is 0.
                This value is only used when the padding_mode is constant
            padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
                Default is constant. Mode symmetric is not yet supported for Tensor inputs.

                - constant: pads with a constant value, this value is specified with fill

                - edge: pads with the last value at the edge of the image

                - reflect: pads with reflection of image without repeating the last value on the edge

                    For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                    will result in [3, 2, 1, 2, 3, 4, 3, 2]

                - symmetric: pads with reflection of image repeating the last value on the edge

                    For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                    will result in [2, 1, 1, 2, 3, 4, 4, 3]
        """

    def __init__(self, padding, fill=0, padding_mode="constant"):
        assert(padding_mode in ['constant', 'edge', 'reflect', 'symmetric'])
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, spk_data):
        raw_shape = spk_data.shape
        len_shape = len(raw_shape)
        pad_width = [(0,0) for _ in range(len_shape)]
        if isinstance(self.padding, numbers.Number):
            pad_width[-2] = (self.padding, self.padding)
            pad_width[-1] = (self.padding, self.padding)
        elif isinstance(self.padding, Sequence):
            if len(self.padding) == 4:
                pad_width[-2] = (self.padding[1], self.padding[3])
                pad_width[-1] = (self.padding[0], self.padding[2])
            elif len(self.padding) == 2 and \
                    isinstance(self.padding[0], numbers.Number) and \
                    isinstance(self.padding[1], numbers.Number):
                pad_width[-2] = (self.padding[1], self.padding[1])
                pad_width[-1] = (self.padding[0], self.padding[0])
            else:
                raise ValueError('Unexpected padding value. ', self.padding)
        else:
            raise ValueError('Unexpected padding value. ', self.padding)
        return F.pad(spk_data, pad_width=pad_width, mode=self.padding_mode, value=self.fill)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.padding, self.fill, self.padding_mode)

class TemporalPad:
    """Pad the given data along the temporal axis with the given "pad" value.
            The spike data is expected
            to have [..., T, H, W] shape, where ... means an arbitrary number of leading dimensions

            Args:
                padding (int or tuple or list): Padding on each border. If a single int is provided this
                    is used to pad all borders. If tuple of length 2 is provided this is the padding
                    on left/right.
                fill (int): Pixel fill value for constant fill. Default is 0.
                    This value is only used when the padding_mode is constant
                padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
                    Default is constant. Mode symmetric is not yet supported for Tensor inputs.

                    - constant: pads with a constant value, this value is specified with fill

                    - edge: pads with the last value at the edge of the image

                    - reflect: pads with reflection of image without repeating the last value on the edge

                        For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                        will result in [3, 2, 1, 2, 3, 4, 3, 2]

                    - symmetric: pads with reflection of image repeating the last value on the edge

                        For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                        will result in [2, 1, 1, 2, 3, 4, 4, 3]
            """

    def __init__(self, padding, fill=0, padding_mode="constant"):
        assert (padding_mode in ['constant', 'edge', 'reflect', 'symmetric'])
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, spk_data):
        raw_shape = spk_data.shape
        len_shape = len(raw_shape)
        pad_width = [(0, 0) for _ in range(len_shape)]
        if isinstance(self.padding, numbers.Number):
            pad_width[-3] = (self.padding, self.padding)
        elif isinstance(self.padding, Sequence):
            if len(self.padding) == 2 and \
                    isinstance(self.padding[0], numbers.Number) and \
                    isinstance(self.padding[1], numbers.Number):
                pad_width[-3] = self.padding
            else:
                raise ValueError('Unexpected padding value. ', self.padding)
        else:
            raise ValueError('Unexpected padding value. ', self.padding)
        return F.pad(spk_data, pad_width=pad_width, mode=self.padding_mode, value=self.fill)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'. \
            format(self.padding, self.fill, self.padding_mode)

class RandomRotation:
    """Rotate the spike data by angle.
        The spike data is expected
        to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

        Args:
            degrees (sequence or float or int): Range of degrees to select from.
                If degrees is a number instead of sequence like (min, max), the range of degrees
                will be (-degrees, +degrees).
            resample (str): If input is Tensor, only ``NEAREST`` and ``BILINEAR`` are supported.
            expand (bool, optional): Optional expansion flag.
                If true, expands the output to make it large enough to hold the entire rotated data.
                If false or omitted, make the output data the same size as the input data.
                Note that the expand flag assumes rotation around the center and no translation.
            center (list or tuple, optional): Optional center of rotation, (x, y). Origin is the upper left corner.
                Default is the center of the data.
            fill (n-tuple or int or float): Pixel fill value for area outside the rotated
                data. If int or float, the value is used for all bands respectively.
                Defaults to 0 for all bands. This option is only available for Pillow>=5.2.0.
                This option is not supported for Tensor input. Fill value for the area outside the transform in the output
                data is always 0.
        """

    def __init__(self, degrees, resample='NEAREST', expand=False, center=None, fill=None):
        super().__init__()
        self.degrees = F.setup_angle(degrees, name="degrees", req_sizes=(2,))
        assert(resample in ['NEAREST', 'BILINEAR'])
        if center is not None:
            F.check_sequence_input(center, "center", req_sizes=(2,))

        self.center = center

        self.resample = resample
        self.expand = expand
        self.fill = fill

    def __call__(self, spk_data):
        angle = float(torch.empty(1).uniform_(float(self.degrees[0]), float(self.degrees[1])).item())
        return F.rotate(spk_data, angle, _str_to_pil_interpolation[self.resample], self.expand, self.center, self.fill)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        if self.fill is not None:
            format_string += ', fill={0}'.format(self.fill)
        format_string += ')'
        return format_string

class RandomAffine:
    """Random affine transformation of the spike data keeping center invariant.
        The spike data is expected
        to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

        Args:
            degrees (sequence or float or int): Range of degrees to select from.
                If degrees is a number instead of sequence like (min, max), the range of degrees
                will be (-degrees, +degrees). Set to 0 to deactivate rotations.
            translate (tuple, optional): tuple of maximum absolute fraction for horizontal
                and vertical translations. For example translate=(a, b), then horizontal shift
                is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
                randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
            scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
                randomly sampled from the range a <= scale <= b. Will keep original scale by default.
            shear (sequence or float or int, optional): Range of degrees to select from.
                If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
                will be applied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
                range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
                a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
                Will not apply shear by default.
            resample (int, optional): only ``NEAREST`` and ``BILINEAR`` are supported.

        """

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample='NEAREST'):
        super().__init__()
        self.degrees = F.setup_angle(degrees, name="degrees", req_sizes=(2,))

        if translate is not None:
            F.check_sequence_input(translate, "translate", req_sizes=(2,))
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            F.check_sequence_input(scale, "scale", req_sizes=(2,))
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            self.shear = F.setup_angle(shear, name="shear", req_sizes=(2, 4))
        else:
            self.shear = shear

        self.resample = resample

    @staticmethod
    def get_params(
            degrees: List[float],
            translate: Optional[List[float]],
            scale_ranges: Optional[List[float]],
            shears: Optional[List[float]],
            img_size: List[int]
    ) -> Tuple[float, Tuple[int, int], float, Tuple[float, float]]:
        """Get parameters for affine transformation

        Returns:
            params to be passed to the affine transformation
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear

    def __call__(self, spk_data):
        raw_shape = spk_data.shape
        len_shape = len(raw_shape)
        size = (spk_data.shape[len_shape - 2], spk_data.shape[len_shape - 1])

        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, size)
        return F.affine(spk_data, *ret, resample=_str_to_pil_interpolation[self.resample], fillcolor=None)

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        s += ', resample={resample}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = d['resample']
        return s.format(name=self.__class__.__name__, **d)

class RandomBlockErasing:
    """ Randomly selects a rectangle region in the spike data and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

        Args:
             p: probability that the random erasing operation will be performed.
             scale: range of proportion of erased area against input data.
             ratio: range of aspect ratio of erased area.
             value: erasing value. Default is 0.
    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        assert(value in [0,1,0.0, 1.0])
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1")
        if p < 0 or p > 1:
            raise ValueError("Random erasing probability should be between 0 and 1")

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    @staticmethod
    def get_params(spk_data, scale: Tuple[float, float], ratio: Tuple[float, float]):
        raw_shape = spk_data.shape
        len_shape = len(raw_shape)
        spk_h, spk_w = spk_data.shape[len_shape - 2], spk_data.shape[len_shape - 1]
        area = spk_h * spk_w

        for _ in range(10):
            erase_area = area * random.uniform(scale[0], scale[1])
            aspect_ratio = random.uniform(ratio[0], ratio[1])

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < spk_h and w < spk_w):
                continue

            i = torch.randint(0, spk_h - h + 1, size=(1,)).item()
            j = torch.randint(0, spk_w - w + 1, size=(1,)).item()
            return i, j, h, w

        # Return original data
        return 0, 0, spk_h, spk_w

    def __call__(self, spk_data):
        if torch.rand(1) < self.p:
            x, y, h, w = self.get_params(spk_data, scale=self.scale, ratio=self.ratio)
            return F.block_erase(spk_data, x, y, h, w, self.value)
        return spk_data

    def __repr__(self):
        s = '{name}(p={p}, scale={scale}, ratio={ratio}, value={value}'
        s += ')'
        d = dict(self.__dict__)
        return s.format(name=self.__class__.__name__, **d)

class RandomSpikeErasing:
    """ Randomly erase some spikes to 0 in the spike data

        Args:
             p: probability that the random erasing operation will be performed.
    """
    def __init__(self, p=0.5, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("Random erasing probability should be between 0 and 1")

        self.p = p
        self.inplace = inplace

    def __call__(self, spk_data):
        return F.random_erase(spk_data, p=self.p, value=0, inplace=self.inplace)

    def __repr__(self):
        s = '{name}(p={p}, inplace={inplace})'.format(name=self.__class__.__name__, p=self.p, inplace=self.inplace)
        return s

class RandomSpikeAdding:
    """ Randomly add some spikes in the spike data

            Args:
                 p: probability that the random adding operation will be performed.
        """

    def __init__(self, p=0.5, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("Random adding probability should be between 0 and 1")

        self.p = p
        self.inplace = inplace

    def __call__(self, spk_data):
        return F.random_add(spk_data, p=self.p, inplace=self.inplace)

    def __repr__(self):
        s = '{name}(p={p}, inplace={inplace})'.format(name=self.__class__.__name__, p=self.p, inplace=self.inplace)
        return s




