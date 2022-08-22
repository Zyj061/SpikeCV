# -*- coding: utf-8 -*-
# @Time : 2022/8/5 17:18
# @Author Homepage : https://github.com/DingJianhao
# File : data_transform.py

import numpy as np
import torch
from typing import Optional
from torch.types import _dtype

__all__ = ["ToTorchTensor", "ToNPYArray"]


class ToTorchTensor:
    """

    Convert a ``numpy.ndarray`` to ``torch.tensor``.

    """
    def __init__(self, type: torch.dtype = torch.FloatTensor):
        self.type = type

    def __call__(self, spk_data):
        """
        Args:
            numpy.ndarray: Spike data to be converted to ``torch.tensor``.

        Returns:
            Tensor: Converted spike data.
        """
        if isinstance(spk_data, np.ndarray):
            return torch.from_numpy(spk_data).type(self.type)
        elif isinstance(spk_data, torch.Tensor):
            return spk_data.type(self.type)
        else:
            raise TypeError('Expected torch.Tensor' +
                            ' but got {0}'.format(type(spk_data)))
        return spk_data

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToNPYArray:
    """

    Convert a ``torch.tensor`` to ``numpy.ndarray``.

    """
    def __init__(self, type: np.dtype = np.float):
        self.type = type

    def __call__(self, spk_data):
        """
        Args:
            torch.tensor: Spike data to be converted to ``numpy.ndarray``.

        Returns:
            Ndarray: Converted spike data.
        """
        if isinstance(spk_data, np.ndarray):
            return spk_data.astype(self.type)
        elif isinstance(spk_data, torch.Tensor):
            return spk_data.cpu().detach().numpy().astype(self.type)
        else:
            raise TypeError('Expected torch.Tensor' +
                            ' but got {0}'.format(type(spk_data)))
        return spk_data

    def __repr__(self):
        return self.__class__.__name__ + '()'