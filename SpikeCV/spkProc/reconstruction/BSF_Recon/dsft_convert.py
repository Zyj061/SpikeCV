# -*- coding: utf-8 -*-
# @Time : 2026/07/06
# @File : dsft_convert.py

import torch


def convert_dsft4(dsft, spike):
    """
    Build four DSFT variants used by BSF.

    Parameters
    ----------
    dsft : torch.Tensor
        DSFT tensor of shape (B, T, H, W).
    spike : torch.Tensor
        Binary spike tensor of shape (B, T, H, W).

    Returns
    -------
    dict
        Keys: dsft11, dsft12, dsft21, dsft22.
    """
    b, t_len, _, _ = spike.shape

    dmls1 = -1 * torch.ones(spike.shape, device=spike.device, dtype=torch.float32)
    dmrs1 = -1 * torch.ones(spike.shape, device=spike.device, dtype=torch.float32)

    flag = -2 * torch.ones([b, spike.shape[2], spike.shape[3]], device=spike.device, dtype=torch.float32)
    for ii in range(t_len - 1, -1, -1):
        flag += (spike[:, ii] == 1)
        copy_pad_coord = (flag < 0)
        dmls1[:, ii][copy_pad_coord] = dsft[:, ii][copy_pad_coord]
        if ii < t_len - 1:
            update_coord = (spike[:, ii + 1] == 1) * (~copy_pad_coord)
            dmls1[:, ii][update_coord] = dsft[:, ii + 1][update_coord]
            non_update_coord = (spike[:, ii + 1] != 1) * (~copy_pad_coord)
            dmls1[:, ii][non_update_coord] = dmls1[:, ii + 1][non_update_coord]

    flag = -2 * torch.ones([b, spike.shape[2], spike.shape[3]], device=spike.device, dtype=torch.float32)
    for ii in range(t_len):
        flag += (spike[:, ii] == 1)
        copy_pad_coord = (flag < 0)
        dmrs1[:, ii][copy_pad_coord] = dsft[:, ii][copy_pad_coord]
        if ii > 0:
            update_coord = (spike[:, ii] == 1) * (~copy_pad_coord)
            dmrs1[:, ii][update_coord] = dsft[:, ii - 1][update_coord]
            non_update_coord = (spike[:, ii] != 1) * (~copy_pad_coord)
            dmrs1[:, ii][non_update_coord] = dmrs1[:, ii - 1][non_update_coord]

    return {
        'dsft11': dsft,
        'dsft12': dsft + dmls1,
        'dsft21': dsft + dmrs1,
        'dsft22': dsft + dmls1 + dmrs1,
    }
