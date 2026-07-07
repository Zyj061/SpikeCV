# -*- coding: utf-8 -*-
# @Time : 2026/07/06
# @File : dsft.py

import numpy as np
import torch


class DSFT:
    """Compute DSFT (inter-spike interval) representation from spike streams."""

    def __init__(self, spike_h, spike_w, device):
        self.spike_h = spike_h
        self.spike_w = spike_w
        self.device = device

    def spikes2images(self, spikes, max_search_half_window=20):
        """
        Convert spike stream to DSFT images.

        Parameters
        ----------
        spikes : np.ndarray
            Spike stream of shape (T, H, W).
        max_search_half_window : int
            Half window size for temporal search on each side.

        Returns
        -------
        np.ndarray
            DSFT images of shape (T - 2 * max_search_half_window, H, W), uint8.
        """
        t_len = spikes.shape[0]
        t_im = t_len - 2 * max_search_half_window
        if t_im < 0:
            raise ValueError(
                f"Spike stream length {t_len} is shorter than "
                f"2 * max_search_half_window ({2 * max_search_half_window})."
            )

        spikes_t = torch.from_numpy(spikes).to(self.device).float()

        pre_idx = -1 * torch.ones([t_len, self.spike_h, self.spike_w], device=self.device)
        cur_idx = -1 * torch.ones([t_len, self.spike_h, self.spike_w], device=self.device)

        for ii in range(t_len):
            if ii > 0:
                pre_idx[ii] = cur_idx[ii - 1]
                cur_idx[ii] = cur_idx[ii - 1]
            cur_spk = spikes_t[ii]
            cur_idx[ii][cur_spk == 1] = ii

        diff = cur_idx - pre_idx
        interval = -1 * torch.ones([t_len, self.spike_h, self.spike_w], device=self.device)
        for ii in range(t_len - 1, -1, -1):
            interval[ii][diff[ii] != 0] = diff[ii][diff[ii] != 0]
            if ii < t_len - 1:
                interval[ii][diff[ii] == 0] = interval[ii + 1][diff[ii] == 0]

        interval[interval == -1] = 255
        interval[pre_idx == -1] = 255
        interval = torch.clip(interval, 0, 255)

        return interval[max_search_half_window:-max_search_half_window].cpu().numpy().copy().astype(np.uint8)
