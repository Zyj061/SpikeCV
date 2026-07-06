# -*- coding: utf-8 -*-
# @Time : 2026/07/06
# @File : bsf_recon.py

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from spikecv.spkProc.reconstruction.BSF_Recon.bsf_model import BSF
from spikecv.spkProc.reconstruction.BSF_Recon.dsft import DSFT
from spikecv.spkProc.reconstruction.BSF_Recon.dsft_convert import convert_dsft4


def default_weight_path():
    """Return the default path for BSF pretrained weights."""
    return Path(__file__).resolve().parent / "pretrained" / "bsf.pth"


def resolve_weight_path(weight_path=None):
    """Resolve and validate the BSF weight file path."""
    if weight_path is None:
        weight_path = default_weight_path()
    weight_path = Path(weight_path)
    if not weight_path.exists():
        download_hint = weight_path.parent / "download_link.txt"
        raise FileNotFoundError(
            f"BSF pretrained weights not found at '{weight_path}'. "
            f"Please download weights and place them there. "
            f"See '{download_hint}' for instructions."
        )
    return weight_path


def load_bsf_weights(model, weight_path, device):
    """Load BSF checkpoint, stripping DataParallel prefixes if needed."""
    checkpoint = torch.load(weight_path, map_location=device)
    if any(key.startswith('module.') for key in checkpoint.keys()):
        state_dict = OrderedDict()
        for key, value in checkpoint.items():
            state_dict[key[7:]] = value
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def _pad_to_multiple_of_16(raw_h, raw_w):
    """Compute padded dimensions that are multiples of 16."""
    pad_h = ((raw_h + 15) // 16) * 16 - raw_h
    pad_w = ((raw_w + 15) // 16) * 16 - raw_w
    return pad_h, pad_w


class BSFRecon:
    """
    BSF-based spike camera image reconstruction.

    Uses a sliding temporal window (default 61 frames) with DSFT representation
    and pretrained BSF network weights.

    The spatial dimensions are padded to multiples of 16 before processing
    (matching the original BSF training/inference setup), then cropped back
    to the original size in the output.
    """

    def __init__(
        self,
        spike_h,
        spike_w,
        device,
        weight_path=None,
        window_size=61,
        max_search_half_window=20,
        step=1,
        gamma=2.2,
    ):
        self.spike_h = spike_h
        self.spike_w = spike_w
        self.device = torch.device(device)
        self.window_size = window_size
        self.max_search_half_window = max_search_half_window
        self.step = step
        self.gamma = gamma

        # Pad to multiples of 16 for alignment compatibility
        self.pad_h, self.pad_w = _pad_to_multiple_of_16(spike_h, spike_w)
        self.padded_h = spike_h + self.pad_h
        self.padded_w = spike_w + self.pad_w

        weight_path = resolve_weight_path(weight_path)
        self.model = BSF().to(self.device)
        load_bsf_weights(self.model, weight_path, self.device)
        self.dsft_calc = DSFT(self.padded_h, self.padded_w, self.device)

    def spikes2images(self, spikes, show_progress=True):
        """
        Reconstruct visible images from a spike stream.

        Parameters
        ----------
        spikes : np.ndarray
            Spike stream of shape (T, spike_h, spike_w).
        show_progress : bool
            Whether to show a tqdm progress bar.

        Returns
        -------
        np.ndarray
            Reconstructed grayscale frames of shape (N, spike_h, spike_w), uint8.
        """
        spikes = np.asarray(spikes)
        if spikes.ndim != 3:
            raise ValueError(f"Expected spikes shape (T, H, W), got {spikes.shape}")

        T, raw_h, raw_w = spikes.shape
        if raw_h != self.spike_h or raw_w != self.spike_w:
            raise ValueError(
                f"Spike dimensions ({raw_h}, {raw_w}) do not match "
                f"initialized dimensions ({self.spike_h}, {self.spike_w})"
            )

        # Pad to multiples of 16 (same as original BSF inference)
        spikes_padded = np.pad(
            spikes,
            ((0, 0), (0, self.pad_h), (0, self.pad_w)),
            mode='constant',
            constant_values=0
        )

        # DSFT on padded data
        dsft_full = self.dsft_calc.spikes2images(
            spikes_padded, max_search_half_window=self.max_search_half_window
        )

        # Temporal alignment
        start = self.max_search_half_window
        end = T - self.max_search_half_window
        spikes_aligned = spikes_padded[start:end]
        dsft_aligned = dsft_full

        T_valid = len(spikes_aligned)
        half_win = self.window_size // 2
        center_start = half_win
        center_end = T_valid - half_win

        if center_end <= center_start:
            min_needed = 2 * self.max_search_half_window + 2 * half_win + 1
            raise ValueError(
                f"Spike stream too short ({T} frames). "
                f"Need at least {min_needed} frames for window_size={self.window_size} "
                f"and max_search_half_window={self.max_search_half_window}."
            )

        centers = range(center_start, center_end, self.step)
        iterator = tqdm(centers, desc="BSF reconstruction") if show_progress else centers
        frame_list = []

        with torch.no_grad():
            for center in iterator:
                t_start = center - half_win
                t_end = center + half_win + 1

                spike_block = spikes_aligned[t_start:t_end]
                dsft_block = dsft_aligned[t_start:t_end]

                spike_t = torch.from_numpy(spike_block).float().unsqueeze(0).to(self.device)
                dsft_t = torch.from_numpy(dsft_block).float().unsqueeze(0).to(self.device)
                dsft_dict = convert_dsft4(spike=spike_t, dsft=dsft_t)
                rec = self.model({'spikes': spike_t, 'dsft_dict': dsft_dict})

                img = rec.squeeze().cpu().numpy()
                img = np.clip(img, 0, 1)

                # Crop back to original spatial size (remove padding)
                img = img[:self.spike_h, :self.spike_w]

                # Gamma correction
                if self.gamma is not None and self.gamma > 0:
                    img = img ** (1.0 / self.gamma) * 255.0
                else:
                    img = img * 255.0

                frame_list.append(img.astype(np.uint8))

        return np.stack(frame_list, axis=0)
