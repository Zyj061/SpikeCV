# -*- coding: utf-8 -*-
# @Time : 2026/07/06
# @File : test_bsf_cli.py

import argparse
import os
from pathlib import Path
from pprint import pprint

import torch

from spikecv.spkData.load_dat import SpikeStream, data_parameter_dict_cli
from spikecv.spkProc.reconstruction.BSF_Recon.bsf_recon import BSFRecon, default_weight_path
from spikecv.visualization.get_video import obtain_reconstruction_video


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="BSF Reconstruction Example")
        parser.add_argument(
            "--yaml_file_path", "-yaml", type=str,
            default="recVidarReal2019/config.yaml",
            help="Path to dataset config yaml",
        )
        parser.add_argument(
            "--dat_file_path", "-dat", type=str,
            default="recVidarReal2019/classA/car-100kmh.dat",
            help="Path to spike .dat file",
        )
        parser.add_argument("--begin_idx", "-begin", type=int, default=500, help="Begin index of spikes")
        parser.add_argument("--block_len", "-b", type=int, default=300, help="Number of spike frames to process")
        parser.add_argument(
            "--weight_path", "-w", type=str, default=None,
            help="Path to BSF pretrained weights (.pth)",
        )
        parser.add_argument("--step", type=int, default=3, help="Temporal step between reconstructed frames")
        parser.add_argument("--window_size", type=int, default=61, help="Sliding window size")
        parser.add_argument(
            "--max_half_win", type=int, default=20,
            help="Half window for DSFT max search",
        )
        parser.add_argument("--gamma", type=float, default=2.2, help="Gamma correction factor")
        args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.yaml_file_path):
        raise FileNotFoundError(f"Config path '{args.yaml_file_path}' does not exist.")

    para_dict = data_parameter_dict_cli(args.yaml_file_path, args.dat_file_path, "raw")
    pprint(para_dict)

    vidar_spikes = SpikeStream(**para_dict)
    spikes = vidar_spikes.get_block_spikes(begin_idx=args.begin_idx, block_len=args.block_len)

    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    weight_path = args.weight_path or str(default_weight_path())

    reconstructor = BSFRecon(
        spike_h=para_dict["spike_h"],
        spike_w=para_dict["spike_w"],
        device=device,
        weight_path=weight_path,
        window_size=args.window_size,
        max_search_half_window=args.max_half_win,
        step=args.step,
        gamma=args.gamma,
    )

    print(f"Running BSF reconstruction on {device_name}...")
    rec_img = reconstructor.spikes2images(spikes)

    base_stem = Path(args.dat_file_path).stem + "_bsf"
    result_video_path = results_dir / f"{base_stem}.avi"
    counter = 1
    while result_video_path.exists():
        result_video_path = results_dir / f"{base_stem}({counter}).avi"
        counter += 1
    result_video = result_video_path.as_posix()

    obtain_reconstruction_video(rec_img, result_video, **para_dict)
    print(f"Reconstruction video saved to: {result_video}")

    return {
        "reconstructed_video_file": result_video,
        "reconstructed_images_shape": rec_img.shape,
        "weight_path": weight_path,
        "status": "success",
    }


if __name__ == "__main__":
    main()
