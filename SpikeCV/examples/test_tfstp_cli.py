# -*- coding: utf-8 -*- 
# @Time : 2022/7/13 22:00 
# @Author : Yajing Zheng
# @File : test_tfstp.py
import os
import torch
import sys
import argparse
from pathlib import Path
from pprint import pprint

from spikecv.spkData.load_dat import data_parameter_dict_cli

# Ensure spikecv can be imported if running from examples
# sys.path.append("..") 
# Better way to handle path for stability
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))

from spikecv.spkData.load_dat import data_parameter_dict, data_parameter_dict_cli, SpikeStream
from spikecv.spkProc.reconstruction.tfstp import TFSTP
from spikecv.visualization.get_video import obtain_reconstruction_video
from spikecv.utils import path

def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser(description="TFSTP Reconstruction Example")
        parser.add_argument("--yaml_file_path", "-yaml", type=str, default="recVidarReal2019/config.yaml", help="Path to spike data")
        parser.add_argument("--dat_file_path", "-dat", type=str, default="recVidarReal2019/classA/car-100kmh.dat", help="Scene path under the dataset directory")
        parser.add_argument("--begin_idx", "-begin", type=int, default=500, help="Begin index of spikes")
        parser.add_argument("--block_len", "-b", type=int, default=1500, help="Number of spike frames to process")
        parser.add_argument("--stp_u0", "-u0",type=float, default=0.15, help="STP parameter u0")
        parser.add_argument("--stp_D", "-D", type=float, default=0.05 * 20, help="STP parameter D")
        parser.add_argument("--stp_F", "-F", type=float, default=0.5 * 20, help="STP parameter F")
        parser.add_argument("--stp_f", "-f", type=float, default=0.15, help="STP parameter f")
        parser.add_argument("--stp_time_unit", "-tu", type=int, default=1, help="STP parameter time unit")
        args = parser.parse_args()

    RESULTS_DIR = Path("results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.yaml_file_path):
        raise FileNotFoundError(f"Data path '{args.yaml_file_path}' does not exist.")

    try:
        paraDict = data_parameter_dict_cli(args.yaml_file_path, args.dat_file_path, "raw")
    except Exception as e:
        print(f"Error occurred: {e}")
        raise e
    pprint(paraDict)

    # initial SpikeStream object for format input data
    vidarSpikes = SpikeStream(**paraDict)
    spikes = vidarSpikes.get_block_spikes(begin_idx=args.begin_idx, block_len=args.block_len)

    device_name = "cpu"  # Default to CPU
    if torch.cuda.is_available():
        device_name = "cuda"
    device = torch.device(device_name)

    stpPara = {
        "u0": args.stp_u0,
        "D": args.stp_D,
        "F": args.stp_F,
        "f": args.stp_f,
        "time_unit": args.stp_time_unit
    }
    
    reconstructor = TFSTP(paraDict.get('spike_h'), paraDict.get('spike_w'), device, stpPara)

    print(f"Running TFSTP reconstruction on {device_name}...")
    recImg = reconstructor.spikes2images_offline(spikes)

    filename_parts = Path(args.dat_file_path).stem
    result_video = (RESULTS_DIR / (filename_parts + '_tfstp.avi')).as_posix()

    obtain_reconstruction_video(recImg, result_video, **paraDict)
    print(f"Reconstruction video saved to: {result_video}")

    return {
        "reconstructed_video_file": result_video,
        "reconstructed_images_shape": recImg.shape,
        "status": "success"
    }

if __name__ == "__main__":
    main()


