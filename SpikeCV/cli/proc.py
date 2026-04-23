import typer


app = typer.Typer(name="proc", 
                  help="Commands related to spkProc in SpikeCV.", 
                  no_args_is_help=True)


@app.command(name="track", 
             help="Run the tracking module of SpikeCV. Using SNNTracker as default.", 
             short_help="Run tracking module",
             no_args_is_help=True)
def track(
    algorithm: str = typer.Option("snn_tracker", "--algorithm", "-a", help="Tracking algorithm to use. Currently only 'snn_tracker' is supported."),
    scene_idx: int = typer.Option(0, "--scene-idx", "-s", 
                                  help="Index of the test scene. 0: spike59, 1: rotTrans, 2: cplCam, 3: cpl1, 4: ball, 5: badminton, 6: pingpong"),
    attention_size: int = typer.Option(15, "--attention-size", "-attn_size", help="Size of attention window"),
    data_path: str = typer.Option("motVidarReal2020/", "--data-path", "-d", help="Relative Path to dataset root"),
    metrics: bool = typer.Option(False, "--metrics", "-m", help="Enable quantitative metrics (requires GT)"),
    agent_used: bool = typer.Option(False, "--agent-used", "-agent", help="Whether this command is called by an agent, which may require different logging/return format")
):
    """
    Run the tracking module of SpikeCV. Using SNNTracker as default.
    """
    import json
    import os
    import sys
    from argparse import Namespace
    import contextlib
    result_dict = {"status": "error", "message": "Unknown error", "result": None} 
    err_msg = ""
    try:
        if algorithm != "snn_tracker":
            err_msg = f"Error: Unsupported algorithm '{algorithm}'. Currently only 'snn_tracker' is supported."
            raise ValueError(err_msg)
        from spikecv.examples import test_snntracker as snn_tracker
        
        typer.echo("Running the SNNTracker tracking module of SpikeCV...", err=True)
        if snn_tracker is None:
            err_msg = "Error: Could not import snn_tracker from spikecv.examples.test_snntracker"
            raise ImportError(err_msg)
            
        if metrics and scene_idx not in [0, 1]:
            err_msg = "Error: Metrics can only be enabled for scenes with ground truth (0: spike59 and 1: rotTrans)."
            raise ValueError(err_msg)
            
        if not os.path.isabs(data_path):
            data_path = os.path.abspath(data_path)
        if not os.path.exists(data_path):
            err_msg = f"Error: Data path '{data_path}' does not exist."
            raise FileNotFoundError(err_msg)
        
        args = Namespace(
            scene_idx=scene_idx,
            attention_size=attention_size,
            data_path=data_path,
            label_type="tracking",
            metrics=metrics,
        )
        typer.echo("Arguments for tracking module:", err=True)
        typer.echo(args, err=True)
        
        # 先预设为成功，如果后面中途出错会被 except 覆盖
        result_dict.update({
            "status": "success",
            "message": "Tracking module ran successfully."
        })
        
        with contextlib.redirect_stdout(sys.stderr):
            with contextlib.redirect_stderr(sys.stderr):
                result_dict["result"] = snn_tracker.main(args)
                
    except (Exception, SystemExit) as e:
        final_msg = err_msg if err_msg else str(e)
        result_dict.update({"status": "error", "message": final_msg})
        typer.echo(final_msg, err=True)
        if not agent_used:
            raise typer.Exit(1)
    
    finally:
        if agent_used:
            typer.echo(json.dumps(result_dict))
            
            
@app.command(name="reconst",
             help="Run the reconstruction module of SpikeCV.",
             short_help="Run reconstruction module",
             no_args_is_help=True)
def reconst(
    algorithm: str = typer.Option("tfstp", "--algorithm", "-a", help="Reconstruction algorithm to use. Currently only 'tfstp' is supported."),
    yaml_file_path: str = typer.Option("recVidarReal2019/config.yaml", "--yaml-file-path", "-yaml", help="Relative Path to spike dataset yaml file"),
    dat_file_path: str = typer.Option("recVidarReal2019/classA/car-100kmh.dat", "--dat-file-path", "-dat", help="Relative Path to spike data file, e.g., 'recVidarReal2019/classA/car-100kmh.dat'"),
    begin_idx: int = typer.Option(500, "--begin-idx", "-begin", help="Begin index of spikes"),
    block_len: int = typer.Option(1500, "--block-len", "-b", help="Number of spike frames to process"),
    stp_u0: float = typer.Option(0.15, "--u0", help="STP parameter u0"),
    stp_d: float = typer.Option(1.0, "--stp-d", help="STP parameter D (D = 0.05 * 20)"),
    stp_f: float = typer.Option(10.0, "--stp-f", help="STP parameter F (F = 0.5 * 20)"),
    stp_f_small: float = typer.Option(0.15, "--stp-f-small", help="STP parameter f"),
    stp_time_unit: int = typer.Option(1, "--time-unit", "-tu", help="STP parameter time unit"),
    agent_used: bool = typer.Option(False, "--agent-used", "-agent", help="Whether this command is called by an agent, which may require different logging/return format")
):
    """
    Run the reconstruction module of SpikeCV.
    """
    import json
    import os
    import sys
    from argparse import Namespace
    import contextlib
    from spikecv.examples import test_tfstp_cli as tfstp
    
    result_dict = {"status": "error", "message": "Unknown error", "result": None} 
    err_msg = ""
    try:
        if algorithm != "tfstp":
            err_msg = f"Error: Unsupported algorithm '{algorithm}'. Currently only 'tfstp' is supported."
            raise ValueError(err_msg)
            
        typer.echo("Running the TFSTP reconstruction module of SpikeCV...", err=True)
        
        # 路径处理
        if not os.path.isabs(yaml_file_path):
            yaml_file_path = os.path.abspath(yaml_file_path)
            
        args = Namespace(
            yaml_file_path=yaml_file_path,
            dat_file_path=dat_file_path,
            begin_idx=begin_idx,
            block_len=block_len,
            stp_u0=stp_u0,
            stp_D=stp_d,
            stp_F=stp_f,
            stp_f=stp_f_small,
            stp_time_unit=stp_time_unit
        )
        
        with contextlib.redirect_stdout(sys.stderr):
            with contextlib.redirect_stderr(sys.stderr):
                result_dict["result"] = tfstp.main(args)
        
        result_dict.update({
            "status": "success",
            "message": "Reconstruction module ran successfully."
        })
                
    except (Exception, SystemExit) as e:
        final_msg = err_msg if err_msg else str(e)
        result_dict.update({"status": "error", "message": final_msg})
        typer.echo(final_msg, err=True)
        if not agent_used:
            raise typer.Exit(1)
    
    finally:
        if agent_used:
            typer.echo(json.dumps(result_dict))