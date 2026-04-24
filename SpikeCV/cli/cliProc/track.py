import typer

app = typer.Typer(name="track", help="Tracking modules in SpikeCV.", no_args_is_help=False)


@app.callback(invoke_without_command=True)
def default_track(
    ctx: typer.Context,
    scene_idx: int = typer.Option(0, "--scene-idx", "-s", 
                                  help="Index of the test scene. 0: spike59, 1: rotTrans, 2: cplCam, 3: cpl1, 4: ball, 5: badminton, 6: pingpong"),
    attention_size: int = typer.Option(15, "--attention-size", "-attn_size", help="Size of attention window"),
    data_path: str = typer.Option("motVidarReal2020/", "--data-path", "-d", help="Path to dataset root"),
    metrics: bool = typer.Option(False, "--metrics", "-m", help="Enable quantitative metrics (requires GT)"),
    agent_used: bool = typer.Option(False, "--agent-used", "-agent", help="Whether this command is called by an agent")
):
    """
    Tracking module group. If no algorithm specified, SNNTracker is used as default. Use track --help for list of algorithms.
    """
    if ctx.invoked_subcommand is None:
        typer.echo("Note: No algorithm specified. Running SNNTracker with default demo settings...", err=True)
        snn_tracker(scene_idx, attention_size, data_path, metrics, agent_used)


@app.command(name="snn-tracker", 
             help="Run tracking using SNNTracker.", 
             short_help="SNNTracker", 
             no_args_is_help=False,
             rich_help_panel="Algorithms")
def snn_tracker(
    scene_idx: int = typer.Option(0, "--scene-idx", "-s", 
                                  help="Index of the test scene. 0: spike59, 1: rotTrans, 2: cplCam, 3: cpl1, 4: ball, 5: badminton, 6: pingpong"),
    attention_size: int = typer.Option(15, "--attention-size", "-attn_size", help="Size of attention window"),
    data_path: str = typer.Option("motVidarReal2020/", "--data-path", "-d", help="Path to dataset root"),
    metrics: bool = typer.Option(False, "--metrics", "-m", help="Enable quantitative metrics (requires GT)"),
    agent_used: bool = typer.Option(False, "--agent-used", "-agent", help="Whether this command is called by an agent")
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
        typer.echo("Running the SNNTracker tracking module of SpikeCV...", err=True)
            
        if metrics and scene_idx not in [0, 1]:
            err_msg = "Error: Metrics can only be enabled for scenes with ground truth (0: spike59 and 1: rotTrans)."
            raise ValueError(err_msg)
            
        if not os.path.isabs(data_path):
            data_path = os.path.abspath(data_path)
        if not os.path.exists(data_path):
            err_msg = f"Error: Data path '{data_path}' does not exist."
            raise FileNotFoundError(err_msg)
    
        from spikecv.examples import test_snntracker as snn_tracker_impl
        
        args = Namespace(
            scene_idx=scene_idx,
            attention_size=attention_size,
            data_path=data_path,
            label_type="tracking",
            metrics=metrics,
        )
        
        typer.echo("Arguments for module:", err=True)
        typer.echo(args, err=True)
        
        result_dict.update({
            "status": "success",
            "message": "Tracking module ran successfully."
        })
        
        with contextlib.redirect_stdout(sys.stderr):
            with contextlib.redirect_stderr(sys.stderr):
                result_dict["result"] = snn_tracker_impl.main(args)
                
    except (Exception, SystemExit) as e:
        final_msg = err_msg if err_msg else str(e)
        result_dict.update({"status": "error", "message": final_msg})
        typer.echo(final_msg, err=True)
        if not agent_used:
            raise typer.Exit(1)
    
    finally:
        if agent_used:
            typer.echo(json.dumps(result_dict))