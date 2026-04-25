import typer

app = typer.Typer(name="reconst", help="Reconstruction modules in SpikeCV.", no_args_is_help=False)


@app.callback(invoke_without_command=True)
def default_reconst(
    ctx: typer.Context,
    yaml_file_path: str = typer.Option("recVidarReal2019/config.yaml", "--yaml-file-path", "-yaml", help="Path to spike dataset yaml file"),
    dat_file_path: str = typer.Option("recVidarReal2019/classA/car-100kmh.dat", "--dat-file-path", "-dat", help="Path to spike data file"),
    begin_idx: int = typer.Option(500, "--begin-idx", "-begin", help="Begin index of spikes"),
    block_len: int = typer.Option(1500, "--block-len", "-b", help="Number of spike frames to process"),
    stp_u0: float = typer.Option(0.15, "--u0", "-u0", help="STP parameter u0"),
    stp_d: float = typer.Option(1.0, "--stp-d", "-d", help="STP parameter D"),
    stp_f: float = typer.Option(10.0, "--stp-F", "-F", help="STP parameter F"),
    stp_f_small: float = typer.Option(0.15, "--stp-f", "-f", help="STP parameter f"),
    stp_time_unit: int = typer.Option(1, "--time-unit", "-tu", help="STP parameter time unit"),
    agent_used: bool = typer.Option(False, "--agent-used", "-agent", help="Whether this command is called by an agent")
):
    """
    Reconstruction module group. If no algorithm specified, TFSTP is used as default. Use reconst --help for list of algorithms.
    """
    if ctx.invoked_subcommand is None:
        typer.echo("Note: No algorithm specified. Running TFSTP with default demo settings...", err=True)
        tfstp(yaml_file_path, dat_file_path, begin_idx, block_len, stp_u0, stp_d, stp_f, stp_f_small, stp_time_unit, agent_used)


@app.command(
    name="tfstp", 
    help="Run reconstruction using TFSTP.", 
    short_help="TFSTP",
    no_args_is_help=False,
    rich_help_panel="Algorithms",
    )
def tfstp(
    yaml_file_path: str = typer.Option("recVidarReal2019/config.yaml", "--yaml-file-path", "-yaml", help="Path to spike dataset yaml file"),
    dat_file_path: str = typer.Option("recVidarReal2019/classA/car-100kmh.dat", "--dat-file-path", "-dat", help="Path to spike data file"),
    begin_idx: int = typer.Option(500, "--begin-idx", "-begin", help="Begin index of spikes"),
    block_len: int = typer.Option(1500, "--block-len", "-b", help="Number of spike frames to process"),
    stp_u0: float = typer.Option(0.15, "--u0", "-u0", help="STP parameter u0"),
    stp_d: float = typer.Option(1.0, "--stp-d", "-d", help="STP parameter D"),
    stp_f: float = typer.Option(10.0, "--stp-f", "-F", help="STP parameter F"),
    stp_f_small: float = typer.Option(0.15, "--stp-f-small", "-fs", help="STP parameter f"),
    stp_time_unit: int = typer.Option(1, "--time-unit", "-tu", help="STP parameter time unit"),
    agent_used: bool = typer.Option(False, "--agent-used", "-agent", help="Whether this command is called by an agent")
):
    import json
    import os
    import sys
    from argparse import Namespace
    import contextlib
    from spikecv.examples import test_tfstp_cli as tfstp_impl
    
    result_dict = {"status": "error", "message": "Unknown error", "result": None} 
    err_msg = ""
    try:
        typer.echo("Running the TFSTP reconstruction module of SpikeCV...", err=True)
        
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
                result_dict["result"] = tfstp_impl.main(args)
        
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
