import typer
from enum import Enum

class SpikeDataset(Enum):
    MOT_VIDAR_REAL_2020 = "motVidarReal2020"
    REC_VIDAR_REAL_2019 = "recVidarReal2019"

app = typer.Typer(name="data", 
                  help="Commands related to data handling in SpikeCV.", 
                  no_args_is_help=True)


@app.command(name="download",
             help="Download datasets for SpikeCV.",
             short_help="Download datasets",
             no_args_is_help=True)
def download(
    dataset: SpikeDataset = typer.Option(SpikeDataset.MOT_VIDAR_REAL_2020, "--dataset", "-d", help="Dataset to download. Currently only 'motVidarReal2020'(dataset for tracking) and 'recVidarReal2019'(dataset for reconstruction) are supported."),
    local_dir: str = typer.Option("datasets/", "--local-dir", "-l", help="Local directory to save the downloaded dataset. Default is 'datasets/'."),
    force: bool = typer.Option(False, "--force", help="Force re-download the dataset even if it already exists locally."),
    max_workers: int = typer.Option(10, "--max-workers", "-w", help="Maximum number of parallel workers for downloading. Default is 10."),
    agent_used: bool = typer.Option(False, "--agent-used", "-agent", help="Whether this command is called by an agent, which may require different logging/return format")
):
    import json
    import sys, contextlib
    from openi import openi_download_file
    repo_id: dict[str, str] = {
        # 可从数据集详情页中获取，格式为 “拥有者/数据集名或模型名”
        "motVidarReal2020": "Cordium/motVidarReal2025", 
        "recVidarReal2019": "Cordium/recVidarReal2019"
        
    }
    
    result_dict = {"status": "error", "message": "Unknown error", "result": None}
    err_msg = ""

    try:
            
        typer.echo(f"Downloading dataset '{str(dataset.value)}' to '{local_dir}'...", err=True)
        with contextlib.redirect_stdout(sys.stderr):
            with contextlib.redirect_stderr(sys.stderr):
                openi_download_file(repo_id=repo_id[str(dataset.value)], local_dir=local_dir, force=force, max_workers=max_workers)
        
        result_dict.update({
            "status": "success",
            "message": "Download completed successfully.",
            "result": {
                "dataset": str(dataset.value),
                "local_dir": local_dir
            }
        })
        typer.echo("Download completed successfully.", err=True)

    except (Exception, SystemExit) as e:
        final_msg = err_msg if err_msg else str(e)
        result_dict.update({"status": "error", "message": final_msg})
        typer.echo(f"An error occurred: {final_msg}", err=True)
        if not agent_used:
            raise typer.Exit(code=1)
            
    finally:
        if agent_used:
            typer.echo(json.dumps(result_dict))