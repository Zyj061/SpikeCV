import typer
from SpikeCV.cli import proc, data
from importlib.metadata import version as get_version

def version_callback(value: bool):
    if value:
        version = get_version("SpikeCV")
        typer.echo(f"SpikeCV version: {version}")
        raise typer.Exit(0)


# 创建主 CLI 应用
app = typer.Typer(
    name="SpikeCV", 
    help="SpikeCV: An open-source framework for Spiking Computer Vision.",
    epilog="Use 'spikecv [COMMAND] --help' for more information on a command.",
    short_help="SpikeCV CLI",
    no_args_is_help=True
    )

@app.callback()
def common(
    version: bool = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,  # 让 --version 优先处理
        help="Show the application's version and exit."
    )
):
    """Common callback for the whole CLI."""
    pass

# 加载子命令
app.add_typer(proc.app) # spkProc 相关命令
app.add_typer(data.app) # download dataset

if __name__ == "__main__":
    app()