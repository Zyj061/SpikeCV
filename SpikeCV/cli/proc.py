import typer
from spikecv.cli.cliProc.track import app as track_app
from spikecv.cli.cliProc.reconst import app as reconst_app

app = typer.Typer(name="proc", 
                  help="Commands related to spkProc in SpikeCV.", 
                  no_args_is_help=True)

app.add_typer(track_app)
app.add_typer(reconst_app)
