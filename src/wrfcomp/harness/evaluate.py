"""`wrfcomp-eval` command line entry point. The scoring logic lands with the harness lanes."""

from __future__ import annotations

import typer

app = typer.Typer(help="WRFComp deterministic forecast verification CLI.")


@app.command()
def evaluate() -> None:
    """Placeholder until the harness is implemented."""
    typer.echo("wrfcomp-eval is not implemented yet.")
    raise typer.Exit(code=2)
