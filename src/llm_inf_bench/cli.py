"""CLI entry point for llm-inf-bench."""

import typer

app = typer.Typer(
    name="llm-inf-bench",
    help="Benchmarking and observability framework for LLM inference on RunPod GPUs.",
)


@app.command()
def run() -> None:
    """Run a benchmark suite."""
    typer.echo("Not yet implemented.")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
