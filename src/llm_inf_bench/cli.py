"""CLI entry point for llm-inf-bench."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from llm_inf_bench.config.loader import APP_CONFIG_DIR, APP_CONFIG_PATH, load_experiment
from llm_inf_bench.config.schema import AppConfig
from llm_inf_bench.config.validation import ConfigValidationError
from llm_inf_bench.providers.runpod.client import RunPodProvider
from llm_inf_bench.providers.runpod.pricing import check_pricing_staleness, estimate_cost

app = typer.Typer(
    name="llm-inf-bench",
    help="Benchmarking and observability framework for LLM inference on RunPod GPUs.",
)

console = Console()


@app.command()
def init() -> None:
    """Create default application config at ~/.llm-inf-bench/config.yaml."""
    APP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if APP_CONFIG_PATH.exists():
        console.print(f"[yellow]Config already exists:[/yellow] {APP_CONFIG_PATH}")
        console.print("Edit it to update your settings. Not overwriting.")
        return

    defaults = AppConfig()
    with open(APP_CONFIG_PATH, "w") as f:
        yaml.dump(defaults.model_dump(), f, default_flow_style=False, sort_keys=False)

    console.print(f"[green]Created config:[/green] {APP_CONFIG_PATH}")
    console.print("Set your RunPod API key:")
    console.print(f"  Edit {APP_CONFIG_PATH} and set runpod.api_key")
    console.print("  Or set the RUNPOD_API_KEY environment variable")


@app.command()
def doctor() -> None:
    """Check environment readiness."""
    all_ok = True

    # 1. Python version
    py_ver = sys.version_info
    if py_ver >= (3, 10):
        console.print(f"[green]OK[/green]   Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    else:
        console.print(
            f"[red]FAIL[/red] Python {py_ver.major}.{py_ver.minor}.{py_ver.micro} "
            f"(need >= 3.10)"
        )
        all_ok = False

    # 2. RunPod API key
    if RunPodProvider.check_api_key_configured():
        console.print("[green]OK[/green]   RunPod API key configured")
    else:
        console.print("[red]FAIL[/red] RunPod API key not found")
        console.print("       Set RUNPOD_API_KEY or run 'llm-inf-bench init'")
        all_ok = False

    # 3. Pricing staleness
    is_stale, age_days = check_pricing_staleness()
    if is_stale:
        console.print(f"[yellow]WARN[/yellow] GPU pricing data is {age_days} days old")
    else:
        console.print(f"[green]OK[/green]   GPU pricing data ({age_days} days old)")

    # 4. Network connectivity (non-fatal)
    try:
        import httpx

        resp = httpx.get("https://api.runpod.io/graphql", timeout=5)
        if resp.status_code in (200, 400):  # 400 = no query, but API is reachable
            console.print("[green]OK[/green]   RunPod API reachable")
        else:
            console.print(f"[yellow]WARN[/yellow] RunPod API returned status {resp.status_code}")
    except Exception:
        console.print("[yellow]WARN[/yellow] Could not reach RunPod API (are you offline?)")

    if not all_ok:
        raise typer.Exit(1)


@app.command()
def run(
    config: str = typer.Argument(help="Path to experiment YAML config"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config and show summary only"),
    confirm: bool = typer.Option(False, "--confirm", help="Skip cost confirmation prompt"),
    server_url: str | None = typer.Option(
        None, "--server-url", help="Use existing server instead of provisioning"
    ),
) -> None:
    """Run a benchmark experiment."""
    # Load and validate config
    config_path = Path(config)
    if not config_path.exists():
        console.print(f"[red]Error:[/red] Config file not found: {config}")
        raise typer.Exit(1)

    try:
        experiment = load_experiment(config_path)
    except ConfigValidationError as e:
        console.print("[red]Config validation failed:[/red]")
        for err in e.errors:
            console.print(f"  - {err}")
        raise typer.Exit(1)
    except ValidationError as e:
        console.print("[red]Config parsing failed:[/red]")
        console.print(str(e))
        raise typer.Exit(1)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Display summary
    table = Table(title=f"Experiment: {experiment.name}")
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    table.add_row("Model", experiment.model.name)
    if experiment.model.quantization:
        table.add_row("Quantization", experiment.model.quantization)
    table.add_row("Framework", experiment.framework)
    table.add_row("GPU", f"{experiment.infrastructure.gpu_type} x{experiment.infrastructure.gpu_count}")
    table.add_row("Workload", experiment.workload.type)
    table.add_row("Requests", str(experiment.workload.requests.count))

    cost = estimate_cost(experiment.infrastructure.gpu_type, experiment.infrastructure.gpu_count, 60)
    if cost is not None:
        table.add_row("Est. cost", f"${cost:.2f}/hr (community)")

    console.print(table)

    # Staleness warning
    is_stale, age_days = check_pricing_staleness()
    if is_stale:
        console.print(
            f"\n[yellow]Warning:[/yellow] GPU pricing data is {age_days} days old. "
            f"Costs may be inaccurate."
        )

    if dry_run:
        console.print("\n[green]Config is valid.[/green] Dry run complete.")
        return

    console.print("\n[yellow]Execution not yet implemented.[/yellow]")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
