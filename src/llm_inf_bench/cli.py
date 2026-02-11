"""CLI entry point for llm-inf-bench."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from llm_inf_bench.config.loader import APP_CONFIG_DIR, APP_CONFIG_PATH, load_experiment
from llm_inf_bench.config.schema import AppConfig, ExperimentConfig
from llm_inf_bench.config.validation import ConfigValidationError
from llm_inf_bench.metrics.aggregator import aggregate_results
from llm_inf_bench.metrics.collector import RequestResult, RunMetadata
from llm_inf_bench.metrics.storage import generate_run_id, save_results
from llm_inf_bench.output.console import BenchmarkProgress
from llm_inf_bench.output.summary import print_summary
from llm_inf_bench.providers.runpod.client import RunPodProvider
from llm_inf_bench.providers.runpod.pricing import check_pricing_staleness, estimate_cost
from llm_inf_bench.runners.base import HealthCheckTimeout
from llm_inf_bench.runners.vllm import VLLMRunner
from llm_inf_bench.workloads.single import SingleWorkload, load_prompts

logger = logging.getLogger(__name__)

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

    # Cost confirmation
    if not confirm and cost is not None and cost >= 0.50:
        if not typer.confirm("\nProceed?"):
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(0)

    try:
        asyncio.run(
            _execute_run(experiment, server_url=server_url)
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(1)


async def _execute_run(
    experiment: ExperimentConfig,
    server_url: str | None = None,
) -> None:
    """Async orchestration: provision -> health -> execute -> aggregate -> save -> cleanup."""
    progress = BenchmarkProgress()
    provider: RunPodProvider | None = None
    pod_id: str | None = None
    runner: VLLMRunner | None = None
    results: list[RequestResult] = []
    run_started = datetime.now(timezone.utc)

    try:
        # Phase 1: Provisioning
        if server_url:
            progress.phase_provisioning_skipped(server_url)
            base_url = server_url.rstrip("/")
        else:
            progress.phase_provisioning(
                experiment.infrastructure.gpu_type,
                experiment.infrastructure.gpu_count,
            )
            provider = RunPodProvider()
            pod = provider.create(experiment)
            pod_id = pod.pod_id
            base_url = pod.base_url
            progress.phase_provisioning_done(pod_id, base_url)

        # Phase 2: Health check
        runner = VLLMRunner(base_url=base_url, model=experiment.model.name)
        progress.phase_model_loading(experiment.model.name)
        health_start = time.monotonic()
        try:
            await runner.wait_for_health(timeout=600, interval=5)
        except HealthCheckTimeout as e:
            console.print(f"\n[red]Health check failed:[/red] {e}")
            raise typer.Exit(1)
        progress.phase_model_loading_done(time.monotonic() - health_start)

        # Phase 3: Execution
        prompts = load_prompts(
            experiment.workload.requests.source,
            experiment.workload.requests.count,
        )
        workload = SingleWorkload(
            prompts=prompts,
            model=experiment.model.name,
            max_tokens=experiment.workload.parameters.max_tokens,
            temperature=experiment.workload.parameters.temperature,
            on_request_complete=progress.on_request_complete,
        )

        rich_progress = progress.phase_execution_start(
            workload.total_requests(), experiment.workload.type
        )
        exec_start = time.monotonic()
        with rich_progress:
            results = await workload.execute(runner)
        exec_duration = time.monotonic() - exec_start

        # Phase 4: Aggregate, save, summarize
        aggregated = aggregate_results(results, total_duration_s=exec_duration)
        run_id = generate_run_id(experiment.name)
        metadata = RunMetadata(
            run_id=run_id,
            experiment_name=experiment.name,
            started_at=run_started,
            finished_at=datetime.now(timezone.utc),
            server_url=base_url,
            pod_id=pod_id,
            status="completed" if aggregated.failed_requests == 0 else "partial",
        )

        output_dir = experiment.metrics.output_dir
        result_path = save_results(output_dir, metadata, experiment, results, aggregated)
        console.print(f"\n[green]Results saved:[/green] {result_path}")
        print_summary(aggregated)

    except KeyboardInterrupt:
        # Save partial results
        if results:
            exec_duration = time.monotonic() - exec_start if 'exec_start' in dir() else 0
            aggregated = aggregate_results(results, total_duration_s=exec_duration)
            run_id = generate_run_id(experiment.name)
            metadata = RunMetadata(
                run_id=run_id,
                experiment_name=experiment.name,
                started_at=run_started,
                finished_at=datetime.now(timezone.utc),
                server_url=server_url,
                pod_id=pod_id,
                status="partial",
            )
            output_dir = experiment.metrics.output_dir
            result_path = save_results(output_dir, metadata, experiment, results, aggregated)
            console.print(f"\n[yellow]Partial results saved:[/yellow] {result_path}")
        raise

    finally:
        # Cleanup
        progress.phase_cleanup()
        terminated_pod = False
        if runner:
            await runner.close()
        if provider and pod_id:
            try:
                provider.terminate(pod_id)
                terminated_pod = True
            except Exception as e:
                console.print(
                    f"[red]Failed to terminate pod {pod_id}:[/red] {e}\n"
                    f"[red]MANUAL ACTION REQUIRED: terminate via RunPod console.[/red]"
                )
        progress.phase_cleanup_done(terminated_pod)


if __name__ == "__main__":
    app()
