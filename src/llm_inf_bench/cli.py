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
from llm_inf_bench.config.sweep import expand_sweep
from llm_inf_bench.config.validation import ConfigValidationError
from llm_inf_bench.metrics.aggregator import aggregate_multi_turn_results, aggregate_results
from llm_inf_bench.metrics.collector import RequestResult, RunMetadata
from llm_inf_bench.metrics.gpu import GpuMetricsScraper, GpuTimeSeries
from llm_inf_bench.metrics.storage import (
    generate_run_id,
    list_results,
    load_result,
    reconstruct_aggregated_metrics,
    save_results,
)
from llm_inf_bench.output.console import BenchmarkProgress
from llm_inf_bench.output.summary import print_comparison, print_summary
from llm_inf_bench.providers.runpod.client import RunPodProvider
from llm_inf_bench.providers.runpod.pricing import check_pricing_staleness, estimate_cost
from llm_inf_bench.runners import Runner, create_runner
from llm_inf_bench.runners.base import HealthCheckTimeout
from llm_inf_bench.workloads import create_workload, load_multi_turn_prompts, load_prompts

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="llm-inf-bench",
    help="Benchmarking and observability framework for LLM inference on RunPod GPUs.",
)

console = Console()

results_app = typer.Typer(help="Browse and compare experiment results.")
app.add_typer(results_app, name="results")


@results_app.command("list")
def results_list(
    output_dir: str = typer.Option("results/", "--dir", help="Results directory"),
) -> None:
    """List completed experiment runs."""
    stored = list_results(output_dir)
    if not stored:
        console.print("No results found.")
        return

    table = Table(title="Experiment Results")
    table.add_column("Run ID", style="bold")
    table.add_column("Status")
    table.add_column("Framework")
    table.add_column("Model")
    table.add_column("Requests")
    table.add_column("Date")

    for r in stored:
        framework = r.experiment.get("framework", "?")
        model_cfg = r.experiment.get("model", {})
        model_name = model_cfg.get("name", "?") if isinstance(model_cfg, dict) else str(model_cfg)
        total = r.summary.get("total_requests", "?")
        success = r.summary.get("successful_requests", "?")
        date = r.metadata.get("started_at", "")
        if isinstance(date, str) and len(date) > 10:
            date = date[:10]
        table.add_row(r.run_id, r.status, framework, model_name, f"{success}/{total}", date)

    console.print(table)


@results_app.command("show")
def results_show(
    run_id: str = typer.Argument(help="Run ID (exact or partial match)"),
    output_dir: str = typer.Option("results/", "--dir", help="Results directory"),
) -> None:
    """Show details for a single experiment run."""
    try:
        result = load_result(output_dir, run_id)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Config summary
    framework = result.experiment.get("framework", "?")
    model_cfg = result.experiment.get("model", {})
    model_name = model_cfg.get("name", "?") if isinstance(model_cfg, dict) else str(model_cfg)
    infra = result.experiment.get("infrastructure", {})
    gpu = infra.get("gpu_type", "?") if isinstance(infra, dict) else "?"
    workload_cfg = result.experiment.get("workload", {})
    workload_type = workload_cfg.get("type", "?") if isinstance(workload_cfg, dict) else "?"

    config_table = Table(title=f"Run: {result.run_id}")
    config_table.add_column("Setting", style="bold")
    config_table.add_column("Value")
    config_table.add_row("Status", result.status)
    config_table.add_row("Framework", framework)
    config_table.add_row("Model", model_name)
    config_table.add_row("GPU", gpu)
    config_table.add_row("Workload", workload_type)
    console.print(config_table)

    # Metrics summary
    try:
        metrics = reconstruct_aggregated_metrics(result.summary)
        print_summary(metrics)
    except (KeyError, TypeError):
        console.print("[yellow]Could not reconstruct metrics summary.[/yellow]")

    # Metadata
    console.print("\n[bold]Metadata[/bold]")
    metadata_keys = (
        "server_url",
        "pod_id",
        "gpu_type",
        "gpu_count",
        "cost_per_hr",
        "started_at",
        "finished_at",
    )
    for key in metadata_keys:
        val = result.metadata.get(key)
        if val:
            console.print(f"  {key}: {val}")


@results_app.command("compare")
def results_compare(
    id_a: str = typer.Argument(help="Run ID for run A"),
    id_b: str = typer.Argument(help="Run ID for run B"),
    output_dir: str = typer.Option("results/", "--dir", help="Results directory"),
) -> None:
    """Compare two experiment runs side by side."""
    try:
        result_a = load_result(output_dir, id_a)
        result_b = load_result(output_dir, id_b)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    print_comparison(result_a, result_b)


@app.command()
def dashboard(
    port: int = typer.Option(8420, "--port", help="Port number for the dashboard server"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't auto-open browser"),
) -> None:
    """Launch the experiment dashboard with live metrics visualization."""
    try:
        from llm_inf_bench.dashboard.server import run_server
    except ImportError:
        console.print(
            "[red]Dashboard dependencies not installed.[/red]\n"
            "Install them with: [bold]uv sync --extra dashboard[/bold]"
        )
        raise typer.Exit(1)

    console.print(f"Starting dashboard on [bold]http://localhost:{port}[/bold]")
    run_server(port=port, open_browser=not no_browser)


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
            f"[red]FAIL[/red] Python {py_ver.major}.{py_ver.minor}.{py_ver.micro} (need >= 3.10)"
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
    table.add_row(
        "GPU", f"{experiment.infrastructure.gpu_type} x{experiment.infrastructure.gpu_count}"
    )
    table.add_row("Workload", experiment.workload.type)
    if experiment.workload.type == "multi_turn" and experiment.workload.conversation:
        conv = experiment.workload.conversation
        total = experiment.workload.requests.count * conv.turns
        table.add_row("Conversations", str(experiment.workload.requests.count))
        table.add_row("Turns", str(conv.turns))
        table.add_row("Total requests", str(total))
    else:
        table.add_row("Requests", str(experiment.workload.requests.count))
    if experiment.workload.batch_size is not None:
        table.add_row("Batch size", str(experiment.workload.batch_size))
    if experiment.workload.concurrency is not None:
        table.add_row("Concurrency", str(experiment.workload.concurrency))

    # Show sweep info
    sweep_variations = expand_sweep(experiment)
    if len(sweep_variations) > 1:
        sweep_desc = f"{len(sweep_variations)} iterations"
        if experiment.workload.sweep:
            parts: list[str] = []
            if experiment.workload.sweep.concurrency:
                parts.append(f"concurrency={experiment.workload.sweep.concurrency}")
            if experiment.workload.sweep.batch_size:
                parts.append(f"batch_size={experiment.workload.sweep.batch_size}")
            sweep_desc += f" ({', '.join(parts)})"
        table.add_row("Sweep", sweep_desc)

    cost = estimate_cost(
        experiment.infrastructure.gpu_type, experiment.infrastructure.gpu_count, 60
    )
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
        asyncio.run(_execute_run(experiment, server_url=server_url))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(1)


async def _execute_run(
    experiment: ExperimentConfig,
    server_url: str | None = None,
) -> None:
    """Async orchestration: provision -> health -> sweep loop -> cleanup."""
    progress = BenchmarkProgress()
    provider: RunPodProvider | None = None
    pod_id: str | None = None
    gpu_type: str | None = None
    gpu_count: int | None = None
    cost_per_hr: float | None = None
    runner: Runner | None = None

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
            gpu_type = pod.gpu_type
            gpu_count = pod.gpu_count
            cost_per_hr = pod.cost_per_hr
            base_url = pod.base_url
            progress.phase_provisioning_done(pod_id, base_url)

        # Phase 2: Health check
        runner = create_runner(
            framework=experiment.framework,
            base_url=base_url,
            model=experiment.model.name,
        )
        progress.phase_model_loading(experiment.model.name)
        health_start = time.monotonic()
        try:
            await runner.wait_for_health()
        except HealthCheckTimeout as e:
            console.print(f"\n[red]Health check failed:[/red] {e}")
            raise typer.Exit(1)
        progress.phase_model_loading_done(time.monotonic() - health_start)

        # Sweep loop
        variations = expand_sweep(experiment)
        for i, (variation, params) in enumerate(variations, 1):
            if len(variations) > 1:
                progress.phase_sweep_iteration(i, len(variations), params)
                progress.reset_progress()

            await _execute_iteration(
                variation,
                runner,
                progress,
                base_url,
                pod_id,
                server_url,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                cost_per_hr=cost_per_hr,
            )

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


async def _execute_iteration(
    experiment: ExperimentConfig,
    runner: Runner,
    progress: BenchmarkProgress,
    base_url: str,
    pod_id: str | None,
    server_url: str | None,
    *,
    gpu_type: str | None = None,
    gpu_count: int | None = None,
    cost_per_hr: float | None = None,
) -> None:
    """Execute a single benchmark iteration (one sweep variation)."""
    scraper: GpuMetricsScraper | None = None
    gpu_time_series: GpuTimeSeries | None = None
    results: list[RequestResult] = []
    run_started = datetime.now(timezone.utc)
    exec_start: float | None = None

    try:
        # Start GPU metrics scraper
        if experiment.metrics.collect_gpu_metrics and hasattr(runner, "http_client"):
            scraper = GpuMetricsScraper(
                client=runner.http_client,
                framework=experiment.framework,
                sample_interval_ms=experiment.metrics.sample_interval_ms,
            )
            scraper.start()

        # Execution
        conversations = None
        conversation_turns = None
        if experiment.workload.type == "multi_turn":
            conv = experiment.workload.conversation
            assert conv is not None  # validated earlier
            conversations = load_multi_turn_prompts(
                source=experiment.workload.requests.source,
                count=experiment.workload.requests.count,
                turns=conv.turns,
                system_prompt=conv.system_prompt,
                user_messages=conv.user_messages,
                shared_prefix=conv.shared_prefix,
            )
            conversation_turns = conv.turns
            prompts = []  # not used for multi_turn
        else:
            prompts = load_prompts(
                experiment.workload.requests.source,
                experiment.workload.requests.count,
            )
        workload = create_workload(
            workload_type=experiment.workload.type,
            prompts=prompts,
            model=experiment.model.name,
            max_tokens=experiment.workload.parameters.max_tokens,
            temperature=experiment.workload.parameters.temperature,
            on_request_complete=progress.on_request_complete,
            batch_size=experiment.workload.batch_size,
            concurrency=experiment.workload.concurrency,
            conversations=conversations,
            conversation_turns=conversation_turns,
        )

        rich_progress = progress.phase_execution_start(
            workload.total_requests(), experiment.workload.type
        )
        exec_start = time.monotonic()
        with rich_progress:
            results = await workload.execute(runner)
        exec_duration = time.monotonic() - exec_start

        # Stop GPU metrics scraper
        if scraper is not None:
            gpu_time_series = await scraper.stop()

        # Aggregate, save, summarize
        multi_turn_metrics = None
        if experiment.workload.type == "multi_turn":
            conv = experiment.workload.conversation
            assert conv is not None
            multi_turn_metrics = aggregate_multi_turn_results(
                results,
                total_duration_s=exec_duration,
                turns=conv.turns,
            )
            aggregated = multi_turn_metrics.overall
        else:
            aggregated = aggregate_results(results, total_duration_s=exec_duration)
        if gpu_time_series is not None:
            aggregated.gpu_summary = GpuMetricsScraper.summarize(gpu_time_series)

        run_id = generate_run_id(experiment.name)
        metadata = RunMetadata(
            run_id=run_id,
            experiment_name=experiment.name,
            started_at=run_started,
            finished_at=datetime.now(timezone.utc),
            server_url=base_url,
            pod_id=pod_id,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            cost_per_hr=cost_per_hr,
            status="completed" if aggregated.failed_requests == 0 else "partial",
        )

        output_dir = experiment.metrics.output_dir
        result_path = save_results(
            output_dir,
            metadata,
            experiment,
            results,
            aggregated,
            gpu_time_series=gpu_time_series,
        )
        console.print(f"\n[green]Results saved:[/green] {result_path}")
        print_summary(aggregated)
        if multi_turn_metrics is not None:
            from llm_inf_bench.output.summary import print_turn_breakdown

            print_turn_breakdown(multi_turn_metrics)

    except KeyboardInterrupt:
        # Stop scraper (best-effort)
        if scraper is not None and gpu_time_series is None:
            try:
                gpu_time_series = await scraper.stop()
            except Exception:
                logger.warning("Failed to stop GPU scraper during interrupt", exc_info=True)

        # Save partial results
        if results:
            exec_duration = time.monotonic() - exec_start if exec_start is not None else 0
            aggregated = aggregate_results(results, total_duration_s=exec_duration)
            if gpu_time_series is not None:
                aggregated.gpu_summary = GpuMetricsScraper.summarize(gpu_time_series)
            run_id = generate_run_id(experiment.name)
            metadata = RunMetadata(
                run_id=run_id,
                experiment_name=experiment.name,
                started_at=run_started,
                finished_at=datetime.now(timezone.utc),
                server_url=server_url,
                pod_id=pod_id,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                cost_per_hr=cost_per_hr,
                status="partial",
            )
            output_dir = experiment.metrics.output_dir
            result_path = save_results(
                output_dir,
                metadata,
                experiment,
                results,
                aggregated,
                gpu_time_series=gpu_time_series,
            )
            console.print(f"\n[yellow]Partial results saved:[/yellow] {result_path}")
        raise


if __name__ == "__main__":
    app()
