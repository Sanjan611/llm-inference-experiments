"""Rich console progress display for benchmark phases."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn

from llm_inf_bench.metrics.collector import RequestResult

console = Console()


class BenchmarkProgress:
    """Manages Rich output for the four benchmark phases."""

    def __init__(self) -> None:
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._completed: int = 0
        self._total_tokens: int = 0
        self._total_ttft: float = 0.0
        self._ttft_count: int = 0

    # --- Phase 1: Provisioning ---

    def phase_provisioning(self, gpu_type: str, gpu_count: int) -> None:
        console.print(
            f"\n[bold][1/4] Provisioning...[/bold]\n"
            f"      GPU: {gpu_type} x{gpu_count}"
        )

    def phase_provisioning_done(self, pod_id: str, server_url: str) -> None:
        console.print(f"      [green]Pod created:[/green] {pod_id}")
        console.print(f"      [green]Server URL:[/green] {server_url}")

    def phase_provisioning_skipped(self, server_url: str) -> None:
        console.print(
            f"\n[bold][1/4] Provisioning[/bold] [dim]skipped (using existing server)[/dim]\n"
            f"      Server URL: {server_url}"
        )

    # --- Phase 2: Model loading / health check ---

    def phase_model_loading(self, model: str) -> None:
        console.print(
            f"\n[bold][2/4] Waiting for model...[/bold]\n"
            f"      Model: {model}"
        )

    def phase_model_loading_done(self, elapsed_s: float) -> None:
        console.print(f"      [green]Health check passed[/green] ({elapsed_s:.1f}s)")

    # --- Phase 3: Execution ---

    def phase_execution_start(self, total_requests: int, workload_type: str) -> Progress:
        console.print(
            f"\n[bold][3/4] Running benchmark...[/bold]\n"
            f"      Workload: {workload_type}\n"
            f"      Requests: {total_requests}"
        )
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TextColumn("[dim]{task.fields[live_stats]}[/dim]"),
            console=console,
        )
        self._task_id = self._progress.add_task(
            "      Progress", total=total_requests, live_stats=""
        )
        self._completed = 0
        self._total_tokens = 0
        self._total_ttft = 0.0
        self._ttft_count = 0
        return self._progress

    def on_request_complete(self, result: RequestResult) -> None:
        """Callback for updating progress bar after each request."""
        if self._progress is None or self._task_id is None:
            return

        self._completed += 1

        if result.completion_tokens:
            self._total_tokens += result.completion_tokens
        if result.ttft_ms is not None:
            self._total_ttft += result.ttft_ms
            self._ttft_count += 1

        stats_parts: list[str] = []
        if self._total_tokens > 0 and result.e2e_latency_ms:
            stats_parts.append(f"tok={self._total_tokens}")
        if self._ttft_count > 0:
            avg_ttft = self._total_ttft / self._ttft_count
            stats_parts.append(f"avg_ttft={avg_ttft:.0f}ms")

        live_stats = "  ".join(stats_parts)
        self._progress.update(self._task_id, advance=1, live_stats=live_stats)

    # --- Sweep support ---

    def phase_sweep_iteration(
        self, iteration: int, total: int, params: dict[str, Any]
    ) -> None:
        """Print sweep progress header before each iteration."""
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        console.print(
            f"\n[bold cyan]── Sweep iteration {iteration}/{total} ──[/bold cyan]"
            f"  {params_str}"
        )

    def reset_progress(self) -> None:
        """Reset progress counters between sweep iterations."""
        self._progress = None
        self._task_id = None
        self._completed = 0
        self._total_tokens = 0
        self._total_ttft = 0.0
        self._ttft_count = 0

    # --- Phase 4: Cleanup ---

    def phase_cleanup(self) -> None:
        console.print("\n[bold][4/4] Cleanup...[/bold]")

    def phase_cleanup_done(self, terminated_pod: bool) -> None:
        if terminated_pod:
            console.print("      [green]Pod terminated[/green]")
        console.print("      [green]Done[/green]")
