"""Post-run summary rendering with Rich."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

from llm_inf_bench.metrics.aggregator import AggregatedMetrics, PercentileStats

console = Console()


def _fmt_percentiles(stats: PercentileStats | None) -> str:
    """Format percentile stats as ``p50=42ms p95=78ms p99=112ms``."""
    if stats is None:
        return "n/a"
    return f"p50={stats.p50:.0f}ms  p95={stats.p95:.0f}ms  p99={stats.p99:.0f}ms"


def print_summary(metrics: AggregatedMetrics) -> None:
    """Render the benchmark summary panel."""
    lines = [
        f"  Requests:     {metrics.successful_requests}/{metrics.total_requests}"
        f"        Errors: {metrics.failed_requests}",
        f"  Duration:     {metrics.total_duration_s:.1f}s",
        f"  Throughput:   {metrics.tokens_per_second:.1f} tok/s"
        f"   ({metrics.requests_per_second:.1f} req/s)",
        "",
        f"  TTFT:         {_fmt_percentiles(metrics.ttft)}",
        f"  Latency:      {_fmt_percentiles(metrics.e2e_latency)}",
        f"  TBT:          {_fmt_percentiles(metrics.tbt)}",
    ]

    panel = Panel(
        "\n".join(lines),
        title="Summary",
        expand=False,
    )
    console.print(panel)
