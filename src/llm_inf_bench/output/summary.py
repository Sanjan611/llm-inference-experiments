"""Post-run summary rendering with Rich."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_inf_bench.metrics.aggregator import AggregatedMetrics, PercentileStats

if TYPE_CHECKING:
    from llm_inf_bench.metrics.storage import StoredResult

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


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _fmt_delta(a: float, b: float, lower_is_better: bool = True) -> str:
    """Format the difference between two values with colour.

    *a* is the baseline, *b* is the comparison.  Returns a string like
    ``"-13ms (-16.7%)"`` coloured green for improvements, red for regressions.
    """
    diff = b - a
    if a != 0:
        pct = diff / abs(a) * 100
        pct_str = f" ({pct:+.1f}%)"
    else:
        pct_str = ""

    is_improvement = (diff < 0) if lower_is_better else (diff > 0)
    colour = "green" if is_improvement else ("red" if diff != 0 else "")

    sign = "+" if diff > 0 else ""
    text = f"{sign}{diff:.1f}{pct_str}"
    return f"[{colour}]{text}[/{colour}]" if colour else text


def _safe_fmt(val: float | None, suffix: str = "") -> str:
    if val is None:
        return "n/a"
    return f"{val:.1f}{suffix}"


def _percentile_rows(
    label: str,
    stats_a: PercentileStats | None,
    stats_b: PercentileStats | None,
) -> list[tuple[str, str, str, str]]:
    """Generate table rows for p50/p95/p99 of a percentile metric."""
    rows: list[tuple[str, str, str, str]] = []
    for pname in ("p50", "p95", "p99"):
        va = getattr(stats_a, pname, None) if stats_a else None
        vb = getattr(stats_b, pname, None) if stats_b else None
        delta = _fmt_delta(va, vb) if va is not None and vb is not None else ""
        rows.append((
            f"  {label} {pname}",
            _safe_fmt(va, "ms"),
            _safe_fmt(vb, "ms"),
            delta,
        ))
    return rows


def print_comparison(result_a: StoredResult, result_b: StoredResult) -> None:
    """Render a side-by-side comparison table for two results."""
    from llm_inf_bench.metrics.storage import reconstruct_aggregated_metrics

    table = Table(title="Comparison", expand=False)
    table.add_column("Metric", style="bold")
    table.add_column(f"Run A ({result_a.run_id})")
    table.add_column(f"Run B ({result_b.run_id})")
    table.add_column("Delta")

    # --- Header / config ---
    def _cfg(r: StoredResult, key: str) -> str:
        return str(r.experiment.get(key, "n/a"))

    def _model(r: StoredResult) -> str:
        m = r.experiment.get("model", {})
        return m.get("name", "n/a") if isinstance(m, dict) else str(m)

    def _gpu(r: StoredResult) -> str:
        infra = r.experiment.get("infrastructure", {})
        return infra.get("gpu_type", "n/a") if isinstance(infra, dict) else "n/a"

    table.add_row("Framework", _cfg(result_a, "framework"), _cfg(result_b, "framework"), "")
    table.add_row("Model", _model(result_a), _model(result_b), "")
    table.add_row("GPU", _gpu(result_a), _gpu(result_b), "")
    table.add_section()

    # --- Metrics ---
    ma = reconstruct_aggregated_metrics(result_a.summary)
    mb = reconstruct_aggregated_metrics(result_b.summary)

    table.add_row(
        "Requests",
        f"{ma.successful_requests}/{ma.total_requests}",
        f"{mb.successful_requests}/{mb.total_requests}",
        "",
    )
    table.add_row(
        "Duration",
        _safe_fmt(ma.total_duration_s, "s"),
        _safe_fmt(mb.total_duration_s, "s"),
        _fmt_delta(ma.total_duration_s, mb.total_duration_s),
    )
    table.add_row(
        "Throughput (tok/s)",
        _safe_fmt(ma.tokens_per_second),
        _safe_fmt(mb.tokens_per_second),
        _fmt_delta(ma.tokens_per_second, mb.tokens_per_second, lower_is_better=False),
    )
    table.add_row(
        "Throughput (req/s)",
        _safe_fmt(ma.requests_per_second),
        _safe_fmt(mb.requests_per_second),
        _fmt_delta(ma.requests_per_second, mb.requests_per_second, lower_is_better=False),
    )
    table.add_section()

    # TTFT
    for row in _percentile_rows("TTFT", ma.ttft, mb.ttft):
        table.add_row(*row)
    table.add_section()

    # E2E Latency
    for row in _percentile_rows("Latency", ma.e2e_latency, mb.e2e_latency):
        table.add_row(*row)
    table.add_section()

    # TBT
    for row in _percentile_rows("TBT", ma.tbt, mb.tbt):
        table.add_row(*row)

    console.print(table)
