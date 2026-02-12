"""Post-run summary rendering with Rich."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_inf_bench.metrics.aggregator import AggregatedMetrics, PercentileStats

if TYPE_CHECKING:
    from llm_inf_bench.metrics.gpu import GpuSummary
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

    # Server Metrics section (when GPU data is available)
    gs = metrics.gpu_summary
    if gs is not None and gs.total_samples > 0:
        lines.append("")
        lines.append("  Server Metrics")
        if gs.kv_cache_usage_peak is not None and gs.kv_cache_usage_mean is not None:
            lines.append(
                f"  KV Cache:     peak={gs.kv_cache_usage_peak * 100:.1f}%"
                f"  mean={gs.kv_cache_usage_mean * 100:.1f}%"
            )
        if gs.prefix_cache_hit_rate is not None:
            lines.append(f"  Cache Hits:   {gs.prefix_cache_hit_rate * 100:.1f}%")
        if gs.generation_throughput is not None:
            lines.append(f"  Gen Tput:     {gs.generation_throughput:.1f} tok/s (server-side)")
        if gs.active_requests_peak is not None and gs.active_requests_mean is not None:
            lines.append(
                f"  Active Reqs:  peak={gs.active_requests_peak:.0f}"
                f"  mean={gs.active_requests_mean:.1f}"
            )
        lines.append(f"  Samples:      {gs.total_samples} ({gs.scrape_errors} errors)")

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

    # GPU / Server Metrics (only if at least one run has data)
    ga = ma.gpu_summary
    gb = mb.gpu_summary
    if ga is not None or gb is not None:
        table.add_section()

        def _gpu_val(gs: "GpuSummary | None", attr: str) -> float | None:
            if gs is None:
                return None
            return getattr(gs, attr, None)

        # KV Cache (peak) — displayed as percentage
        kv_a = _gpu_val(ga, "kv_cache_usage_peak")
        kv_b = _gpu_val(gb, "kv_cache_usage_peak")
        kv_a_pct = kv_a * 100 if kv_a is not None else None
        kv_b_pct = kv_b * 100 if kv_b is not None else None
        delta = _fmt_delta(kv_a_pct, kv_b_pct) if kv_a_pct is not None and kv_b_pct is not None else ""
        table.add_row(
            "KV Cache (peak)",
            _safe_fmt(kv_a_pct, "%"),
            _safe_fmt(kv_b_pct, "%"),
            delta,
        )

        # Cache Hit Rate
        hr_a = _gpu_val(ga, "prefix_cache_hit_rate")
        hr_b = _gpu_val(gb, "prefix_cache_hit_rate")
        hr_a_pct = hr_a * 100 if hr_a is not None else None
        hr_b_pct = hr_b * 100 if hr_b is not None else None
        delta = (
            _fmt_delta(hr_a_pct, hr_b_pct, lower_is_better=False)
            if hr_a_pct is not None and hr_b_pct is not None
            else ""
        )
        table.add_row(
            "Cache Hit Rate",
            _safe_fmt(hr_a_pct, "%"),
            _safe_fmt(hr_b_pct, "%"),
            delta,
        )

        # Gen Throughput (server)
        gt_a = _gpu_val(ga, "generation_throughput")
        gt_b = _gpu_val(gb, "generation_throughput")
        delta = (
            _fmt_delta(gt_a, gt_b, lower_is_better=False)
            if gt_a is not None and gt_b is not None
            else ""
        )
        table.add_row(
            "Gen Tput (server)",
            _safe_fmt(gt_a, " tok/s"),
            _safe_fmt(gt_b, " tok/s"),
            delta,
        )

    console.print(table)
