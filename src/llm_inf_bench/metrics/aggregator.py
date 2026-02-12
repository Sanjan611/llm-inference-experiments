"""Percentile statistics and result aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from llm_inf_bench.metrics.collector import RequestResult

if TYPE_CHECKING:
    from llm_inf_bench.metrics.gpu import GpuSummary


@dataclass
class PercentileStats:
    """Percentile distribution for a metric."""

    p50: float
    p95: float
    p99: float
    mean: float
    min: float
    max: float


@dataclass
class AggregatedMetrics:
    """Summary statistics for a complete benchmark run."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration_s: float
    requests_per_second: float
    total_prompt_tokens: int
    total_completion_tokens: int
    tokens_per_second: float
    ttft: PercentileStats | None
    e2e_latency: PercentileStats | None
    tbt: PercentileStats | None
    gpu_summary: GpuSummary | None = None


def compute_percentiles(values: list[float]) -> PercentileStats | None:
    """Compute percentile statistics using linear interpolation.

    Returns None if the input list is empty.
    """
    if not values:
        return None

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def _percentile(p: float) -> float:
        if n == 1:
            return sorted_vals[0]
        # Linear interpolation (same as numpy's default method)
        idx = p / 100.0 * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])

    return PercentileStats(
        p50=_percentile(50),
        p95=_percentile(95),
        p99=_percentile(99),
        mean=sum(sorted_vals) / n,
        min=sorted_vals[0],
        max=sorted_vals[-1],
    )


def aggregate_results(
    results: list[RequestResult],
    total_duration_s: float,
) -> AggregatedMetrics:
    """Aggregate per-request results into summary statistics."""
    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    ttft_values = [r.ttft_ms for r in successful if r.ttft_ms is not None]
    e2e_values = [r.e2e_latency_ms for r in successful if r.e2e_latency_ms is not None]

    # Flatten all inter-token latencies across requests into one population
    tbt_values: list[float] = []
    for r in successful:
        tbt_values.extend(r.inter_token_latencies_ms)

    total_prompt = sum(r.prompt_tokens or 0 for r in successful)
    total_completion = sum(r.completion_tokens or 0 for r in successful)

    tokens_per_second = total_completion / total_duration_s if total_duration_s > 0 else 0.0
    requests_per_second = len(successful) / total_duration_s if total_duration_s > 0 else 0.0

    return AggregatedMetrics(
        total_requests=len(results),
        successful_requests=len(successful),
        failed_requests=len(failed),
        total_duration_s=total_duration_s,
        requests_per_second=requests_per_second,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        tokens_per_second=tokens_per_second,
        ttft=compute_percentiles(ttft_values),
        e2e_latency=compute_percentiles(e2e_values),
        tbt=compute_percentiles(tbt_values),
    )
