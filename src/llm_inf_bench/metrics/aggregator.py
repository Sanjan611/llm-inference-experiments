"""Percentile statistics and result aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass
class TurnStats:
    """Per-turn aggregated statistics."""

    turn_index: int
    request_count: int
    successful: int
    failed: int
    avg_prompt_tokens: float
    avg_completion_tokens: float
    ttft: PercentileStats | None
    e2e_latency: PercentileStats | None
    tbt: PercentileStats | None


@dataclass
class MultiTurnAggregatedMetrics:
    """Summary statistics for a multi-turn benchmark run."""

    overall: AggregatedMetrics
    per_turn: list[TurnStats] = field(default_factory=list)
    total_conversations: int = 0
    turns_per_conversation: int = 0


def aggregate_multi_turn_results(
    results: list[RequestResult],
    total_duration_s: float,
    turns: int,
) -> MultiTurnAggregatedMetrics:
    """Aggregate multi-turn results with per-turn breakdown.

    Groups results by ``turn_index`` and computes per-turn statistics
    in addition to the overall aggregate.
    """
    overall = aggregate_results(results, total_duration_s)

    # Group by turn_index
    by_turn: dict[int, list[RequestResult]] = {}
    for r in results:
        t = r.turn_index if r.turn_index is not None else 0
        by_turn.setdefault(t, []).append(r)

    per_turn: list[TurnStats] = []
    for turn_idx in range(turns):
        turn_results = by_turn.get(turn_idx, [])
        successful = [r for r in turn_results if r.error is None]
        failed = [r for r in turn_results if r.error is not None]

        ttft_values = [r.ttft_ms for r in successful if r.ttft_ms is not None]
        e2e_values = [r.e2e_latency_ms for r in successful if r.e2e_latency_ms is not None]
        tbt_values: list[float] = []
        for r in successful:
            tbt_values.extend(r.inter_token_latencies_ms)

        prompt_toks = [r.prompt_tokens for r in successful if r.prompt_tokens is not None]
        compl_toks = [r.completion_tokens for r in successful if r.completion_tokens is not None]

        avg_prompt = sum(prompt_toks) / len(prompt_toks) if prompt_toks else 0.0
        avg_compl = sum(compl_toks) / len(compl_toks) if compl_toks else 0.0

        per_turn.append(
            TurnStats(
                turn_index=turn_idx,
                request_count=len(turn_results),
                successful=len(successful),
                failed=len(failed),
                avg_prompt_tokens=avg_prompt,
                avg_completion_tokens=avg_compl,
                ttft=compute_percentiles(ttft_values),
                e2e_latency=compute_percentiles(e2e_values),
                tbt=compute_percentiles(tbt_values),
            )
        )

    # Determine conversation count from results
    conv_indices = {r.conversation_index for r in results if r.conversation_index is not None}
    total_conversations = len(conv_indices) if conv_indices else 0

    return MultiTurnAggregatedMetrics(
        overall=overall,
        per_turn=per_turn,
        total_conversations=total_conversations,
        turns_per_conversation=turns,
    )
