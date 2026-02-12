"""Tests for percentile computation and result aggregation."""

from __future__ import annotations

import math

from llm_inf_bench.metrics.aggregator import (
    AggregatedMetrics,
    aggregate_results,
    compute_percentiles,
)
from llm_inf_bench.metrics.collector import RequestResult


class TestComputePercentiles:
    def test_empty_list(self):
        assert compute_percentiles([]) is None

    def test_single_value(self):
        stats = compute_percentiles([42.0])
        assert stats is not None
        assert stats.p50 == 42.0
        assert stats.p95 == 42.0
        assert stats.p99 == 42.0
        assert stats.mean == 42.0
        assert stats.min == 42.0
        assert stats.max == 42.0

    def test_known_values(self):
        # 1..100 should give p50=50.5, p95=95.05, p99=99.01
        values = [float(i) for i in range(1, 101)]
        stats = compute_percentiles(values)
        assert stats is not None
        assert stats.min == 1.0
        assert stats.max == 100.0
        assert stats.mean == 50.5
        assert math.isclose(stats.p50, 50.5, rel_tol=1e-6)
        assert math.isclose(stats.p95, 95.05, rel_tol=1e-3)

    def test_two_values(self):
        stats = compute_percentiles([10.0, 20.0])
        assert stats is not None
        assert stats.min == 10.0
        assert stats.max == 20.0
        assert stats.mean == 15.0
        assert stats.p50 == 15.0  # midpoint via linear interpolation

    def test_unsorted_input(self):
        stats = compute_percentiles([30.0, 10.0, 20.0])
        assert stats is not None
        assert stats.min == 10.0
        assert stats.max == 30.0


class TestAggregateResults:
    def test_all_successful(self):
        results = [
            RequestResult(
                request_index=i,
                ttft_ms=50.0,
                e2e_latency_ms=300.0,
                inter_token_latencies_ms=[10.0, 12.0, 11.0],
                prompt_tokens=20,
                completion_tokens=30,
            )
            for i in range(10)
        ]
        agg = aggregate_results(results, total_duration_s=10.0)
        assert agg.total_requests == 10
        assert agg.successful_requests == 10
        assert agg.failed_requests == 0
        assert agg.total_prompt_tokens == 200
        assert agg.total_completion_tokens == 300
        assert agg.tokens_per_second == 30.0  # 300 / 10s
        assert agg.requests_per_second == 1.0
        assert agg.ttft is not None
        assert agg.e2e_latency is not None
        assert agg.tbt is not None

    def test_mixed_success_failure(self):
        results = [
            RequestResult(
                request_index=0,
                ttft_ms=50.0,
                e2e_latency_ms=300.0,
                prompt_tokens=20,
                completion_tokens=30,
            ),
            RequestResult(
                request_index=1,
                error="connection timeout",
                e2e_latency_ms=5000.0,
            ),
            RequestResult(
                request_index=2,
                ttft_ms=60.0,
                e2e_latency_ms=350.0,
                prompt_tokens=25,
                completion_tokens=35,
            ),
        ]
        agg = aggregate_results(results, total_duration_s=6.0)
        assert agg.total_requests == 3
        assert agg.successful_requests == 2
        assert agg.failed_requests == 1
        assert agg.total_prompt_tokens == 45
        assert agg.total_completion_tokens == 65

    def test_all_failed(self):
        results = [
            RequestResult(request_index=i, error="fail")
            for i in range(5)
        ]
        agg = aggregate_results(results, total_duration_s=5.0)
        assert agg.total_requests == 5
        assert agg.successful_requests == 0
        assert agg.failed_requests == 5
        assert agg.ttft is None
        assert agg.e2e_latency is None
        assert agg.tbt is None
        assert agg.tokens_per_second == 0.0

    def test_empty_results(self):
        agg = aggregate_results([], total_duration_s=0.0)
        assert agg.total_requests == 0
        assert agg.ttft is None

    def test_tbt_flattened_across_requests(self):
        """Inter-token latencies from all requests form one population."""
        results = [
            RequestResult(
                request_index=0,
                inter_token_latencies_ms=[10.0, 20.0],
                ttft_ms=50.0,
                e2e_latency_ms=300.0,
            ),
            RequestResult(
                request_index=1,
                inter_token_latencies_ms=[30.0],
                ttft_ms=60.0,
                e2e_latency_ms=350.0,
            ),
        ]
        agg = aggregate_results(results, total_duration_s=1.0)
        assert agg.tbt is not None
        assert agg.tbt.mean == 20.0  # (10 + 20 + 30) / 3

    def test_zero_duration(self):
        results = [
            RequestResult(request_index=0, completion_tokens=10)
        ]
        agg = aggregate_results(results, total_duration_s=0.0)
        assert agg.tokens_per_second == 0.0
        assert agg.requests_per_second == 0.0

    def test_gpu_summary_defaults_to_none(self):
        agg = aggregate_results([], total_duration_s=0.0)
        assert agg.gpu_summary is None

    def test_gpu_summary_can_be_set(self):
        from llm_inf_bench.metrics.gpu import GpuSummary

        agg = aggregate_results([], total_duration_s=0.0)
        gs = GpuSummary(
            kv_cache_usage_peak=0.5,
            kv_cache_usage_mean=0.3,
            total_samples=10,
        )
        agg.gpu_summary = gs
        assert agg.gpu_summary is gs
        assert agg.gpu_summary.kv_cache_usage_peak == 0.5
