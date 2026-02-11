"""Smoke tests for summary rendering."""

from __future__ import annotations

from llm_inf_bench.metrics.aggregator import AggregatedMetrics, PercentileStats
from llm_inf_bench.output.summary import _fmt_percentiles, print_summary


class TestFmtPercentiles:
    def test_formats_stats(self):
        stats = PercentileStats(p50=42.0, p95=78.0, p99=112.0, mean=55.0, min=10.0, max=200.0)
        result = _fmt_percentiles(stats)
        assert "p50=42ms" in result
        assert "p95=78ms" in result
        assert "p99=112ms" in result

    def test_none_returns_na(self):
        assert _fmt_percentiles(None) == "n/a"


class TestPrintSummary:
    def test_valid_metrics(self):
        metrics = AggregatedMetrics(
            total_requests=100,
            successful_requests=98,
            failed_requests=2,
            total_duration_s=42.3,
            requests_per_second=2.32,
            total_prompt_tokens=2000,
            total_completion_tokens=5000,
            tokens_per_second=118.2,
            ttft=PercentileStats(p50=42.0, p95=78.0, p99=112.0, mean=55.0, min=10.0, max=200.0),
            e2e_latency=PercentileStats(p50=298.0, p95=412.0, p99=523.0, mean=310.0, min=100.0, max=600.0),
            tbt=PercentileStats(p50=12.0, p95=18.0, p99=24.0, mean=13.0, min=5.0, max=30.0),
        )
        # Should not raise
        print_summary(metrics)

    def test_degenerate_metrics(self):
        metrics = AggregatedMetrics(
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            total_duration_s=0.0,
            requests_per_second=0.0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            tokens_per_second=0.0,
            ttft=None,
            e2e_latency=None,
            tbt=None,
        )
        # Should not raise
        print_summary(metrics)

    def test_all_failures(self):
        metrics = AggregatedMetrics(
            total_requests=10,
            successful_requests=0,
            failed_requests=10,
            total_duration_s=5.0,
            requests_per_second=0.0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            tokens_per_second=0.0,
            ttft=None,
            e2e_latency=None,
            tbt=None,
        )
        print_summary(metrics)
