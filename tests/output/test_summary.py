"""Smoke tests for summary rendering."""

from __future__ import annotations

from llm_inf_bench.metrics.aggregator import AggregatedMetrics, PercentileStats
from llm_inf_bench.metrics.gpu import GpuSummary
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

    def test_with_gpu_summary(self):
        metrics = AggregatedMetrics(
            total_requests=10,
            successful_requests=10,
            failed_requests=0,
            total_duration_s=5.0,
            requests_per_second=2.0,
            total_prompt_tokens=200,
            total_completion_tokens=500,
            tokens_per_second=100.0,
            ttft=PercentileStats(p50=42.0, p95=78.0, p99=112.0, mean=55.0, min=10.0, max=200.0),
            e2e_latency=PercentileStats(p50=298.0, p95=412.0, p99=523.0, mean=310.0, min=100.0, max=600.0),
            tbt=PercentileStats(p50=12.0, p95=18.0, p99=24.0, mean=13.0, min=5.0, max=30.0),
            gpu_summary=GpuSummary(
                kv_cache_usage_peak=0.234,
                kv_cache_usage_mean=0.187,
                active_requests_peak=1,
                active_requests_mean=1.0,
                prefix_cache_hit_rate=0.673,
                generation_throughput=142.3,
                total_samples=52,
                scrape_errors=0,
            ),
        )
        # Should not raise
        print_summary(metrics)

    def test_gpu_summary_none_no_section(self):
        metrics = AggregatedMetrics(
            total_requests=5,
            successful_requests=5,
            failed_requests=0,
            total_duration_s=2.0,
            requests_per_second=2.5,
            total_prompt_tokens=100,
            total_completion_tokens=200,
            tokens_per_second=100.0,
            ttft=None,
            e2e_latency=None,
            tbt=None,
            gpu_summary=None,
        )
        # Should not raise — no GPU section rendered
        print_summary(metrics)

    def test_gpu_summary_zero_samples_no_section(self):
        metrics = AggregatedMetrics(
            total_requests=5,
            successful_requests=5,
            failed_requests=0,
            total_duration_s=2.0,
            requests_per_second=2.5,
            total_prompt_tokens=100,
            total_completion_tokens=200,
            tokens_per_second=100.0,
            ttft=None,
            e2e_latency=None,
            tbt=None,
            gpu_summary=GpuSummary(total_samples=0, scrape_errors=5),
        )
        # Should not raise — zero samples means no section
        print_summary(metrics)

    def test_gpu_summary_partial_data(self):
        """GPU summary with only some fields populated."""
        metrics = AggregatedMetrics(
            total_requests=5,
            successful_requests=5,
            failed_requests=0,
            total_duration_s=2.0,
            requests_per_second=2.5,
            total_prompt_tokens=100,
            total_completion_tokens=200,
            tokens_per_second=100.0,
            ttft=None,
            e2e_latency=None,
            tbt=None,
            gpu_summary=GpuSummary(
                kv_cache_usage_peak=0.3,
                kv_cache_usage_mean=0.2,
                total_samples=10,
                scrape_errors=0,
                # No prefix cache, no throughput, no active requests
            ),
        )
        # Should not raise
        print_summary(metrics)
