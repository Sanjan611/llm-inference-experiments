"""Tests for GPU metrics scraper and Prometheus parsing."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from llm_inf_bench.metrics.gpu import (
    SGLANG_METRIC_MAP,
    VLLM_METRIC_MAP,
    GpuMetricsScraper,
    GpuSample,
    GpuSummary,
    GpuTimeSeries,
    get_metric_flexible,
    get_metric_value,
    parse_prometheus_text,
)


# ---------------------------------------------------------------------------
# Prometheus text parsing
# ---------------------------------------------------------------------------

SAMPLE_PROMETHEUS_TEXT = """\
# HELP vllm:kv_cache_usage_perc KV cache usage percentage
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc 0.234
# HELP vllm:num_requests_running Number of running requests
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 3
vllm:num_requests_waiting 1
vllm:prompt_tokens_total 1500
vllm:generation_tokens_total 800
vllm:prefix_cache_hits_total{model="test"} 50
vllm:prefix_cache_queries_total{model="test"} 100
"""


class TestParsePrometheusText:
    def test_parses_gauges(self):
        metrics = parse_prometheus_text(SAMPLE_PROMETHEUS_TEXT)
        assert "vllm:kv_cache_usage_perc" in metrics
        assert metrics["vllm:kv_cache_usage_perc"][0] == ({}, 0.234)

    def test_parses_labeled_metrics(self):
        metrics = parse_prometheus_text(SAMPLE_PROMETHEUS_TEXT)
        entries = metrics.get("vllm:prefix_cache_hits_total", [])
        assert len(entries) == 1
        labels, value = entries[0]
        assert labels == {"model": "test"}
        assert value == 50.0

    def test_ignores_comments_and_help(self):
        metrics = parse_prometheus_text(SAMPLE_PROMETHEUS_TEXT)
        # No key should start with '#'
        for key in metrics:
            assert not key.startswith("#")

    def test_empty_input(self):
        assert parse_prometheus_text("") == {}

    def test_comments_only(self):
        text = "# HELP foo\n# TYPE foo gauge\n"
        assert parse_prometheus_text(text) == {}

    def test_multiple_label_pairs(self):
        text = 'my_metric{host="a",port="8080"} 42.5\n'
        metrics = parse_prometheus_text(text)
        labels, value = metrics["my_metric"][0]
        assert labels == {"host": "a", "port": "8080"}
        assert value == 42.5

    def test_colon_and_underscore_separators(self):
        text = "vllm:prompt_tokens_total 100\nvllm_generation_tokens_total 200\n"
        metrics = parse_prometheus_text(text)
        assert "vllm:prompt_tokens_total" in metrics
        assert "vllm_generation_tokens_total" in metrics

    def test_malformed_lines_skipped(self):
        text = "good_metric 42\nbadline\n\n  \n"
        metrics = parse_prometheus_text(text)
        assert "good_metric" in metrics
        assert len(metrics) == 1


class TestGetMetricValue:
    def test_simple_lookup(self):
        metrics = parse_prometheus_text(SAMPLE_PROMETHEUS_TEXT)
        val = get_metric_value(metrics, "vllm:kv_cache_usage_perc")
        assert val == 0.234

    def test_label_filter(self):
        metrics = parse_prometheus_text(SAMPLE_PROMETHEUS_TEXT)
        val = get_metric_value(
            metrics, "vllm:prefix_cache_hits_total", {"model": "test"}
        )
        assert val == 50.0

    def test_label_mismatch_returns_none(self):
        metrics = parse_prometheus_text(SAMPLE_PROMETHEUS_TEXT)
        val = get_metric_value(
            metrics, "vllm:prefix_cache_hits_total", {"model": "other"}
        )
        assert val is None

    def test_missing_metric_returns_none(self):
        metrics = parse_prometheus_text(SAMPLE_PROMETHEUS_TEXT)
        assert get_metric_value(metrics, "nonexistent") is None


class TestGetMetricFlexible:
    def test_colon_to_underscore(self):
        text = "vllm_kv_cache_usage_perc 0.5\n"
        metrics = parse_prometheus_text(text)
        # Search with colon separator — should try underscore fallback
        val = get_metric_flexible(metrics, "vllm:kv_cache_usage_perc")
        assert val == 0.5

    def test_underscore_to_colon(self):
        text = "vllm:kv_cache_usage_perc 0.5\n"
        metrics = parse_prometheus_text(text)
        # Search with underscore separator — should try colon fallback
        val = get_metric_flexible(metrics, "vllm_kv_cache_usage_perc")
        assert val == 0.5

    def test_direct_hit_no_fallback_needed(self):
        text = "vllm:kv_cache_usage_perc 0.5\n"
        metrics = parse_prometheus_text(text)
        val = get_metric_flexible(metrics, "vllm:kv_cache_usage_perc")
        assert val == 0.5

    def test_missing_returns_none(self):
        metrics = parse_prometheus_text("")
        assert get_metric_flexible(metrics, "vllm:foo") is None

    def test_no_separator_returns_none(self):
        metrics = parse_prometheus_text("")
        assert get_metric_flexible(metrics, "noseparator") is None

    def test_with_labels(self):
        text = 'vllm_prefix_cache_hits_total{model="m"} 42\n'
        metrics = parse_prometheus_text(text)
        val = get_metric_flexible(
            metrics, "vllm:prefix_cache_hits_total", {"model": "m"}
        )
        assert val == 42.0


# ---------------------------------------------------------------------------
# GpuMetricsScraper
# ---------------------------------------------------------------------------

VLLM_METRICS_RESPONSE = """\
vllm:kv_cache_usage_perc 0.234
vllm:num_requests_running 1
vllm:num_requests_waiting 0
vllm:prompt_tokens_total 500
vllm:generation_tokens_total 200
vllm:prefix_cache_hits_total 10
vllm:prefix_cache_queries_total 20
"""

SGLANG_METRICS_RESPONSE = """\
sglang:token_usage 0.15
sglang:num_running_reqs 2
sglang:num_queue_reqs 1
sglang:gen_throughput 142.3
sglang:cache_hit_rate 0.673
sglang:prompt_tokens_total 600
sglang:generation_tokens_total 300
"""


def _mock_client(response_text: str, status_code: int = 200) -> httpx.AsyncClient:
    """Create a mock httpx.AsyncClient that returns the given text for GET /metrics."""
    client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.text = response_text
    mock_response.status_code = status_code
    mock_response.raise_for_status = MagicMock()
    if status_code >= 400:
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=mock_response
        )
    client.get = AsyncMock(return_value=mock_response)
    return client


class TestGpuMetricsScraper:
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        client = _mock_client(VLLM_METRICS_RESPONSE)
        scraper = GpuMetricsScraper(client, "vllm", sample_interval_ms=50)
        scraper.start()
        await asyncio.sleep(0.15)  # Allow a few scrapes
        ts = await scraper.stop()

        assert isinstance(ts, GpuTimeSeries)
        assert ts.framework == "vllm"
        assert ts.sample_interval_ms == 50
        assert len(ts.samples) > 0
        assert ts.scrape_errors == 0

    @pytest.mark.asyncio
    async def test_vllm_extraction(self):
        client = _mock_client(VLLM_METRICS_RESPONSE)
        scraper = GpuMetricsScraper(client, "vllm", sample_interval_ms=50)
        scraper.start()
        await asyncio.sleep(0.1)
        ts = await scraper.stop()

        sample = ts.samples[0]
        assert sample.kv_cache_usage == 0.234
        assert sample.active_requests == 1.0
        assert sample.queued_requests == 0.0
        assert sample.prompt_tokens_total == 500.0
        assert sample.generation_tokens_total == 200.0
        assert sample.prefix_cache_hits_total == 10.0
        assert sample.prefix_cache_queries_total == 20.0
        # First sample has no previous sample to diff, so no computed throughput
        assert sample.generation_throughput is None
        assert sample.prefix_cache_hit_rate is None

        # Subsequent samples should have throughput computed from counter deltas
        if len(ts.samples) > 1:
            assert ts.samples[1].generation_throughput is not None

    @pytest.mark.asyncio
    async def test_sglang_extraction(self):
        client = _mock_client(SGLANG_METRICS_RESPONSE)
        scraper = GpuMetricsScraper(client, "sglang", sample_interval_ms=50)
        scraper.start()
        await asyncio.sleep(0.1)
        ts = await scraper.stop()

        sample = ts.samples[0]
        assert sample.kv_cache_usage == 0.15
        assert sample.active_requests == 2.0
        assert sample.queued_requests == 1.0
        assert sample.generation_throughput == 142.3
        assert sample.prefix_cache_hit_rate == 0.673

    @pytest.mark.asyncio
    async def test_scrape_failure_handling(self):
        client = _mock_client("", status_code=500)
        scraper = GpuMetricsScraper(client, "vllm", sample_interval_ms=50)
        scraper.start()
        await asyncio.sleep(0.15)
        ts = await scraper.stop()

        assert ts.scrape_errors > 0
        assert len(ts.samples) == 0  # No successful samples

    @pytest.mark.asyncio
    async def test_immediate_stop(self):
        client = _mock_client(VLLM_METRICS_RESPONSE)
        scraper = GpuMetricsScraper(client, "vllm", sample_interval_ms=1000)
        scraper.start()
        # Stop immediately — should not hang
        ts = await scraper.stop()
        assert isinstance(ts, GpuTimeSeries)

    @pytest.mark.asyncio
    async def test_vllm_throughput_computed_from_deltas(self):
        """Second+ vLLM samples should have throughput derived from counter deltas."""
        # Use varying generation_tokens_total to produce a meaningful delta
        call_count = 0
        base_tokens = 200

        def make_response() -> MagicMock:
            nonlocal call_count
            call_count += 1
            tokens = base_tokens + (call_count - 1) * 100
            text = (
                f"vllm:kv_cache_usage_perc 0.1\n"
                f"vllm:generation_tokens_total {tokens}\n"
            )
            resp = MagicMock(spec=httpx.Response)
            resp.text = text
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            return resp

        client = AsyncMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(side_effect=lambda *a, **kw: make_response())

        scraper = GpuMetricsScraper(client, "vllm", sample_interval_ms=50)
        scraper.start()
        await asyncio.sleep(0.2)  # Allow several scrapes
        ts = await scraper.stop()

        assert len(ts.samples) >= 2, f"Expected >=2 samples, got {len(ts.samples)}"
        # First sample: no previous → no throughput
        assert ts.samples[0].generation_throughput is None
        # Second+ samples: throughput computed from counter deltas
        for s in ts.samples[1:]:
            assert s.generation_throughput is not None
            assert s.generation_throughput > 0

    @pytest.mark.asyncio
    async def test_sglang_native_throughput_not_overridden(self):
        """SGLang's native gen_throughput gauge should not be overridden by delta logic."""
        client = _mock_client(SGLANG_METRICS_RESPONSE)
        scraper = GpuMetricsScraper(client, "sglang", sample_interval_ms=50)
        scraper.start()
        await asyncio.sleep(0.15)
        ts = await scraper.stop()

        # All samples should have the native gauge value, not a computed one
        for s in ts.samples:
            assert s.generation_throughput == 142.3


# ---------------------------------------------------------------------------
# GpuMetricsScraper.summarize()
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_empty_samples(self):
        ts = GpuTimeSeries(framework="vllm", sample_interval_ms=100, scrape_errors=2)
        summary = GpuMetricsScraper.summarize(ts)
        assert summary.total_samples == 0
        assert summary.scrape_errors == 2
        assert summary.kv_cache_usage_peak is None
        assert summary.generation_throughput is None

    def test_peak_and_mean(self):
        samples = [
            GpuSample(timestamp=0.0, kv_cache_usage=0.1, active_requests=1),
            GpuSample(timestamp=1.0, kv_cache_usage=0.3, active_requests=3),
            GpuSample(timestamp=2.0, kv_cache_usage=0.2, active_requests=2),
        ]
        ts = GpuTimeSeries(framework="vllm", sample_interval_ms=1000, samples=samples)
        summary = GpuMetricsScraper.summarize(ts)

        assert summary.kv_cache_usage_peak == 0.3
        assert abs(summary.kv_cache_usage_mean - 0.2) < 1e-9
        assert summary.active_requests_peak == 3.0
        assert abs(summary.active_requests_mean - 2.0) < 1e-9
        assert summary.total_samples == 3

    def test_vllm_prefix_cache_deltas(self):
        samples = [
            GpuSample(
                timestamp=0.0,
                prefix_cache_hits_total=10,
                prefix_cache_queries_total=20,
            ),
            GpuSample(
                timestamp=1.0,
                prefix_cache_hits_total=30,
                prefix_cache_queries_total=50,
            ),
        ]
        ts = GpuTimeSeries(framework="vllm", sample_interval_ms=1000, samples=samples)
        summary = GpuMetricsScraper.summarize(ts)
        # hits_delta = 30 - 10 = 20, queries_delta = 50 - 20 = 30
        assert abs(summary.prefix_cache_hit_rate - 20 / 30) < 1e-9

    def test_vllm_throughput_from_counters(self):
        samples = [
            GpuSample(timestamp=0.0, generation_tokens_total=100),
            GpuSample(timestamp=2.0, generation_tokens_total=500),
        ]
        ts = GpuTimeSeries(framework="vllm", sample_interval_ms=1000, samples=samples)
        summary = GpuMetricsScraper.summarize(ts)
        # (500 - 100) / (2.0 - 0.0) = 200 tok/s
        assert abs(summary.generation_throughput - 200.0) < 1e-9

    def test_vllm_no_prefix_cache_data(self):
        samples = [GpuSample(timestamp=0.0, kv_cache_usage=0.5)]
        ts = GpuTimeSeries(framework="vllm", sample_interval_ms=100, samples=samples)
        summary = GpuMetricsScraper.summarize(ts)
        assert summary.prefix_cache_hit_rate is None

    def test_vllm_zero_queries_delta(self):
        samples = [
            GpuSample(timestamp=0.0, prefix_cache_hits_total=5, prefix_cache_queries_total=10),
            GpuSample(timestamp=1.0, prefix_cache_hits_total=5, prefix_cache_queries_total=10),
        ]
        ts = GpuTimeSeries(framework="vllm", sample_interval_ms=1000, samples=samples)
        summary = GpuMetricsScraper.summarize(ts)
        # Zero delta — no hit rate
        assert summary.prefix_cache_hit_rate is None

    def test_sglang_gauge_means(self):
        samples = [
            GpuSample(
                timestamp=0.0,
                prefix_cache_hit_rate=0.5,
                generation_throughput=100.0,
            ),
            GpuSample(
                timestamp=1.0,
                prefix_cache_hit_rate=0.7,
                generation_throughput=200.0,
            ),
        ]
        ts = GpuTimeSeries(framework="sglang", sample_interval_ms=1000, samples=samples)
        summary = GpuMetricsScraper.summarize(ts)
        assert abs(summary.prefix_cache_hit_rate - 0.6) < 1e-9
        assert abs(summary.generation_throughput - 150.0) < 1e-9

    def test_single_sample(self):
        samples = [
            GpuSample(
                timestamp=0.0,
                kv_cache_usage=0.5,
                active_requests=2,
                generation_tokens_total=100,
            ),
        ]
        ts = GpuTimeSeries(framework="vllm", sample_interval_ms=100, samples=samples)
        summary = GpuMetricsScraper.summarize(ts)
        assert summary.kv_cache_usage_peak == 0.5
        assert summary.kv_cache_usage_mean == 0.5
        assert summary.active_requests_peak == 2.0
        # Single sample: time delta = 0, so no throughput
        assert summary.generation_throughput is None
