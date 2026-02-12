"""Prometheus metrics scraper for GPU/server-side observability.

Collects KV cache usage, active/queued requests, prefix cache hit rates,
and generation throughput from vLLM and SGLang ``/metrics`` endpoints.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GpuSample:
    """One scrape snapshot from the server's Prometheus endpoint."""

    timestamp: float  # relative monotonic seconds since scrape start
    kv_cache_usage: float | None = None
    active_requests: float | None = None
    queued_requests: float | None = None
    prefix_cache_hit_rate: float | None = None
    generation_throughput: float | None = None
    # Raw counters for delta computation in summarization
    prompt_tokens_total: float | None = None
    generation_tokens_total: float | None = None
    prefix_cache_hits_total: float | None = None
    prefix_cache_queries_total: float | None = None


@dataclass
class GpuTimeSeries:
    """Collection of samples plus metadata about the scrape session."""

    framework: str
    sample_interval_ms: int
    samples: list[GpuSample] = field(default_factory=list)
    scrape_errors: int = 0
    total_scrapes: int = 0


@dataclass
class GpuSummary:
    """Aggregated stats derived from a GpuTimeSeries."""

    kv_cache_usage_peak: float | None = None
    kv_cache_usage_mean: float | None = None
    active_requests_peak: float | None = None
    active_requests_mean: float | None = None
    prefix_cache_hit_rate: float | None = None
    generation_throughput: float | None = None
    total_samples: int = 0
    scrape_errors: int = 0


# ---------------------------------------------------------------------------
# Prometheus text parsing (lifted from poc/poc_03_deploy_vllm.py)
# ---------------------------------------------------------------------------


def parse_prometheus_text(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """Parse Prometheus text exposition format into a dict.

    Returns ``{metric_name: [(labels_dict, float_value), ...]}``.
    """
    metrics: dict[str, list[tuple[dict[str, str], float]]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Split "metric_name{labels} value" or "metric_name value"
        if "{" in line:
            name_part, rest = line.split("{", 1)
            labels_str, rest = rest.split("}", 1)
            labels: dict[str, str] = {}
            for pair in labels_str.split(","):
                pair = pair.strip()
                if "=" not in pair:
                    continue
                k, v = pair.split("=", 1)
                labels[k.strip()] = v.strip().strip('"')
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            name_part = parts[0]
            rest = parts[1]
            labels = {}

        name = name_part.strip()
        value_str = rest.strip().split()[0] if rest.strip() else "0"
        try:
            value = float(value_str)
        except ValueError:
            continue

        metrics.setdefault(name, []).append((labels, value))
    return metrics


def get_metric_value(
    metrics: dict[str, list[tuple[dict[str, str], float]]],
    name: str,
    labels: dict[str, str] | None = None,
) -> float | None:
    """Look up a single metric value, optionally filtering by labels."""
    entries = metrics.get(name, [])
    for entry_labels, value in entries:
        if labels is None or all(entry_labels.get(k) == v for k, v in labels.items()):
            return value
    return None


def get_metric_flexible(
    metrics: dict[str, list[tuple[dict[str, str], float]]],
    name: str,
    labels: dict[str, str] | None = None,
) -> float | None:
    """Try both ``:`` and ``_`` separators to handle vLLM version variations."""
    val = get_metric_value(metrics, name, labels)
    if val is not None:
        return val
    # Try swapping the first colon to underscore or vice versa
    if ":" in name:
        alt = name.replace(":", "_", 1)
    elif "_" in name:
        parts = name.split("_", 1)
        alt = parts[0] + ":" + parts[1]
    else:
        return None
    return get_metric_value(metrics, alt, labels)


# ---------------------------------------------------------------------------
# Framework metric name maps
# ---------------------------------------------------------------------------

VLLM_METRIC_MAP = {
    "kv_cache_usage": "vllm:kv_cache_usage_perc",
    "active_requests": "vllm:num_requests_running",
    "queued_requests": "vllm:num_requests_waiting",
    "prompt_tokens_total": "vllm:prompt_tokens_total",
    "generation_tokens_total": "vllm:generation_tokens_total",
    "prefix_cache_hits_total": "vllm:prefix_cache_hits_total",
    "prefix_cache_queries_total": "vllm:prefix_cache_queries_total",
}

SGLANG_METRIC_MAP = {
    "kv_cache_usage": "sglang:token_usage",
    "active_requests": "sglang:num_running_reqs",
    "queued_requests": "sglang:num_queue_reqs",
    "generation_throughput": "sglang:gen_throughput",
    "prefix_cache_hit_rate": "sglang:cache_hit_rate",
    "prompt_tokens_total": "sglang:prompt_tokens_total",
    "generation_tokens_total": "sglang:generation_tokens_total",
}

_METRIC_MAPS = {
    "vllm": VLLM_METRIC_MAP,
    "sglang": SGLANG_METRIC_MAP,
}


# ---------------------------------------------------------------------------
# GPU metrics scraper
# ---------------------------------------------------------------------------


class GpuMetricsScraper:
    """Background Prometheus scraper that collects server-side metrics."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        framework: str,
        sample_interval_ms: int = 100,
        on_sample: Callable[[GpuSample], None] | None = None,
    ) -> None:
        self._client = client
        self._framework = framework
        self._sample_interval_ms = sample_interval_ms
        self._on_sample = on_sample
        self._metric_map = _METRIC_MAPS.get(framework, VLLM_METRIC_MAP)
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._time_series = GpuTimeSeries(
            framework=framework,
            sample_interval_ms=sample_interval_ms,
        )
        self._t0: float = 0.0

    def start(self) -> None:
        """Launch the background scrape loop."""
        self._t0 = time.monotonic()
        self._task = asyncio.create_task(self._scrape_loop())

    async def stop(self) -> GpuTimeSeries:
        """Signal the scrape loop to stop, await completion, return data."""
        self._stop_event.set()
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        return self._time_series

    async def _scrape_loop(self) -> None:
        """Scrape at the configured interval until stop is signalled."""
        interval = self._sample_interval_ms / 1000.0
        while not self._stop_event.is_set():
            self._time_series.total_scrapes += 1
            try:
                sample = await self._scrape_once()
                self._time_series.samples.append(sample)
                if self._on_sample is not None:
                    self._on_sample(sample)
            except Exception:
                self._time_series.scrape_errors += 1
                logger.warning("GPU metrics scrape failed", exc_info=True)
            # Wait for interval or stop signal — whichever comes first
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=interval
                )
            except asyncio.TimeoutError:
                pass

    async def _scrape_once(self) -> GpuSample:
        """GET /metrics, parse, and extract framework-specific values."""
        resp = await self._client.get("/metrics")
        resp.raise_for_status()
        parsed = parse_prometheus_text(resp.text)
        timestamp = time.monotonic() - self._t0

        sample = GpuSample(timestamp=timestamp)
        for field_name, metric_name in self._metric_map.items():
            value = get_metric_flexible(parsed, metric_name)
            if value is not None:
                setattr(sample, field_name, value)
        return sample

    @staticmethod
    def summarize(ts: GpuTimeSeries) -> GpuSummary:
        """Compute aggregated stats from a time series.

        For gauges (KV cache, active requests): peak and mean.
        For vLLM prefix cache: first/last counter deltas.
        For vLLM throughput: generation_tokens_total delta / time delta.
        For SGLang: mean of native gauges.
        """
        if not ts.samples:
            return GpuSummary(
                total_samples=0,
                scrape_errors=ts.scrape_errors,
            )

        # --- Gauge aggregation ---
        kv_values = [s.kv_cache_usage for s in ts.samples if s.kv_cache_usage is not None]
        active_values = [s.active_requests for s in ts.samples if s.active_requests is not None]

        kv_peak = max(kv_values) if kv_values else None
        kv_mean = sum(kv_values) / len(kv_values) if kv_values else None
        active_peak = max(active_values) if active_values else None
        active_mean = sum(active_values) / len(active_values) if active_values else None

        # --- Prefix cache hit rate ---
        prefix_hit_rate: float | None = None
        if ts.framework == "sglang":
            # SGLang exposes a pre-computed gauge
            hit_rates = [s.prefix_cache_hit_rate for s in ts.samples if s.prefix_cache_hit_rate is not None]
            if hit_rates:
                prefix_hit_rate = sum(hit_rates) / len(hit_rates)
        else:
            # vLLM: compute from counter deltas (first vs last sample)
            first = ts.samples[0]
            last = ts.samples[-1]
            if (
                first.prefix_cache_hits_total is not None
                and last.prefix_cache_hits_total is not None
                and first.prefix_cache_queries_total is not None
                and last.prefix_cache_queries_total is not None
            ):
                hits_delta = last.prefix_cache_hits_total - first.prefix_cache_hits_total
                queries_delta = last.prefix_cache_queries_total - first.prefix_cache_queries_total
                if queries_delta > 0:
                    prefix_hit_rate = hits_delta / queries_delta

        # --- Generation throughput ---
        gen_throughput: float | None = None
        if ts.framework == "sglang":
            # SGLang exposes a native gauge (tok/s)
            tput_values = [s.generation_throughput for s in ts.samples if s.generation_throughput is not None]
            if tput_values:
                gen_throughput = sum(tput_values) / len(tput_values)
        else:
            # vLLM: compute from counter delta / time delta
            first = ts.samples[0]
            last = ts.samples[-1]
            if (
                first.generation_tokens_total is not None
                and last.generation_tokens_total is not None
            ):
                token_delta = last.generation_tokens_total - first.generation_tokens_total
                time_delta = last.timestamp - first.timestamp
                if time_delta > 0:
                    gen_throughput = token_delta / time_delta

        return GpuSummary(
            kv_cache_usage_peak=kv_peak,
            kv_cache_usage_mean=kv_mean,
            active_requests_peak=active_peak,
            active_requests_mean=active_mean,
            prefix_cache_hit_rate=prefix_hit_rate,
            generation_throughput=gen_throughput,
            total_samples=len(ts.samples),
            scrape_errors=ts.scrape_errors,
        )
