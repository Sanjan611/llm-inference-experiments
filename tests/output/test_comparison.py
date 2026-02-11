"""Tests for print_comparison rendering."""

from __future__ import annotations

from pathlib import Path

from llm_inf_bench.metrics.storage import StoredResult
from llm_inf_bench.output.summary import _fmt_delta, print_comparison


def _make_stored(
    run_id: str = "run-a",
    framework: str = "vllm",
    model: str = "test-model",
    gpu: str = "A100-80GB",
    total_requests: int = 10,
    successful_requests: int = 10,
    failed_requests: int = 0,
    total_duration_s: float = 5.0,
    requests_per_second: float = 2.0,
    tokens_per_second: float = 100.0,
    ttft: dict | None = None,
    e2e_latency: dict | None = None,
    tbt: dict | None = None,
) -> StoredResult:
    default_stats = {"p50": 50.0, "p95": 90.0, "p99": 99.0, "mean": 55.0, "min": 10.0, "max": 120.0}
    return StoredResult(
        run_id=run_id,
        status="completed",
        experiment={
            "framework": framework,
            "model": {"name": model},
            "infrastructure": {"gpu_type": gpu},
        },
        metadata={"started_at": "2025-01-01T00:00:00"},
        summary={
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "total_duration_s": total_duration_s,
            "requests_per_second": requests_per_second,
            "total_prompt_tokens": 200,
            "total_completion_tokens": 500,
            "tokens_per_second": tokens_per_second,
            "ttft": ttft or default_stats,
            "e2e_latency": e2e_latency or default_stats,
            "tbt": tbt or default_stats,
        },
        requests=[],
        file_path=Path("/tmp/fake.json"),
    )


class TestFmtDelta:
    def test_negative_delta_lower_is_better(self):
        result = _fmt_delta(100.0, 80.0, lower_is_better=True)
        assert "-20.0" in result
        assert "-20.0%" in result
        assert "green" in result

    def test_positive_delta_lower_is_better(self):
        result = _fmt_delta(80.0, 100.0, lower_is_better=True)
        assert "+20.0" in result
        assert "red" in result

    def test_higher_is_better(self):
        result = _fmt_delta(100.0, 120.0, lower_is_better=False)
        assert "green" in result

    def test_no_change(self):
        result = _fmt_delta(100.0, 100.0)
        assert "0.0" in result
        # No colour when diff is zero
        assert "green" not in result
        assert "red" not in result


class TestPrintComparison:
    def test_renders_without_error(self):
        a = _make_stored(run_id="run-a", framework="vllm")
        b = _make_stored(run_id="run-b", framework="sglang")
        # Should not raise
        print_comparison(a, b)

    def test_handles_none_percentile_stats(self):
        a = _make_stored(run_id="run-a", ttft=None, e2e_latency=None, tbt=None)
        b = _make_stored(run_id="run-b", ttft=None, e2e_latency=None, tbt=None)
        # Should not raise
        print_comparison(a, b)

    def test_mixed_none_stats(self):
        a = _make_stored(run_id="run-a")
        b = _make_stored(run_id="run-b", ttft=None)
        # Should not raise
        print_comparison(a, b)
