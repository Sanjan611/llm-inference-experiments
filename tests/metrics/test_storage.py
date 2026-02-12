"""Tests for JSON result storage."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from llm_inf_bench.config.schema import ExperimentConfig
from llm_inf_bench.metrics.aggregator import AggregatedMetrics, PercentileStats, aggregate_results
from llm_inf_bench.metrics.collector import RequestResult, RunMetadata
from llm_inf_bench.metrics.storage import (
    generate_run_id,
    list_results,
    load_result,
    reconstruct_aggregated_metrics,
    save_results,
)


class TestGenerateRunId:
    def test_format(self):
        now = datetime(2025, 3, 15, 14, 30, 22)
        run_id = generate_run_id("baseline-vllm", now=now)
        assert run_id == "baseline-vllm-20250315-143022"

    def test_deterministic(self):
        now = datetime(2025, 1, 1, 0, 0, 0)
        a = generate_run_id("test", now=now)
        b = generate_run_id("test", now=now)
        assert a == b

    def test_default_uses_current_time(self):
        run_id = generate_run_id("exp")
        assert run_id.startswith("exp-")
        assert len(run_id) > len("exp-")


class TestSaveResults:
    def _make_config(self) -> ExperimentConfig:
        return ExperimentConfig(
            name="test",
            model={"name": "test-model"},
            framework="vllm",
            infrastructure={"gpu_type": "A100-80GB"},
            workload={
                "type": "single",
                "requests": {"source": "prompts/test.jsonl", "count": 10},
            },
        )

    def _make_results(self) -> list[RequestResult]:
        return [
            RequestResult(
                request_index=0,
                ttft_ms=50.0,
                e2e_latency_ms=300.0,
                inter_token_latencies_ms=[10.0, 12.0],
                prompt_tokens=20,
                completion_tokens=30,
                started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                finished_at=datetime(2025, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            ),
        ]

    def test_creates_directory_and_file(self, tmp_path):
        output_dir = tmp_path / "results" / "nested"
        config = self._make_config()
        results = self._make_results()
        aggregated = aggregate_results(results, total_duration_s=1.0)
        metadata = RunMetadata(
            run_id="test-20250101-000000",
            experiment_name="test",
            status="completed",
        )

        path = save_results(output_dir, metadata, config, results, aggregated)

        assert path.exists()
        assert path.name == "test-20250101-000000.json"
        assert output_dir.exists()

    def test_json_structure(self, tmp_path):
        config = self._make_config()
        results = self._make_results()
        aggregated = aggregate_results(results, total_duration_s=1.0)
        metadata = RunMetadata(
            run_id="test-run",
            experiment_name="test",
            started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            status="completed",
        )

        path = save_results(tmp_path, metadata, config, results, aggregated)
        data = json.loads(path.read_text())

        assert data["run_id"] == "test-run"
        assert data["status"] == "completed"
        assert "experiment" in data
        assert data["experiment"]["name"] == "test"
        assert "metadata" in data
        assert "summary" in data
        assert "requests" in data
        assert len(data["requests"]) == 1
        assert data["requests"][0]["ttft_ms"] == 50.0

    def test_summary_fields(self, tmp_path):
        config = self._make_config()
        results = self._make_results()
        aggregated = aggregate_results(results, total_duration_s=1.0)
        metadata = RunMetadata(run_id="s", experiment_name="t", status="completed")

        path = save_results(tmp_path, metadata, config, results, aggregated)
        data = json.loads(path.read_text())

        summary = data["summary"]
        assert summary["total_requests"] == 1
        assert summary["successful_requests"] == 1
        assert summary["failed_requests"] == 0
        assert "ttft" in summary
        assert "e2e_latency" in summary


# ---------------------------------------------------------------------------
# Helpers for result-loading tests
# ---------------------------------------------------------------------------

def _write_result_file(directory, run_id, *, status="completed", started_at="2025-01-01T00:00:00"):
    """Write a minimal valid result JSON file."""
    data = {
        "run_id": run_id,
        "status": status,
        "experiment": {"name": "test", "framework": "vllm", "model": {"name": "m"}},
        "metadata": {"started_at": started_at},
        "summary": {
            "total_requests": 1,
            "successful_requests": 1,
            "failed_requests": 0,
            "total_duration_s": 1.0,
            "requests_per_second": 1.0,
            "total_prompt_tokens": 10,
            "total_completion_tokens": 20,
            "tokens_per_second": 20.0,
            "ttft": {"p50": 50.0, "p95": 90.0, "p99": 99.0, "mean": 55.0, "min": 10.0, "max": 120.0},
            "e2e_latency": {"p50": 300.0, "p95": 400.0, "p99": 500.0, "mean": 310.0, "min": 100.0, "max": 600.0},
            "tbt": {"p50": 12.0, "p95": 18.0, "p99": 24.0, "mean": 13.0, "min": 5.0, "max": 30.0},
        },
        "requests": [],
    }
    fp = directory / f"{run_id}.json"
    fp.write_text(json.dumps(data))
    return fp


class TestListResults:
    def test_empty_dir(self, tmp_path):
        results = list_results(tmp_path)
        assert results == []

    def test_nonexistent_dir(self, tmp_path):
        results = list_results(tmp_path / "nope")
        assert results == []

    def test_multiple_files_sorted_newest_first(self, tmp_path):
        _write_result_file(tmp_path, "run-a", started_at="2025-01-01T00:00:00")
        _write_result_file(tmp_path, "run-b", started_at="2025-01-02T00:00:00")
        results = list_results(tmp_path)
        assert len(results) == 2
        assert results[0].run_id == "run-b"  # newer first
        assert results[1].run_id == "run-a"

    def test_skips_malformed_files(self, tmp_path):
        _write_result_file(tmp_path, "good-run")
        (tmp_path / "bad.json").write_text("not json at all {{{")
        results = list_results(tmp_path)
        assert len(results) == 1
        assert results[0].run_id == "good-run"


class TestLoadResult:
    def test_exact_match(self, tmp_path):
        _write_result_file(tmp_path, "my-run-20250101-120000")
        result = load_result(tmp_path, "my-run-20250101-120000")
        assert result.run_id == "my-run-20250101-120000"

    def test_partial_match(self, tmp_path):
        _write_result_file(tmp_path, "baseline-vllm-20250315-143022")
        result = load_result(tmp_path, "20250315")
        assert result.run_id == "baseline-vllm-20250315-143022"

    def test_no_match_raises(self, tmp_path):
        _write_result_file(tmp_path, "some-run")
        with pytest.raises(FileNotFoundError, match="No result found"):
            load_result(tmp_path, "nonexistent")

    def test_ambiguous_match_raises(self, tmp_path):
        _write_result_file(tmp_path, "run-abc-1")
        _write_result_file(tmp_path, "run-abc-2")
        with pytest.raises(ValueError, match="Ambiguous"):
            load_result(tmp_path, "abc")


class TestReconstructAggregatedMetrics:
    def test_round_trip(self, tmp_path):
        """Save results → load → reconstruct → verify fields match."""
        config = ExperimentConfig(
            name="test",
            model={"name": "test-model"},
            framework="vllm",
            infrastructure={"gpu_type": "A100-80GB"},
            workload={
                "type": "single",
                "requests": {"source": "prompts/test.jsonl", "count": 10},
            },
        )
        req_results = [
            RequestResult(
                request_index=0,
                ttft_ms=50.0,
                e2e_latency_ms=300.0,
                inter_token_latencies_ms=[10.0, 12.0],
                prompt_tokens=20,
                completion_tokens=30,
                started_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                finished_at=datetime(2025, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            ),
        ]
        aggregated = aggregate_results(req_results, total_duration_s=1.0)
        metadata = RunMetadata(run_id="rt", experiment_name="test", status="completed")

        path = save_results(tmp_path, metadata, config, req_results, aggregated)
        data = json.loads(path.read_text())
        reconstructed = reconstruct_aggregated_metrics(data["summary"])

        assert reconstructed.total_requests == aggregated.total_requests
        assert reconstructed.successful_requests == aggregated.successful_requests
        assert reconstructed.tokens_per_second == aggregated.tokens_per_second
        assert reconstructed.ttft is not None
        assert reconstructed.ttft.p50 == aggregated.ttft.p50
        assert reconstructed.ttft.p95 == aggregated.ttft.p95
        assert reconstructed.e2e_latency.p99 == aggregated.e2e_latency.p99

    def test_none_stats(self):
        summary = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_duration_s": 0.0,
            "requests_per_second": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "tokens_per_second": 0.0,
            "ttft": None,
            "e2e_latency": None,
            "tbt": None,
        }
        metrics = reconstruct_aggregated_metrics(summary)
        assert metrics.ttft is None
        assert metrics.e2e_latency is None
        assert metrics.tbt is None

    def test_gpu_summary_none_when_absent(self):
        summary = {
            "total_requests": 1,
            "successful_requests": 1,
            "failed_requests": 0,
            "total_duration_s": 1.0,
            "requests_per_second": 1.0,
            "total_prompt_tokens": 10,
            "total_completion_tokens": 20,
            "tokens_per_second": 20.0,
            "ttft": None,
            "e2e_latency": None,
            "tbt": None,
        }
        metrics = reconstruct_aggregated_metrics(summary)
        assert metrics.gpu_summary is None

    def test_gpu_summary_reconstructed(self):
        summary = {
            "total_requests": 1,
            "successful_requests": 1,
            "failed_requests": 0,
            "total_duration_s": 1.0,
            "requests_per_second": 1.0,
            "total_prompt_tokens": 10,
            "total_completion_tokens": 20,
            "tokens_per_second": 20.0,
            "ttft": None,
            "e2e_latency": None,
            "tbt": None,
            "gpu_summary": {
                "kv_cache_usage_peak": 0.5,
                "kv_cache_usage_mean": 0.3,
                "active_requests_peak": 2.0,
                "active_requests_mean": 1.5,
                "prefix_cache_hit_rate": 0.67,
                "generation_throughput": 150.0,
                "total_samples": 42,
                "scrape_errors": 1,
            },
        }
        metrics = reconstruct_aggregated_metrics(summary)
        gs = metrics.gpu_summary
        assert gs is not None
        assert gs.kv_cache_usage_peak == 0.5
        assert gs.kv_cache_usage_mean == 0.3
        assert gs.active_requests_peak == 2.0
        assert gs.prefix_cache_hit_rate == 0.67
        assert gs.generation_throughput == 150.0
        assert gs.total_samples == 42
        assert gs.scrape_errors == 1


class TestGpuDataSaveLoad:
    """Save/load results with GPU time-series data."""

    def _make_config(self) -> ExperimentConfig:
        return ExperimentConfig(
            name="gpu-test",
            model={"name": "test-model"},
            framework="vllm",
            infrastructure={"gpu_type": "A100-80GB"},
            workload={
                "type": "single",
                "requests": {"source": "prompts/test.jsonl", "count": 5},
            },
        )

    def test_save_with_gpu_data(self, tmp_path):
        from llm_inf_bench.metrics.gpu import GpuSample, GpuSummary, GpuTimeSeries

        config = self._make_config()
        req_results = [RequestResult(request_index=0, ttft_ms=50.0, e2e_latency_ms=300.0)]
        aggregated = aggregate_results(req_results, total_duration_s=1.0)
        aggregated.gpu_summary = GpuSummary(
            kv_cache_usage_peak=0.5, kv_cache_usage_mean=0.3, total_samples=2,
        )
        metadata = RunMetadata(run_id="gpu-run", experiment_name="gpu-test", status="completed")
        gpu_ts = GpuTimeSeries(
            framework="vllm",
            sample_interval_ms=100,
            samples=[
                GpuSample(timestamp=0.0, kv_cache_usage=0.3),
                GpuSample(timestamp=0.1, kv_cache_usage=0.5),
            ],
            total_scrapes=2,
            scrape_errors=0,
        )

        path = save_results(tmp_path, metadata, config, req_results, aggregated, gpu_time_series=gpu_ts)
        data = json.loads(path.read_text())

        assert "gpu_metrics" in data
        gm = data["gpu_metrics"]
        assert gm["framework"] == "vllm"
        assert gm["sample_interval_ms"] == 100
        assert gm["total_scrapes"] == 2
        assert gm["scrape_errors"] == 0
        assert len(gm["time_series"]) == 2
        assert gm["time_series"][0]["kv_cache_usage"] == 0.3

    def test_save_without_gpu_data(self, tmp_path):
        config = self._make_config()
        req_results = [RequestResult(request_index=0)]
        aggregated = aggregate_results(req_results, total_duration_s=1.0)
        metadata = RunMetadata(run_id="no-gpu", experiment_name="gpu-test", status="completed")

        path = save_results(tmp_path, metadata, config, req_results, aggregated)
        data = json.loads(path.read_text())
        assert "gpu_metrics" not in data

    def test_load_result_with_gpu_metrics(self, tmp_path):
        # Write a result file with gpu_metrics
        data = {
            "run_id": "gpu-load-test",
            "status": "completed",
            "experiment": {"name": "t", "framework": "vllm", "model": {"name": "m"}},
            "metadata": {"started_at": "2025-01-01T00:00:00"},
            "summary": {
                "total_requests": 1, "successful_requests": 1, "failed_requests": 0,
                "total_duration_s": 1.0, "requests_per_second": 1.0,
                "total_prompt_tokens": 10, "total_completion_tokens": 20,
                "tokens_per_second": 20.0, "ttft": None, "e2e_latency": None, "tbt": None,
            },
            "requests": [],
            "gpu_metrics": {
                "framework": "vllm",
                "sample_interval_ms": 100,
                "total_scrapes": 5,
                "scrape_errors": 0,
                "time_series": [{"timestamp": 0.0, "kv_cache_usage": 0.2}],
            },
        }
        fp = tmp_path / "gpu-load-test.json"
        fp.write_text(json.dumps(data))

        result = load_result(tmp_path, "gpu-load-test")
        assert result.gpu_metrics is not None
        assert result.gpu_metrics["framework"] == "vllm"
        assert len(result.gpu_metrics["time_series"]) == 1

    def test_load_result_without_gpu_metrics(self, tmp_path):
        _write_result_file(tmp_path, "no-gpu-run")
        result = load_result(tmp_path, "no-gpu-run")
        assert result.gpu_metrics is None

    def test_list_results_preserves_gpu_metrics(self, tmp_path):
        data = {
            "run_id": "list-gpu-test",
            "status": "completed",
            "experiment": {"name": "t", "framework": "vllm", "model": {"name": "m"}},
            "metadata": {"started_at": "2025-01-01T00:00:00"},
            "summary": {
                "total_requests": 1, "successful_requests": 1, "failed_requests": 0,
                "total_duration_s": 1.0, "requests_per_second": 1.0,
                "total_prompt_tokens": 10, "total_completion_tokens": 20,
                "tokens_per_second": 20.0, "ttft": None, "e2e_latency": None, "tbt": None,
            },
            "requests": [],
            "gpu_metrics": {"framework": "sglang", "sample_interval_ms": 200,
                            "total_scrapes": 3, "scrape_errors": 1, "time_series": []},
        }
        (tmp_path / "list-gpu-test.json").write_text(json.dumps(data))
        results = list_results(tmp_path)
        assert len(results) == 1
        assert results[0].gpu_metrics is not None
        assert results[0].gpu_metrics["framework"] == "sglang"
