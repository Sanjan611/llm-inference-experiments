"""Tests for JSON result storage."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from llm_inf_bench.config.schema import ExperimentConfig
from llm_inf_bench.metrics.aggregator import AggregatedMetrics, PercentileStats, aggregate_results
from llm_inf_bench.metrics.collector import RequestResult, RunMetadata
from llm_inf_bench.metrics.storage import generate_run_id, save_results


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
