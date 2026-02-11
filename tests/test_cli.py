"""Tests for CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import yaml
from typer.testing import CliRunner

from llm_inf_bench.cli import app
from llm_inf_bench.metrics.collector import RequestResult

runner = CliRunner()


class TestInit:
    def test_creates_config(self, tmp_path):
        config_dir = tmp_path / ".llm-inf-bench"
        config_path = config_dir / "config.yaml"

        with (
            patch("llm_inf_bench.cli.APP_CONFIG_DIR", config_dir),
            patch("llm_inf_bench.cli.APP_CONFIG_PATH", config_path),
        ):
            result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert config_path.exists()
        data = yaml.safe_load(config_path.read_text())
        assert "runpod" in data
        assert data["runpod"]["api_key"] == ""

    def test_does_not_overwrite(self, tmp_path):
        config_dir = tmp_path / ".llm-inf-bench"
        config_dir.mkdir()
        config_path = config_dir / "config.yaml"
        config_path.write_text("existing: true\n")

        with (
            patch("llm_inf_bench.cli.APP_CONFIG_DIR", config_dir),
            patch("llm_inf_bench.cli.APP_CONFIG_PATH", config_path),
        ):
            result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert "already exists" in result.output
        assert config_path.read_text() == "existing: true\n"


class TestDoctor:
    def test_passes_with_api_key(self, monkeypatch):
        monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
        with patch("httpx.get") as mock_get:
            mock_get.return_value.status_code = 200
            result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "Python" in result.output

    def test_fails_without_api_key(self, monkeypatch):
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        with (
            patch(
                "llm_inf_bench.config.loader.load_app_config",
            ) as mock_cfg,
            patch("httpx.get") as mock_get,
        ):
            from unittest.mock import MagicMock

            mock_cfg.return_value = MagicMock(runpod=MagicMock(api_key=""))
            mock_get.return_value.status_code = 200
            result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 1
        assert "FAIL" in result.output


class TestRun:
    def test_dry_run_valid_config(self, tmp_path):
        config_path = tmp_path / "exp.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "name": "test-run",
                    "model": {"name": "Qwen/Qwen3-0.6B"},
                    "framework": "vllm",
                    "infrastructure": {"gpu_type": "A100-80GB"},
                    "workload": {
                        "type": "single",
                        "requests": {"source": "test.jsonl", "count": 10},
                    },
                }
            )
        )
        result = runner.invoke(app, ["run", str(config_path), "--dry-run"])
        assert result.exit_code == 0
        assert "Config is valid" in result.output
        assert "test-run" in result.output

    def test_dry_run_invalid_gpu(self, tmp_path):
        config_path = tmp_path / "bad.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "name": "bad",
                    "model": {"name": "test"},
                    "framework": "vllm",
                    "infrastructure": {"gpu_type": "FAKE-GPU"},
                    "workload": {
                        "type": "single",
                        "requests": {"source": "test.jsonl", "count": 10},
                    },
                }
            )
        )
        result = runner.invoke(app, ["run", str(config_path), "--dry-run"])
        assert result.exit_code == 1
        assert "FAKE-GPU" in result.output

    def test_missing_config_file(self):
        result = runner.invoke(app, ["run", "/nonexistent/config.yaml", "--dry-run"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_run_with_server_url_executes(self, tmp_path):
        """Run command with --server-url and mocked runner completes successfully."""
        # Create prompts file
        prompts_path = tmp_path / "prompts.jsonl"
        prompts_path.write_text(
            '{"messages": [{"role": "user", "content": "Hello"}]}\n'
        )
        results_dir = tmp_path / "results"

        config_path = tmp_path / "exp.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "name": "test",
                    "model": {"name": "test-model"},
                    "framework": "vllm",
                    "infrastructure": {"gpu_type": "A100-80GB"},
                    "workload": {
                        "type": "single",
                        "requests": {"source": str(prompts_path), "count": 2},
                    },
                    "metrics": {"output_dir": str(results_dir)},
                }
            )
        )

        mock_result = RequestResult(
            request_index=0,
            ttft_ms=50.0,
            e2e_latency_ms=300.0,
            inter_token_latencies_ms=[10.0, 12.0],
            prompt_tokens=10,
            completion_tokens=20,
        )

        with (
            patch("llm_inf_bench.cli.create_runner") as mock_create_runner,
        ):
            mock_runner_instance = AsyncMock()
            mock_create_runner.return_value = mock_runner_instance
            mock_runner_instance.wait_for_health = AsyncMock()
            mock_runner_instance.chat_completion = AsyncMock(return_value=mock_result)
            mock_runner_instance.close = AsyncMock()

            result = runner.invoke(
                app,
                ["run", str(config_path), "--server-url", "http://localhost:8000", "--confirm"],
            )

        assert result.exit_code == 0, result.output
        assert results_dir.exists()
        json_files = list(results_dir.glob("*.json"))
        assert len(json_files) == 1
        data = json.loads(json_files[0].read_text())
        assert data["status"] in ("completed", "partial")
        assert len(data["requests"]) == 2

    def test_run_cost_confirmation_abort(self, tmp_path):
        """Run command prompts for cost confirmation and aborts when declined."""
        config_path = tmp_path / "exp.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "name": "test",
                    "model": {"name": "test-model"},
                    "framework": "vllm",
                    "infrastructure": {"gpu_type": "A100-80GB"},
                    "workload": {
                        "type": "single",
                        "requests": {"source": "test.jsonl", "count": 10},
                    },
                }
            )
        )
        # Answer "n" to the confirmation prompt
        result = runner.invoke(app, ["run", str(config_path)], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output


# ---------------------------------------------------------------------------
# Helpers for results commands
# ---------------------------------------------------------------------------


def _write_result(directory, run_id, *, framework="vllm", model="test-model", started_at="2025-01-01T00:00:00"):
    """Write a minimal valid result JSON file."""
    directory.mkdir(parents=True, exist_ok=True)
    data = {
        "run_id": run_id,
        "status": "completed",
        "experiment": {
            "name": "test",
            "framework": framework,
            "model": {"name": model},
            "infrastructure": {"gpu_type": "A100-80GB"},
        },
        "metadata": {"started_at": started_at},
        "summary": {
            "total_requests": 10,
            "successful_requests": 10,
            "failed_requests": 0,
            "total_duration_s": 5.0,
            "requests_per_second": 2.0,
            "total_prompt_tokens": 200,
            "total_completion_tokens": 500,
            "tokens_per_second": 100.0,
            "ttft": {"p50": 50.0, "p95": 90.0, "p99": 99.0, "mean": 55.0, "min": 10.0, "max": 120.0},
            "e2e_latency": {"p50": 300.0, "p95": 400.0, "p99": 500.0, "mean": 310.0, "min": 100.0, "max": 600.0},
            "tbt": {"p50": 12.0, "p95": 18.0, "p99": 24.0, "mean": 13.0, "min": 5.0, "max": 30.0},
        },
        "requests": [],
    }
    fp = directory / f"{run_id}.json"
    fp.write_text(json.dumps(data))
    return fp


class TestResultsList:
    def test_no_results(self, tmp_path):
        result = runner.invoke(app, ["results", "list", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No results found" in result.output

    def test_lists_results(self, tmp_path):
        _write_result(tmp_path, "run-a")
        _write_result(tmp_path, "run-b")
        result = runner.invoke(app, ["results", "list", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "run-a" in result.output
        assert "run-b" in result.output


class TestResultsShow:
    def test_shows_result(self, tmp_path):
        _write_result(tmp_path, "my-run")
        result = runner.invoke(app, ["results", "show", "my-run", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "my-run" in result.output
        assert "vllm" in result.output

    def test_missing_result(self, tmp_path):
        tmp_path.mkdir(exist_ok=True)
        result = runner.invoke(app, ["results", "show", "nope", "--dir", str(tmp_path)])
        assert result.exit_code == 1


class TestResultsCompare:
    def test_compares_two_results(self, tmp_path):
        _write_result(tmp_path, "run-a", framework="vllm")
        _write_result(tmp_path, "run-b", framework="sglang")
        result = runner.invoke(
            app, ["results", "compare", "run-a", "run-b", "--dir", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "run-a" in result.output
        assert "run-b" in result.output

    def test_missing_result(self, tmp_path):
        _write_result(tmp_path, "run-a")
        result = runner.invoke(
            app, ["results", "compare", "run-a", "nope", "--dir", str(tmp_path)]
        )
        assert result.exit_code == 1
