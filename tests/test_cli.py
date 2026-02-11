"""Tests for CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml
from typer.testing import CliRunner

from llm_inf_bench.cli import app

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

    def test_run_without_dry_run_exits(self, tmp_path):
        config_path = tmp_path / "exp.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "name": "test",
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
        result = runner.invoke(app, ["run", str(config_path)])
        assert result.exit_code == 1
        assert "not yet implemented" in result.output
