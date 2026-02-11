"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Alias for pytest's tmp_path."""
    return tmp_path


@pytest.fixture()
def minimal_experiment_dict() -> dict[str, Any]:
    """Minimal valid experiment config as a dict."""
    return {
        "name": "test-experiment",
        "model": {"name": "Qwen/Qwen3-0.6B"},
        "framework": "vllm",
        "infrastructure": {"gpu_type": "A100-80GB"},
        "workload": {
            "type": "single",
            "requests": {"source": "prompts/test.jsonl", "count": 10},
        },
    }


@pytest.fixture()
def minimal_experiment_yaml(tmp_path: Path, minimal_experiment_dict: dict[str, Any]) -> Path:
    """Write a minimal valid experiment config to a temp YAML file."""
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.dump(minimal_experiment_dict))
    return config_path


@pytest.fixture()
def base_experiment_yaml(tmp_path: Path) -> Path:
    """Write a base experiment config for extends testing."""
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    base_path = base_dir / "defaults.yaml"
    base_path.write_text(
        yaml.dump(
            {
                "name": "base-experiment",
                "model": {"name": "Qwen/Qwen3-0.6B"},
                "framework": "vllm",
                "infrastructure": {"gpu_type": "A100-80GB"},
                "workload": {
                    "type": "single",
                    "requests": {"source": "prompts/test.jsonl", "count": 100},
                },
            }
        )
    )
    return base_path
