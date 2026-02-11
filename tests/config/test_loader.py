"""Tests for config loader (YAML loading, extends, path resolution)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llm_inf_bench.config.loader import deep_merge, load_experiment, load_yaml, resolve_extends
from llm_inf_bench.config.schema import ExperimentConfig
from llm_inf_bench.config.validation import ConfigValidationError


class TestDeepMerge:
    def test_flat_merge(self):
        assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_override_leaf(self):
        assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_merge(self):
        base = {"x": {"a": 1, "b": 2}}
        override = {"x": {"b": 3, "c": 4}}
        assert deep_merge(base, override) == {"x": {"a": 1, "b": 3, "c": 4}}

    def test_override_dict_with_scalar(self):
        assert deep_merge({"x": {"a": 1}}, {"x": "flat"}) == {"x": "flat"}

    def test_empty_override(self):
        base = {"a": 1}
        assert deep_merge(base, {}) == {"a": 1}


class TestLoadYaml:
    def test_loads_dict(self, tmp_path):
        path = tmp_path / "test.yaml"
        path.write_text("key: value\n")
        assert load_yaml(path) == {"key": "value"}

    def test_rejects_non_dict(self, tmp_path):
        path = tmp_path / "test.yaml"
        path.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            load_yaml(path)


class TestResolveExtends:
    def test_no_extends(self):
        data = {"name": "test"}
        assert resolve_extends(data, Path("/tmp")) == {"name": "test"}

    def test_extends_merges(self, base_experiment_yaml):
        child_data = {
            "extends": "base/defaults.yaml",
            "name": "child-experiment",
            "workload": {
                "type": "single",
                "requests": {"source": "prompts/test.jsonl", "count": 50},
            },
        }
        config_dir = base_experiment_yaml.parent.parent
        result = resolve_extends(child_data, config_dir)

        assert result["name"] == "child-experiment"
        assert result["workload"]["requests"]["count"] == 50
        # Inherited from base
        assert result["model"]["name"] == "Qwen/Qwen3-0.6B"
        assert "extends" not in result

    def test_missing_parent(self, tmp_path):
        data = {"extends": "nonexistent.yaml"}
        with pytest.raises(FileNotFoundError):
            resolve_extends(data, tmp_path)


class TestLoadExperiment:
    def test_loads_valid_config(self, minimal_experiment_yaml):
        config = load_experiment(minimal_experiment_yaml)
        assert isinstance(config, ExperimentConfig)
        assert config.name == "test-experiment"

    def test_extends_inheritance(self, base_experiment_yaml):
        child_path = base_experiment_yaml.parent.parent / "child.yaml"
        child_path.write_text(
            yaml.dump(
                {
                    "extends": "base/defaults.yaml",
                    "name": "child",
                    "workload": {
                        "type": "single",
                        "requests": {"source": "test.jsonl", "count": 25},
                    },
                }
            )
        )
        config = load_experiment(child_path)
        assert config.name == "child"
        assert config.workload.requests.count == 25
        assert config.model.name == "Qwen/Qwen3-0.6B"  # inherited

    def test_semantic_validation_runs(self, tmp_path):
        """Unknown GPU type should raise ConfigValidationError."""
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
        with pytest.raises(ConfigValidationError) as exc_info:
            load_experiment(config_path)
        assert "FAKE-GPU" in str(exc_info.value)

    def test_relative_source_path_resolved(self, tmp_path):
        """Source path should be resolved relative to the config file."""
        subdir = tmp_path / "configs"
        subdir.mkdir()
        config_path = subdir / "exp.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "name": "test",
                    "model": {"name": "test-model"},
                    "framework": "vllm",
                    "infrastructure": {"gpu_type": "A100-80GB"},
                    "workload": {
                        "type": "single",
                        "requests": {"source": "../prompts/data.jsonl", "count": 10},
                    },
                }
            )
        )
        config = load_experiment(config_path)
        expected = str((tmp_path / "prompts" / "data.jsonl").resolve())
        assert config.workload.requests.source == expected
