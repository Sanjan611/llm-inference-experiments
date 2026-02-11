"""Tests for semantic config validation."""

from __future__ import annotations

from llm_inf_bench.config.schema import (
    ConversationConfig,
    ExperimentConfig,
    RequestsConfig,
    WorkloadConfig,
)
from llm_inf_bench.config.validation import validate_experiment


def _make_config(**overrides) -> ExperimentConfig:
    """Build an ExperimentConfig with sensible defaults, applying overrides."""
    base = {
        "name": "test",
        "model": {"name": "Qwen/Qwen3-0.6B"},
        "framework": "vllm",
        "infrastructure": {"gpu_type": "A100-80GB"},
        "workload": {
            "type": "single",
            "requests": {"source": "test.jsonl", "count": 10},
        },
    }
    base.update(overrides)
    return ExperimentConfig(**base)


class TestGPUTypeValidation:
    def test_known_gpu_passes(self):
        config = _make_config(infrastructure={"gpu_type": "A100-80GB"})
        assert validate_experiment(config) == []

    def test_unknown_gpu_fails(self):
        config = _make_config(infrastructure={"gpu_type": "FAKE-GPU"})
        errors = validate_experiment(config)
        assert len(errors) == 1
        assert "Unknown gpu_type" in errors[0]
        assert "FAKE-GPU" in errors[0]

    def test_all_known_gpus_pass(self):
        from llm_inf_bench.providers.runpod.pricing import get_known_gpu_types

        for gpu in get_known_gpu_types():
            config = _make_config(infrastructure={"gpu_type": gpu})
            assert validate_experiment(config) == [], f"GPU {gpu} failed validation"


class TestWorkloadTypeValidation:
    def test_batch_requires_batch_size(self):
        config = _make_config(
            workload={
                "type": "batch",
                "requests": {"source": "test.jsonl", "count": 10},
            }
        )
        errors = validate_experiment(config)
        assert any("batch_size" in e for e in errors)

    def test_batch_with_batch_size_passes(self):
        config = _make_config(
            workload={
                "type": "batch",
                "requests": {"source": "test.jsonl", "count": 10},
                "batch_size": 8,
            }
        )
        errors = validate_experiment(config)
        assert errors == []

    def test_concurrent_requires_concurrency(self):
        config = _make_config(
            workload={
                "type": "concurrent",
                "requests": {"source": "test.jsonl", "count": 10},
            }
        )
        errors = validate_experiment(config)
        assert any("concurrency" in e for e in errors)

    def test_concurrent_with_concurrency_passes(self):
        config = _make_config(
            workload={
                "type": "concurrent",
                "requests": {"source": "test.jsonl", "count": 10},
                "concurrency": 4,
            }
        )
        assert validate_experiment(config) == []

    def test_multi_turn_requires_conversation(self):
        config = _make_config(
            workload={
                "type": "multi_turn",
                "requests": {"source": "test.jsonl", "count": 10},
            }
        )
        errors = validate_experiment(config)
        assert any("conversation" in e for e in errors)

    def test_multi_turn_with_conversation_passes(self):
        config = _make_config(
            workload={
                "type": "multi_turn",
                "requests": {"source": "test.jsonl", "count": 10},
                "conversation": {"turns": 3},
            }
        )
        assert validate_experiment(config) == []

    def test_single_workload_passes(self):
        config = _make_config()
        assert validate_experiment(config) == []


class TestMultipleErrors:
    def test_collects_all_errors(self):
        config = _make_config(
            infrastructure={"gpu_type": "FAKE-GPU"},
            workload={
                "type": "batch",
                "requests": {"source": "test.jsonl", "count": 10},
            },
        )
        errors = validate_experiment(config)
        assert len(errors) == 2
        assert any("gpu_type" in e for e in errors)
        assert any("batch_size" in e for e in errors)
