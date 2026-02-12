"""Tests for sweep expansion."""

from __future__ import annotations

from llm_inf_bench.config.schema import (
    ExperimentConfig,
    SweepConfig,
    WorkloadConfig,
)
from llm_inf_bench.config.sweep import expand_sweep


def _make_config(**workload_overrides) -> ExperimentConfig:
    """Build an ExperimentConfig with sensible defaults."""
    wl = {
        "type": "single",
        "requests": {"source": "test.jsonl", "count": 10},
    }
    wl.update(workload_overrides)
    return ExperimentConfig(
        name="test-experiment",
        model={"name": "Qwen/Qwen3-0.6B"},
        framework="vllm",
        infrastructure={"gpu_type": "A100-80GB"},
        workload=wl,
    )


class TestExpandSweep:
    def test_no_sweep_returns_original(self):
        config = _make_config()
        result = expand_sweep(config)

        assert len(result) == 1
        cfg, params = result[0]
        assert cfg is config  # same object, no copy needed
        assert params == {}

    def test_empty_sweep_returns_original(self):
        config = _make_config(sweep={})
        result = expand_sweep(config)

        assert len(result) == 1
        assert result[0][1] == {}

    def test_concurrency_sweep(self):
        config = _make_config(
            type="concurrent",
            concurrency=1,
            sweep={"concurrency": [1, 2, 4]},
        )
        result = expand_sweep(config)

        assert len(result) == 3
        for cfg, params in result:
            assert cfg.workload.type == "concurrent"
            assert cfg.workload.concurrency == params["concurrency"]

        assert result[0][1] == {"concurrency": 1}
        assert result[1][1] == {"concurrency": 2}
        assert result[2][1] == {"concurrency": 4}

    def test_batch_size_sweep(self):
        config = _make_config(
            type="batch",
            batch_size=1,
            sweep={"batch_size": [2, 4, 8]},
        )
        result = expand_sweep(config)

        assert len(result) == 3
        for cfg, params in result:
            assert cfg.workload.type == "batch"
            assert cfg.workload.batch_size == params["batch_size"]

    def test_deep_copy_independence(self):
        config = _make_config(
            type="concurrent",
            concurrency=1,
            sweep={"concurrency": [1, 2]},
        )
        result = expand_sweep(config)

        # Mutating one variation doesn't affect the other
        result[0][0].workload.concurrency = 999
        assert result[1][0].workload.concurrency == 2

        # Original config is also unaffected
        assert config.workload.concurrency == 1

    def test_type_promotion_single_to_concurrent(self):
        config = _make_config(
            type="single",
            sweep={"concurrency": [1, 4]},
        )
        result = expand_sweep(config)

        assert len(result) == 2
        for cfg, _ in result:
            assert cfg.workload.type == "concurrent"

    def test_type_promotion_single_to_batch(self):
        config = _make_config(
            type="single",
            sweep={"batch_size": [2, 4]},
        )
        result = expand_sweep(config)

        assert len(result) == 2
        for cfg, _ in result:
            assert cfg.workload.type == "batch"

    def test_name_suffix_concurrency(self):
        config = _make_config(
            type="concurrent",
            concurrency=1,
            sweep={"concurrency": [1, 4, 8]},
        )
        result = expand_sweep(config)

        assert result[0][0].name == "test-experiment-c1"
        assert result[1][0].name == "test-experiment-c4"
        assert result[2][0].name == "test-experiment-c8"

    def test_name_suffix_batch(self):
        config = _make_config(
            type="batch",
            batch_size=1,
            sweep={"batch_size": [2, 8]},
        )
        result = expand_sweep(config)

        assert result[0][0].name == "test-experiment-b2"
        assert result[1][0].name == "test-experiment-b8"

    def test_cartesian_product(self):
        config = _make_config(
            type="concurrent",
            concurrency=1,
            batch_size=1,
            sweep={"concurrency": [2, 4], "batch_size": [8, 16]},
        )
        result = expand_sweep(config)

        assert len(result) == 4
        params_set = {
            (p["concurrency"], p["batch_size"]) for _, p in result
        }
        assert params_set == {(2, 8), (2, 16), (4, 8), (4, 16)}

    def test_cartesian_product_names(self):
        config = _make_config(
            type="concurrent",
            concurrency=1,
            batch_size=1,
            sweep={"concurrency": [2], "batch_size": [8]},
        )
        result = expand_sweep(config)

        assert len(result) == 1
        assert result[0][0].name == "test-experiment-c2-b8"
