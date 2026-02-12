"""Tests for config schema (Pydantic models)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_inf_bench.config.schema import (
    AppConfig,
    ExperimentConfig,
    FrameworkOptions,
    InfrastructureConfig,
    MetricsConfig,
    ModelConfig,
    RequestParameters,
    RequestsConfig,
    WorkloadConfig,
)


class TestModelConfig:
    def test_basic(self):
        m = ModelConfig(name="meta-llama/Llama-3-8B")
        assert m.name == "meta-llama/Llama-3-8B"
        assert m.quantization is None

    def test_with_quantization(self):
        m = ModelConfig(name="TheBloke/Llama-2-7B-GPTQ", quantization="gptq")
        assert m.quantization == "gptq"


class TestInfrastructureConfig:
    def test_defaults(self):
        ic = InfrastructureConfig(gpu_type="A100-80GB")
        assert ic.provider == "runpod"
        assert ic.gpu_count == 1

    def test_gpu_count_must_be_positive(self):
        with pytest.raises(ValidationError):
            InfrastructureConfig(gpu_type="A100-80GB", gpu_count=0)

    def test_gpu_count_negative(self):
        with pytest.raises(ValidationError):
            InfrastructureConfig(gpu_type="A100-80GB", gpu_count=-1)


class TestRequestsConfig:
    def test_count_positive(self):
        r = RequestsConfig(source="prompts/test.jsonl", count=10)
        assert r.count == 10

    def test_count_must_be_positive(self):
        with pytest.raises(ValidationError):
            RequestsConfig(source="prompts/test.jsonl", count=0)


class TestRequestParameters:
    def test_defaults(self):
        p = RequestParameters()
        assert p.max_tokens == 256
        assert p.temperature == 0.7


class TestWorkloadConfig:
    def test_single_workload(self):
        wl = WorkloadConfig(
            type="single",
            requests=RequestsConfig(source="test.jsonl", count=10),
        )
        assert wl.type == "single"
        assert wl.batch_size is None
        assert wl.parameters.max_tokens == 256

    def test_invalid_type(self):
        with pytest.raises(ValidationError):
            WorkloadConfig(
                type="unknown",
                requests=RequestsConfig(source="test.jsonl", count=10),
            )


class TestFrameworkOptions:
    def test_defaults(self):
        opts = FrameworkOptions()
        assert opts.max_model_len is None
        assert opts.gpu_memory_utilization == 0.9
        assert opts.enable_prefix_caching is None


class TestMetricsConfig:
    def test_defaults(self):
        m = MetricsConfig()
        assert m.collect_gpu_metrics is True
        assert m.sample_interval_ms == 100
        assert m.output_dir == "results/"


class TestExperimentConfig:
    def test_minimal(self, minimal_experiment_dict):
        config = ExperimentConfig(**minimal_experiment_dict)
        assert config.name == "test-experiment"
        assert config.framework == "vllm"
        assert config.framework_options.gpu_memory_utilization == 0.9
        assert config.metrics.output_dir == "results/"

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(name="test")

    def test_invalid_framework(self, minimal_experiment_dict):
        minimal_experiment_dict["framework"] = "invalid"
        with pytest.raises(ValidationError):
            ExperimentConfig(**minimal_experiment_dict)

    def test_extends_field(self, minimal_experiment_dict):
        minimal_experiment_dict["extends"] = "base/defaults.yaml"
        config = ExperimentConfig(**minimal_experiment_dict)
        assert config.extends == "base/defaults.yaml"


class TestAppConfig:
    def test_defaults(self):
        cfg = AppConfig()
        assert cfg.runpod.api_key == ""
        assert cfg.defaults.output_dir == "./results"

    def test_with_api_key(self):
        cfg = AppConfig(runpod={"api_key": "test-key"})
        assert cfg.runpod.api_key == "test-key"
