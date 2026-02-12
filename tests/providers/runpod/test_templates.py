"""Tests for pod template generation."""

from __future__ import annotations

import pytest

from llm_inf_bench.config.schema import ExperimentConfig, FrameworkOptions
from llm_inf_bench.providers.runpod.templates import (
    build_pod_params,
    build_sglang_cmd,
    build_vllm_args,
    get_framework_image,
)


class TestGetFrameworkImage:
    def test_vllm(self):
        assert "vllm" in get_framework_image("vllm")

    def test_sglang(self):
        assert "sglang" in get_framework_image("sglang")

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown framework"):
            get_framework_image("unknown")


class TestBuildVllmArgs:
    def test_basic(self):
        args = build_vllm_args("Qwen/Qwen3-0.6B")
        assert args == "Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000"

    def test_custom_port(self):
        args = build_vllm_args("model", port=9000)
        assert "--port 9000" in args

    def test_with_max_model_len(self):
        opts = FrameworkOptions(max_model_len=4096)
        args = build_vllm_args("model", options=opts)
        assert "--max-model-len 4096" in args

    def test_with_gpu_memory_utilization(self):
        opts = FrameworkOptions(gpu_memory_utilization=0.8)
        args = build_vllm_args("model", options=opts)
        assert "--gpu-memory-utilization 0.8" in args

    def test_default_gpu_memory_not_included(self):
        opts = FrameworkOptions()
        args = build_vllm_args("model", options=opts)
        assert "--gpu-memory-utilization" not in args

    def test_with_prefix_caching(self):
        opts = FrameworkOptions(enable_prefix_caching=True)
        args = build_vllm_args("model", options=opts)
        assert "--enable-prefix-caching" in args


class TestBuildSglangCmd:
    def test_basic(self):
        cmd = build_sglang_cmd("Qwen/Qwen3-0.6B")
        assert cmd.startswith("python3 -m sglang.launch_server")
        assert "--model-path Qwen/Qwen3-0.6B" in cmd
        assert "--enable-metrics" in cmd

    def test_with_context_length(self):
        opts = FrameworkOptions(max_model_len=4096)
        cmd = build_sglang_cmd("model", options=opts)
        assert "--context-length 4096" in cmd
        assert "--max-model-len" not in cmd

    def test_with_mem_fraction(self):
        opts = FrameworkOptions(gpu_memory_utilization=0.8)
        cmd = build_sglang_cmd("model", options=opts)
        assert "--mem-fraction-static 0.8" in cmd


class TestBuildPodParams:
    def _make_config(self, framework="vllm", gpu_type="A100-80GB") -> ExperimentConfig:
        return ExperimentConfig(
            name="test",
            model={"name": "test-model"},
            framework=framework,
            infrastructure={"gpu_type": gpu_type},
            workload={
                "type": "single",
                "requests": {"source": "test.jsonl", "count": 10},
            },
        )

    def test_vllm_params(self):
        config = self._make_config(framework="vllm")
        params = build_pod_params(config)
        assert params["image_name"] == "vllm/vllm-openai:latest"
        assert params["gpu_type_id"] == "NVIDIA A100 80GB PCIe"
        assert "test-model" in params["docker_args"]
        assert params["gpu_count"] == 1
        assert params["ports"] == "8000/http"

    def test_sglang_params(self):
        config = self._make_config(framework="sglang")
        params = build_pod_params(config)
        assert params["image_name"] == "lmsysorg/sglang:latest"
        assert "python3 -m sglang.launch_server" in params["docker_args"]

    def test_unknown_gpu_id_raises(self):
        config = self._make_config(gpu_type="A100-80GB")
        # Patch the mapping to simulate unknown
        from llm_inf_bench.providers.runpod import templates

        original = templates.GPU_TYPE_TO_RUNPOD_ID.copy()
        try:
            templates.GPU_TYPE_TO_RUNPOD_ID.clear()
            with pytest.raises(ValueError, match="No RunPod GPU type ID"):
                build_pod_params(config)
        finally:
            templates.GPU_TYPE_TO_RUNPOD_ID.update(original)
