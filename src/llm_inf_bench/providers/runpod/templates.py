"""Translate an ExperimentConfig into ``runpod.create_pod()`` kwargs.

Refactored from POC 3 (``build_vllm_args``) and POC 4 (``build_sglang_cmd``).
"""

from __future__ import annotations

from typing import Any

from llm_inf_bench.config.schema import ExperimentConfig, FrameworkOptions
from llm_inf_bench.providers.runpod.pricing import GPU_TYPE_TO_RUNPOD_ID

FRAMEWORK_IMAGES: dict[str, str] = {
    "vllm": "vllm/vllm-openai:latest",
    "sglang": "lmsysorg/sglang:latest",
}

DEFAULT_PORT = 8000
DEFAULT_CONTAINER_DISK_GB = 10
DEFAULT_VOLUME_DISK_GB = 10


def get_framework_image(framework: str) -> str:
    """Return the Docker image for a framework."""
    image = FRAMEWORK_IMAGES.get(framework)
    if image is None:
        raise ValueError(f"Unknown framework '{framework}'. Known: {list(FRAMEWORK_IMAGES)}")
    return image


def build_vllm_args(model: str, port: int = DEFAULT_PORT, options: FrameworkOptions | None = None) -> str:
    """Build arguments for vLLM's ``vllm serve`` entrypoint.

    The vllm/vllm-openai image has ``ENTRYPOINT ["vllm", "serve"]``, so
    docker_args is appended as arguments to that command.
    """
    args = f"{model} --host 0.0.0.0 --port {port}"

    if options is not None:
        if options.max_model_len is not None:
            args += f" --max-model-len {options.max_model_len}"
        if options.gpu_memory_utilization != 0.9:
            args += f" --gpu-memory-utilization {options.gpu_memory_utilization}"
        if options.enable_prefix_caching is True:
            args += " --enable-prefix-caching"

    return args


def build_sglang_cmd(model: str, port: int = DEFAULT_PORT, options: FrameworkOptions | None = None) -> str:
    """Build the full SGLang server startup command.

    The lmsysorg/sglang image has no ENTRYPOINT, so docker_args must be
    the complete command including ``python3 -m sglang.launch_server``.
    """
    cmd = (
        f"python3 -m sglang.launch_server "
        f"--model-path {model} "
        f"--host 0.0.0.0 "
        f"--port {port} "
        f"--enable-metrics"
    )

    if options is not None:
        if options.max_model_len is not None:
            cmd += f" --context-length {options.max_model_len}"
        if options.gpu_memory_utilization != 0.9:
            cmd += f" --mem-fraction-static {options.gpu_memory_utilization}"
        if options.enable_prefix_caching is True:
            cmd += " --enable-cache-report"

    return cmd


def build_pod_params(config: ExperimentConfig) -> dict[str, Any]:
    """Assemble full ``runpod.create_pod()`` kwargs from an experiment config."""
    gpu_type = config.infrastructure.gpu_type
    runpod_gpu_id = GPU_TYPE_TO_RUNPOD_ID.get(gpu_type)
    if runpod_gpu_id is None:
        raise ValueError(
            f"No RunPod GPU type ID mapping for '{gpu_type}'. "
            f"Known mappings: {list(GPU_TYPE_TO_RUNPOD_ID)}"
        )

    image = get_framework_image(config.framework)
    port = DEFAULT_PORT
    options = config.framework_options

    if config.framework == "vllm":
        docker_args = build_vllm_args(config.model.name, port, options)
    else:
        docker_args = build_sglang_cmd(config.model.name, port, options)

    return {
        "name": f"llm-inf-bench-{config.name}",
        "image_name": image,
        "gpu_type_id": runpod_gpu_id,
        "gpu_count": config.infrastructure.gpu_count,
        "container_disk_in_gb": DEFAULT_CONTAINER_DISK_GB,
        "volume_in_gb": DEFAULT_VOLUME_DISK_GB,
        "support_public_ip": True,
        "ports": f"{port}/http",
        "docker_args": docker_args,
    }
