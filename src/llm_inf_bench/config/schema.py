"""Pydantic v2 models defining all configuration structures."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# --- Experiment config models (in dependency order) ---


class ModelConfig(BaseModel):
    """Model to benchmark."""

    name: str
    quantization: str | None = None


class InfrastructureConfig(BaseModel):
    """GPU infrastructure settings."""

    provider: Literal["runpod"] = "runpod"
    gpu_type: str
    gpu_count: int = Field(default=1, gt=0)


class RequestsConfig(BaseModel):
    """Request source and count."""

    source: str
    count: int = Field(gt=0)


class RequestParameters(BaseModel):
    """Parameters applied to each inference request."""

    max_tokens: int = 256
    temperature: float = 0.7


class SweepConfig(BaseModel):
    """Parameter sweep configuration (Phase 5)."""

    concurrency: list[int] | None = None
    batch_size: list[int] | None = None


class ConversationConfig(BaseModel):
    """Multi-turn conversation settings (Phase 6)."""

    turns: int = Field(gt=0)
    system_prompt: str | None = None
    user_messages: list[str] | None = None
    shared_prefix: str | None = None


class WorkloadConfig(BaseModel):
    """Workload definition. Type-specific fields are validated semantically in validation.py."""

    type: Literal["single", "batch", "concurrent", "multi_turn"]
    requests: RequestsConfig
    parameters: RequestParameters = RequestParameters()
    batch_size: int | None = None
    concurrency: int | None = None
    sweep: SweepConfig | None = None
    conversation: ConversationConfig | None = None


class FrameworkOptions(BaseModel):
    """Engine-specific tuning knobs."""

    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.9
    enable_prefix_caching: bool | None = None


class MetricsConfig(BaseModel):
    """Metrics collection settings."""

    collect_gpu_metrics: bool = True
    sample_interval_ms: int = 100
    output_dir: str = "results/"


class ExperimentConfig(BaseModel):
    """Top-level experiment definition loaded from YAML."""

    name: str
    description: str | None = None
    extends: str | None = None
    model: ModelConfig
    framework: Literal["vllm", "sglang"]
    framework_options: FrameworkOptions = FrameworkOptions()
    infrastructure: InfrastructureConfig = InfrastructureConfig(gpu_type="A100-80GB")
    workload: WorkloadConfig
    metrics: MetricsConfig = MetricsConfig()


# --- App config models (separate from experiment configs) ---


class RunPodAppConfig(BaseModel):
    """RunPod-specific application settings."""

    api_key: str = ""


class DefaultsAppConfig(BaseModel):
    """Default application settings."""

    output_dir: str = "./results"


class AppConfig(BaseModel):
    """Application-level config stored at ~/.llm-inf-bench/config.yaml."""

    runpod: RunPodAppConfig = RunPodAppConfig()
    defaults: DefaultsAppConfig = DefaultsAppConfig()
