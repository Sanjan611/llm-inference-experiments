"""vLLM runner — thin subclass of OpenAICompatibleRunner."""

from __future__ import annotations

from llm_inf_bench.runners.openai_compat import OpenAICompatibleRunner


class VLLMRunner(OpenAICompatibleRunner):
    """Runner for vLLM's OpenAI-compatible API."""

    def __init__(self, base_url: str, model: str) -> None:
        super().__init__(base_url=base_url, model=model, health_interval=5.0)
