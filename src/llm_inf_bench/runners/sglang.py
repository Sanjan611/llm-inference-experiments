"""SGLang runner — thin subclass of OpenAICompatibleRunner."""

from __future__ import annotations

from llm_inf_bench.runners.openai_compat import OpenAICompatibleRunner


class SGLangRunner(OpenAICompatibleRunner):
    """Runner for SGLang's OpenAI-compatible API.

    Uses a 10s health-check interval (vs 5s for vLLM) because SGLang
    model loading tends to be slower.
    """

    def __init__(self, base_url: str, model: str) -> None:
        super().__init__(base_url=base_url, model=model, health_interval=10.0)
