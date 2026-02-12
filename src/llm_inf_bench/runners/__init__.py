"""Runner package — factory and re-exports."""

from __future__ import annotations

from llm_inf_bench.runners.base import HealthCheckTimeout, Runner, RunnerError
from llm_inf_bench.runners.sglang import SGLangRunner
from llm_inf_bench.runners.vllm import VLLMRunner

__all__ = [
    "Runner",
    "RunnerError",
    "HealthCheckTimeout",
    "VLLMRunner",
    "SGLangRunner",
    "create_runner",
]

_RUNNER_MAP: dict[str, type[Runner]] = {
    "vllm": VLLMRunner,
    "sglang": SGLangRunner,
}


def create_runner(framework: str, base_url: str, model: str) -> Runner:
    """Instantiate the appropriate runner for *framework*.

    Raises ``ValueError`` for unknown framework names.
    """
    cls = _RUNNER_MAP.get(framework)
    if cls is None:
        supported = ", ".join(sorted(_RUNNER_MAP))
        raise ValueError(
            f"Unknown framework {framework!r}. Supported: {supported}"
        )
    return cls(base_url=base_url, model=model)
