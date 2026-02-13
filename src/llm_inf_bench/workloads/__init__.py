"""Workload package — factory and re-exports."""

from __future__ import annotations

from collections.abc import Callable

from llm_inf_bench.metrics.collector import RequestResult
from llm_inf_bench.workloads.base import Workload
from llm_inf_bench.workloads.batch import BatchWorkload
from llm_inf_bench.workloads.concurrent import ConcurrentWorkload
from llm_inf_bench.workloads.multi_turn import (
    ConversationScript,
    MultiTurnWorkload,
    load_multi_turn_prompts,
)
from llm_inf_bench.workloads.single import SingleWorkload, load_prompts

__all__ = [
    "BatchWorkload",
    "ConcurrentWorkload",
    "ConversationScript",
    "MultiTurnWorkload",
    "SingleWorkload",
    "Workload",
    "create_workload",
    "load_multi_turn_prompts",
    "load_prompts",
]


def create_workload(
    workload_type: str,
    prompts: list[list[dict[str, str]]],
    model: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    on_request_complete: Callable[[RequestResult], None] | None = None,
    batch_size: int | None = None,
    concurrency: int | None = None,
    conversations: list[ConversationScript] | None = None,
    conversation_turns: int | None = None,
) -> Workload:
    """Instantiate the appropriate workload for *workload_type*.

    Raises ``ValueError`` for unknown types or missing required parameters.
    """
    if workload_type == "single":
        return SingleWorkload(
            prompts=prompts,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            on_request_complete=on_request_complete,
        )

    if workload_type == "batch":
        if batch_size is None:
            raise ValueError("Workload type 'batch' requires 'batch_size'")
        return BatchWorkload(
            prompts=prompts,
            model=model,
            batch_size=batch_size,
            max_tokens=max_tokens,
            temperature=temperature,
            on_request_complete=on_request_complete,
        )

    if workload_type == "concurrent":
        if concurrency is None:
            raise ValueError("Workload type 'concurrent' requires 'concurrency'")
        return ConcurrentWorkload(
            prompts=prompts,
            model=model,
            concurrency=concurrency,
            max_tokens=max_tokens,
            temperature=temperature,
            on_request_complete=on_request_complete,
        )

    if workload_type == "multi_turn":
        if conversations is None:
            raise ValueError("Workload type 'multi_turn' requires 'conversations'")
        if conversation_turns is None:
            raise ValueError("Workload type 'multi_turn' requires 'conversation_turns'")
        return MultiTurnWorkload(
            conversations=conversations,
            turns=conversation_turns,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            on_request_complete=on_request_complete,
        )

    supported = "single, batch, concurrent, multi_turn"
    raise ValueError(f"Unknown workload type {workload_type!r}. Supported: {supported}")
