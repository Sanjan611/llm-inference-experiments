"""Concurrent workload — semaphore-gated parallel request execution."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from llm_inf_bench.metrics.collector import RequestResult
from llm_inf_bench.runners.base import Runner
from llm_inf_bench.workloads.base import Workload

logger = logging.getLogger(__name__)


class ConcurrentWorkload(Workload):
    """Execute requests with a fixed concurrency limit via semaphore.

    All tasks are created upfront. Each acquires the semaphore before calling
    the runner. As one finishes and releases, the next pending task acquires it,
    maintaining steady-state concurrency of exactly ``concurrency``.
    """

    def __init__(
        self,
        prompts: list[list[dict[str, str]]],
        model: str,
        concurrency: int,
        max_tokens: int = 256,
        temperature: float = 0.7,
        on_request_complete: Callable[[RequestResult], None] | None = None,
        max_total_errors: int | None = None,
    ) -> None:
        self._prompts = prompts
        self._model = model
        self._concurrency = concurrency
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._on_request_complete = on_request_complete
        self._max_total_errors = (
            max_total_errors if max_total_errors is not None else len(prompts) // 2
        )

    def total_requests(self) -> int:
        return len(self._prompts)

    async def execute(self, runner: Runner) -> list[RequestResult]:
        """Execute all requests with bounded concurrency."""
        semaphore = asyncio.Semaphore(self._concurrency)
        cancel_event = asyncio.Event()
        error_count = 0
        lock = asyncio.Lock()
        results: list[RequestResult | None] = [None] * len(self._prompts)

        async def _run_one(index: int, messages: list[dict[str, str]]) -> None:
            nonlocal error_count

            if cancel_event.is_set():
                return

            async with semaphore:
                if cancel_event.is_set():
                    return

                result = await runner.chat_completion(
                    messages=messages,
                    model=self._model,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                result.request_index = index
                results[index] = result

                if result.error:
                    async with lock:
                        error_count += 1
                        if error_count >= self._max_total_errors:
                            logger.error(
                                "Circuit breaker: %d total errors (threshold %d), "
                                "cancelling remaining requests",
                                error_count,
                                self._max_total_errors,
                            )
                            cancel_event.set()

                if self._on_request_complete:
                    self._on_request_complete(result)

        await asyncio.gather(
            *[_run_one(i, msgs) for i, msgs in enumerate(self._prompts)]
        )

        # Return only completed results, sorted by request_index
        return sorted(
            [r for r in results if r is not None],
            key=lambda r: r.request_index,
        )
