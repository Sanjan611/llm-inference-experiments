"""Batch workload — dispatches prompts in fixed-size groups via asyncio.gather."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from llm_inf_bench.metrics.collector import RequestResult
from llm_inf_bench.runners.base import Runner
from llm_inf_bench.workloads.base import Workload

logger = logging.getLogger(__name__)


class BatchWorkload(Workload):
    """Execute requests in batches of ``batch_size``.

    Each batch is dispatched concurrently via ``asyncio.gather``.
    Batches run sequentially: batch N+1 starts only after batch N completes.
    """

    def __init__(
        self,
        prompts: list[list[dict[str, str]]],
        model: str,
        batch_size: int,
        max_tokens: int = 256,
        temperature: float = 0.7,
        on_request_complete: Callable[[RequestResult], None] | None = None,
        max_consecutive_failed_batches: int = 3,
        max_total_errors: int | None = None,
    ) -> None:
        self._prompts = prompts
        self._model = model
        self._batch_size = batch_size
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._on_request_complete = on_request_complete
        self._max_consecutive_failed_batches = max_consecutive_failed_batches
        self._max_total_errors = (
            max_total_errors if max_total_errors is not None else len(prompts) // 2
        )

    def total_requests(self) -> int:
        return len(self._prompts)

    async def execute(self, runner: Runner) -> list[RequestResult]:
        """Execute all requests in batches with circuit breaker."""
        results: list[RequestResult] = []
        consecutive_failed_batches = 0
        total_errors = 0

        # Split prompts into batches
        batches: list[list[tuple[int, list[dict[str, str]]]]] = []
        for i in range(0, len(self._prompts), self._batch_size):
            batch = [
                (i + j, msgs)
                for j, msgs in enumerate(self._prompts[i : i + self._batch_size])
            ]
            batches.append(batch)

        for batch in batches:
            # Dispatch all requests in this batch concurrently
            batch_results = await asyncio.gather(
                *[
                    runner.chat_completion(
                        messages=msgs,
                        model=self._model,
                        max_tokens=self._max_tokens,
                        temperature=self._temperature,
                    )
                    for _, msgs in batch
                ]
            )

            # Assign request indices and track errors
            batch_errors = 0
            for (idx, _), result in zip(batch, batch_results, strict=True):
                result.request_index = idx
                if result.error:
                    batch_errors += 1
                    total_errors += 1
                results.append(result)
                if self._on_request_complete:
                    self._on_request_complete(result)

            # Circuit breaker: batch-level consecutive failures
            if batch_errors == len(batch):
                consecutive_failed_batches += 1
                logger.warning(
                    "Batch failed (%d consecutive failed batches)",
                    consecutive_failed_batches,
                )
            else:
                consecutive_failed_batches = 0

            if consecutive_failed_batches >= self._max_consecutive_failed_batches:
                logger.error(
                    "Circuit breaker: %d consecutive failed batches, stopping workload",
                    consecutive_failed_batches,
                )
                break

            # Safety net: total error threshold
            if total_errors >= self._max_total_errors:
                logger.error(
                    "Circuit breaker: %d total errors (threshold %d), stopping workload",
                    total_errors,
                    self._max_total_errors,
                )
                break

        return results
