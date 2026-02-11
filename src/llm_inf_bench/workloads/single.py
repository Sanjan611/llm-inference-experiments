"""Sequential single-request workload with circuit breaker."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

from llm_inf_bench.metrics.collector import RequestResult
from llm_inf_bench.runners.base import Runner
from llm_inf_bench.workloads.base import Workload

logger = logging.getLogger(__name__)


def load_prompts(source: str | Path, count: int) -> list[list[dict[str, str]]]:
    """Load prompts from a JSONL file.

    Each line must have a ``messages`` key with a list of chat messages.
    If *count* exceeds the number of prompts, they cycle.

    Returns:
        A list of *count* message lists.

    Raises:
        FileNotFoundError: If the source file doesn't exist.
        ValueError: If a line is invalid JSON or missing ``messages``.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    raw_prompts: list[list[dict[str, str]]] = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path}: {e}"
                ) from e
            if "messages" not in data:
                raise ValueError(
                    f"Missing 'messages' key on line {line_num} of {path}"
                )
            raw_prompts.append(data["messages"])

    if not raw_prompts:
        raise ValueError(f"No prompts found in {path}")

    # Cycle prompts to reach the desired count
    result: list[list[dict[str, str]]] = []
    for i in range(count):
        result.append(raw_prompts[i % len(raw_prompts)])
    return result


class SingleWorkload(Workload):
    """Execute requests one at a time sequentially."""

    def __init__(
        self,
        prompts: list[list[dict[str, str]]],
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        on_request_complete: Callable[[RequestResult], None] | None = None,
        max_consecutive_errors: int = 5,
    ) -> None:
        self._prompts = prompts
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._on_request_complete = on_request_complete
        self._max_consecutive_errors = max_consecutive_errors

    def total_requests(self) -> int:
        return len(self._prompts)

    async def execute(self, runner: Runner) -> list[RequestResult]:
        """Execute all requests sequentially with circuit breaker."""
        results: list[RequestResult] = []
        consecutive_errors = 0

        for i, messages in enumerate(self._prompts):
            result = await runner.chat_completion(
                messages=messages,
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            result.request_index = i

            if result.error:
                consecutive_errors += 1
                logger.warning(
                    "Request %d failed (%d consecutive): %s",
                    i, consecutive_errors, result.error,
                )
            else:
                consecutive_errors = 0

            results.append(result)

            if self._on_request_complete:
                self._on_request_complete(result)

            if consecutive_errors >= self._max_consecutive_errors:
                logger.error(
                    "Circuit breaker: %d consecutive errors, stopping workload",
                    consecutive_errors,
                )
                break

        return results
