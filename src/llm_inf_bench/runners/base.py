"""Abstract runner interface for inference frameworks."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_inf_bench.metrics.collector import RequestResult


class RunnerError(Exception):
    """Base exception for runner errors."""


class HealthCheckTimeout(RunnerError):
    """Raised when the health check does not pass within the timeout."""


class Runner(ABC):
    """Abstract base class for inference framework runners."""

    @abstractmethod
    async def wait_for_health(
        self, timeout: float = 600, interval: float = 5
    ) -> None:
        """Poll the server until it reports healthy.

        Raises:
            HealthCheckTimeout: If the server is not healthy within *timeout* seconds.
        """

    @abstractmethod
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> RequestResult:
        """Send a streaming chat completion request and return timing data.

        Errors are captured in the returned ``RequestResult.error`` field
        rather than raised as exceptions.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release HTTP client resources."""
