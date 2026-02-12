"""Abstract workload interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_inf_bench.metrics.collector import RequestResult
from llm_inf_bench.runners.base import Runner


class Workload(ABC):
    """Abstract base class for workload execution strategies."""

    @abstractmethod
    async def execute(self, runner: Runner) -> list[RequestResult]:
        """Execute the workload and return per-request results."""

    @abstractmethod
    def total_requests(self) -> int:
        """Return the total number of requests this workload will make."""
