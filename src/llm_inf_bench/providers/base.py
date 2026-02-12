"""Abstract provider interface for GPU pod lifecycle management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from llm_inf_bench.config.schema import ExperimentConfig


@dataclass
class ProviderPod:
    """Represents a provisioned GPU pod."""

    pod_id: str
    base_url: str
    gpu_type: str
    gpu_count: int
    cost_per_hr: float


class ProvisioningError(Exception):
    """Raised when pod creation fails."""


class PodTimeoutError(Exception):
    """Raised when waiting for a pod to become ready times out."""


class Provider(ABC):
    """Abstract base class for GPU infrastructure providers."""

    @abstractmethod
    def create(self, config: ExperimentConfig) -> ProviderPod:
        """Provision a GPU pod from an experiment config.

        Raises:
            ProvisioningError: If pod creation fails.
            PodTimeoutError: If the pod doesn't become ready in time.
        """

    @abstractmethod
    def terminate(self, pod_id: str) -> None:
        """Terminate a running pod."""

    @abstractmethod
    def get_url(self, pod_id: str, port: int) -> str:
        """Construct the public URL for a pod's exposed port."""
