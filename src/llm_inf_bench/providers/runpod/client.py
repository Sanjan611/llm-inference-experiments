"""Stateful RunPod SDK wrapper with safety-net cleanup tracking.

Refactored from POC 3: API key check, pod creation, polling, and cleanup.
"""

from __future__ import annotations

import logging
import os
import time

import runpod as runpod_sdk

from llm_inf_bench.config.schema import ExperimentConfig
from llm_inf_bench.providers.base import PodTimeoutError, Provider, ProviderPod, ProvisioningError
from llm_inf_bench.providers.runpod.templates import DEFAULT_PORT, build_pod_params

logger = logging.getLogger(__name__)


class RunPodProvider(Provider):
    """RunPod GPU pod lifecycle manager."""

    def __init__(
        self,
        api_key: str | None = None,
        poll_interval: int = 5,
        ready_timeout: int = 1200,
    ) -> None:
        resolved_key = self._resolve_api_key(api_key)
        runpod_sdk.api_key = resolved_key
        self._poll_interval = poll_interval
        self._ready_timeout = ready_timeout
        self._managed_pods: set[str] = set()

    @staticmethod
    def _resolve_api_key(explicit_key: str | None = None) -> str:
        """Resolve API key: explicit > env var > app config.

        Raises:
            ProvisioningError: If no API key is found anywhere.
        """
        if explicit_key:
            return explicit_key

        env_key = os.environ.get("RUNPOD_API_KEY", "")
        if env_key:
            return env_key

        from llm_inf_bench.config.loader import load_app_config

        app_config = load_app_config()
        if app_config.runpod.api_key:
            return app_config.runpod.api_key

        raise ProvisioningError(
            "No RunPod API key found. Set RUNPOD_API_KEY environment variable "
            "or run 'llm-inf-bench init' and configure ~/.llm-inf-bench/config.yaml"
        )

    @staticmethod
    def check_api_key_configured() -> bool:
        """Non-raising check for whether an API key is available."""
        if os.environ.get("RUNPOD_API_KEY", ""):
            return True
        from llm_inf_bench.config.loader import load_app_config

        app_config = load_app_config()
        return bool(app_config.runpod.api_key)

    def get_url(self, pod_id: str, port: int) -> str:
        """Construct the RunPod proxy URL for a pod's exposed port."""
        return f"https://{pod_id}-{port}.proxy.runpod.net"

    def _wait_for_pod_ready(self, pod_id: str) -> dict:
        """Poll until the pod is running and has port mappings.

        Raises:
            PodTimeoutError: If the pod doesn't become ready within the timeout.
        """
        logger.info(
            "Waiting for pod %s (polling every %ds, timeout %ds)",
            pod_id,
            self._poll_interval,
            self._ready_timeout,
        )
        start = time.time()
        last_status = None

        while True:
            elapsed = time.time() - start
            if elapsed > self._ready_timeout:
                raise PodTimeoutError(
                    f"Pod {pod_id} did not become ready within {self._ready_timeout}s"
                )

            try:
                pod = runpod_sdk.get_pod(pod_id)
            except Exception as e:
                logger.warning("[%.0fs] API call failed (will retry): %s", elapsed, e)
                time.sleep(self._poll_interval)
                continue

            status = pod.get("desiredStatus")
            if status != last_status:
                logger.info("[%.0fs] Pod status: %s", elapsed, status)
                last_status = status

            if status == "RUNNING" and pod.get("runtime") is not None:
                ports = pod["runtime"].get("ports")
                if ports:
                    logger.info("[%.0fs] Pod is ready", elapsed)
                    return pod

            time.sleep(self._poll_interval)

    def create(self, config: ExperimentConfig) -> ProviderPod:
        """Provision a RunPod GPU pod from an experiment config.

        Raises:
            ProvisioningError: If pod creation fails.
            PodTimeoutError: If the pod doesn't become ready in time.
        """
        params = build_pod_params(config)
        logger.info("Creating pod: %s", params["name"])

        try:
            pod = runpod_sdk.create_pod(**params)
        except Exception as e:
            raise ProvisioningError(f"Failed to create pod: {e}") from e

        pod_id = pod["id"]
        self._managed_pods.add(pod_id)
        logger.info("Pod created: %s", pod_id)

        ready_pod = self._wait_for_pod_ready(pod_id)
        cost_per_hr = ready_pod.get("costPerHr", 0.0)
        base_url = self.get_url(pod_id, DEFAULT_PORT)

        return ProviderPod(
            pod_id=pod_id,
            base_url=base_url,
            gpu_type=config.infrastructure.gpu_type,
            gpu_count=config.infrastructure.gpu_count,
            cost_per_hr=cost_per_hr,
        )

    def terminate(self, pod_id: str) -> None:
        """Terminate a running pod."""
        logger.info("Terminating pod %s", pod_id)
        runpod_sdk.terminate_pod(pod_id)
        self._managed_pods.discard(pod_id)

    def cleanup_all(self) -> None:
        """Terminate all pods this provider created. Best-effort — logs failures."""
        for pod_id in list(self._managed_pods):
            try:
                self.terminate(pod_id)
            except Exception as e:
                logger.error(
                    "Failed to terminate pod %s: %s. "
                    "MANUAL ACTION REQUIRED: terminate via RunPod console.",
                    pod_id,
                    e,
                )
