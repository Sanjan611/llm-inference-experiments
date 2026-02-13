"""Experiment lifecycle manager that bridges orchestration with WebSocket events."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_inf_bench.config.loader import load_experiment
from llm_inf_bench.config.schema import ExperimentConfig
from llm_inf_bench.config.sweep import expand_sweep
from llm_inf_bench.dashboard.websocket import ConnectionManager
from llm_inf_bench.metrics.aggregator import aggregate_results
from llm_inf_bench.metrics.collector import RequestResult, RunMetadata
from llm_inf_bench.metrics.gpu import GpuMetricsScraper, GpuSample, GpuTimeSeries
from llm_inf_bench.metrics.storage import generate_run_id, save_results
from llm_inf_bench.providers.runpod.client import RunPodProvider
from llm_inf_bench.runners import Runner, create_runner
from llm_inf_bench.runners.base import HealthCheckTimeout
from llm_inf_bench.workloads import create_workload, load_prompts
from llm_inf_bench.workloads.multi_turn import load_multi_turn_prompts

logger = logging.getLogger(__name__)


@dataclass
class ActiveRun:
    """Tracks a currently-running experiment."""

    run_id: str
    config_path: str
    experiment: ExperimentConfig
    task: asyncio.Task[None]
    started_at: datetime


@dataclass
class CompletedRun:
    """Summary of a finished experiment."""

    run_id: str
    config_path: str
    experiment_name: str
    status: str
    result_path: str | None = None


class ExperimentManager:
    """Run experiments as asyncio tasks and broadcast events to WebSocket clients."""

    def __init__(self, manager: ConnectionManager) -> None:
        self._ws = manager
        self._active: ActiveRun | None = None
        self._completed: list[CompletedRun] = []
        self._cancel_event: asyncio.Event = asyncio.Event()

    @property
    def active_run(self) -> ActiveRun | None:
        return self._active

    @property
    def completed_runs(self) -> list[CompletedRun]:
        return list(self._completed)

    async def start_experiment(
        self,
        config_path: str,
        server_url: str | None = None,
    ) -> str:
        """Start an experiment. Returns the run_id. Raises if one is already running."""
        if self._active is not None:
            raise RuntimeError("An experiment is already running")

        experiment = load_experiment(Path(config_path))
        run_id = generate_run_id(experiment.name)

        self._cancel_event.clear()
        task = asyncio.create_task(
            self._run_experiment(run_id, config_path, experiment, server_url)
        )
        self._active = ActiveRun(
            run_id=run_id,
            config_path=config_path,
            experiment=experiment,
            task=task,
            started_at=datetime.now(timezone.utc),
        )
        return run_id

    async def stop_experiment(self, run_id: str) -> None:
        """Request graceful stop of a running experiment."""
        if self._active is None or self._active.run_id != run_id:
            return
        self._cancel_event.set()

    # ------------------------------------------------------------------
    # Internal orchestration (mirrors cli._execute_run / _execute_iteration)
    # ------------------------------------------------------------------

    async def _run_experiment(
        self,
        run_id: str,
        config_path: str,
        experiment: ExperimentConfig,
        server_url: str | None,
    ) -> None:
        """Full experiment lifecycle with event broadcasting."""
        provider: RunPodProvider | None = None
        pod_id: str | None = None
        gpu_type: str | None = None
        gpu_count: int | None = None
        cost_per_hr: float | None = None
        runner: Runner | None = None
        status = "completed"
        result_path: str | None = None

        try:
            # Phase 1: Provisioning
            if server_url:
                await self._broadcast_phase(
                    run_id,
                    "provisioning_skipped",
                    {
                        "server_url": server_url,
                    },
                )
                base_url = server_url.rstrip("/")
            else:
                await self._broadcast_phase(
                    run_id,
                    "provisioning",
                    {
                        "gpu_type": experiment.infrastructure.gpu_type,
                        "gpu_count": experiment.infrastructure.gpu_count,
                    },
                )
                provider = RunPodProvider()
                pod = provider.create(experiment)
                pod_id = pod.pod_id
                gpu_type = pod.gpu_type
                gpu_count = pod.gpu_count
                cost_per_hr = pod.cost_per_hr
                base_url = pod.base_url
                await self._broadcast_phase(
                    run_id,
                    "provisioning_done",
                    {
                        "pod_id": pod_id,
                        "gpu_type": gpu_type,
                        "gpu_count": gpu_count,
                        "cost_per_hr": cost_per_hr,
                        "server_url": base_url,
                    },
                )

            if self._cancel_event.is_set():
                return

            # Phase 2: Health check
            runner = create_runner(
                framework=experiment.framework,
                base_url=base_url,
                model=experiment.model.name,
            )
            await self._broadcast_phase(
                run_id,
                "model_loading",
                {
                    "model": experiment.model.name,
                },
            )
            health_start = time.monotonic()
            try:
                await runner.wait_for_health()
            except HealthCheckTimeout as e:
                await self._broadcast_error(f"Health check failed: {e}")
                status = "failed"
                return
            elapsed = time.monotonic() - health_start
            await self._broadcast_phase(
                run_id,
                "model_loading_done",
                {
                    "elapsed_s": round(elapsed, 1),
                },
            )

            if self._cancel_event.is_set():
                return

            # Phase 3: Sweep loop
            variations = expand_sweep(experiment)
            for i, (variation, params) in enumerate(variations, 1):
                if self._cancel_event.is_set():
                    break

                if len(variations) > 1:
                    await self._broadcast_phase(
                        run_id,
                        "sweep_iteration",
                        {
                            "iteration": i,
                            "total": len(variations),
                            "params": params,
                        },
                    )

                iter_result_path = await self._execute_iteration(
                    run_id,
                    variation,
                    runner,
                    base_url,
                    pod_id,
                    server_url,
                    gpu_type=gpu_type,
                    gpu_count=gpu_count,
                    cost_per_hr=cost_per_hr,
                )
                if iter_result_path:
                    result_path = iter_result_path

        except Exception as e:
            logger.exception("Experiment failed: %s", e)
            await self._broadcast_error(str(e))
            status = "failed"
        finally:
            # Cleanup
            await self._broadcast_phase(run_id, "cleanup", {})
            if runner:
                await runner.close()
            if provider and pod_id:
                try:
                    provider.terminate(pod_id)
                except Exception as e:
                    logger.error("Failed to terminate pod %s: %s", pod_id, e)

            await self._broadcast_phase(
                run_id,
                "done",
                {
                    "result_path": result_path,
                    "status": status,
                },
            )

            self._completed.append(
                CompletedRun(
                    run_id=run_id,
                    config_path=config_path,
                    experiment_name=experiment.name,
                    status=status,
                    result_path=result_path,
                )
            )
            self._active = None

    async def _execute_iteration(
        self,
        run_id: str,
        experiment: ExperimentConfig,
        runner: Runner,
        base_url: str,
        pod_id: str | None,
        server_url: str | None,
        *,
        gpu_type: str | None = None,
        gpu_count: int | None = None,
        cost_per_hr: float | None = None,
    ) -> str | None:
        """Execute a single benchmark iteration and return the result file path."""
        scraper: GpuMetricsScraper | None = None
        gpu_time_series: GpuTimeSeries | None = None
        results: list[RequestResult] = []
        run_started = datetime.now(timezone.utc)
        exec_start: float | None = None

        bg_tasks: set[asyncio.Task[None]] = set()

        def _fire_and_forget(coro: Any) -> None:
            task = asyncio.get_running_loop().create_task(coro)
            bg_tasks.add(task)
            task.add_done_callback(bg_tasks.discard)

        def on_sample(sample: GpuSample) -> None:
            """Forward GPU sample to WebSocket clients."""
            _fire_and_forget(
                self._ws.broadcast(
                    {
                        "type": "gpu_sample",
                        "run_id": run_id,
                        "data": {
                            "timestamp": sample.timestamp,
                            "kv_cache_usage": sample.kv_cache_usage,
                            "active_requests": sample.active_requests,
                            "queued_requests": sample.queued_requests,
                            "generation_throughput": sample.generation_throughput,
                            "prefix_cache_hit_rate": sample.prefix_cache_hit_rate,
                        },
                    }
                )
            )

        def on_request_complete(result: RequestResult) -> None:
            """Forward request result to WebSocket clients."""
            _fire_and_forget(
                self._ws.broadcast(
                    {
                        "type": "request_complete",
                        "run_id": run_id,
                        "data": {
                            "request_index": result.request_index,
                            "ttft_ms": result.ttft_ms,
                            "e2e_latency_ms": result.e2e_latency_ms,
                            "tbt_ms": result.tbt_ms,
                            "prompt_tokens": result.prompt_tokens,
                            "completion_tokens": result.completion_tokens,
                            "error": result.error,
                        },
                    }
                )
            )

        try:
            await self._broadcast_phase(
                run_id,
                "execution",
                {
                    "total_requests": experiment.workload.requests.count,
                    "workload_type": experiment.workload.type,
                },
            )

            # Start GPU metrics scraper
            if experiment.metrics.collect_gpu_metrics and hasattr(runner, "http_client"):
                scraper = GpuMetricsScraper(
                    client=runner.http_client,
                    framework=experiment.framework,
                    sample_interval_ms=experiment.metrics.sample_interval_ms,
                    on_sample=on_sample,
                )
                scraper.start()

            # Execute workload
            conversations = None
            conversation_turns = None
            if experiment.workload.type == "multi_turn":
                conv = experiment.workload.conversation
                assert conv is not None  # validated earlier
                conversations = load_multi_turn_prompts(
                    source=experiment.workload.requests.source,
                    count=experiment.workload.requests.count,
                    turns=conv.turns,
                    system_prompt=conv.system_prompt,
                    user_messages=conv.user_messages,
                    shared_prefix=conv.shared_prefix,
                )
                conversation_turns = conv.turns
                prompts = []  # not used for multi_turn
            else:
                prompts = load_prompts(
                    experiment.workload.requests.source,
                    experiment.workload.requests.count,
                )
            workload = create_workload(
                workload_type=experiment.workload.type,
                prompts=prompts,
                model=experiment.model.name,
                max_tokens=experiment.workload.parameters.max_tokens,
                temperature=experiment.workload.parameters.temperature,
                on_request_complete=on_request_complete,
                batch_size=experiment.workload.batch_size,
                concurrency=experiment.workload.concurrency,
                conversations=conversations,
                conversation_turns=conversation_turns,
            )

            exec_start = time.monotonic()
            results = await workload.execute(runner)
            exec_duration = time.monotonic() - exec_start

            # Stop GPU scraper
            if scraper is not None:
                gpu_time_series = await scraper.stop()

            # Aggregate + save
            aggregated = aggregate_results(results, total_duration_s=exec_duration)
            if gpu_time_series is not None:
                aggregated.gpu_summary = GpuMetricsScraper.summarize(gpu_time_series)

            iter_run_id = generate_run_id(experiment.name)
            metadata = RunMetadata(
                run_id=iter_run_id,
                experiment_name=experiment.name,
                started_at=run_started,
                finished_at=datetime.now(timezone.utc),
                server_url=base_url,
                pod_id=pod_id,
                gpu_type=gpu_type,
                gpu_count=gpu_count,
                cost_per_hr=cost_per_hr,
                status="completed" if aggregated.failed_requests == 0 else "partial",
            )

            output_dir = experiment.metrics.output_dir
            result_path = save_results(
                output_dir,
                metadata,
                experiment,
                results,
                aggregated,
                gpu_time_series=gpu_time_series,
            )

            # Broadcast summary
            await self._ws.broadcast(
                {
                    "type": "summary",
                    "run_id": run_id,
                    "data": dataclasses.asdict(aggregated),
                }
            )

            return str(result_path)

        except Exception:
            # Best-effort: stop scraper and save partial results
            if scraper is not None and gpu_time_series is None:
                with contextlib.suppress(Exception):
                    gpu_time_series = await scraper.stop()

            if results:
                exec_duration = time.monotonic() - exec_start if exec_start is not None else 0
                aggregated = aggregate_results(results, total_duration_s=exec_duration)
                if gpu_time_series is not None:
                    aggregated.gpu_summary = GpuMetricsScraper.summarize(gpu_time_series)
                iter_run_id = generate_run_id(experiment.name)
                metadata = RunMetadata(
                    run_id=iter_run_id,
                    experiment_name=experiment.name,
                    started_at=run_started,
                    finished_at=datetime.now(timezone.utc),
                    server_url=server_url,
                    pod_id=pod_id,
                    gpu_type=gpu_type,
                    gpu_count=gpu_count,
                    cost_per_hr=cost_per_hr,
                    status="partial",
                )
                result_path = save_results(
                    experiment.metrics.output_dir,
                    metadata,
                    experiment,
                    results,
                    aggregated,
                    gpu_time_series=gpu_time_series,
                )
                return str(result_path)
            raise

    # ------------------------------------------------------------------
    # Broadcast helpers
    # ------------------------------------------------------------------

    async def _broadcast_phase(
        self,
        run_id: str,
        phase: str,
        data: dict[str, Any],
    ) -> None:
        await self._ws.broadcast(
            {
                "type": "phase",
                "run_id": run_id,
                "phase": phase,
                "data": data,
            }
        )

    async def _broadcast_error(self, message: str) -> None:
        await self._ws.broadcast(
            {
                "type": "error",
                "message": message,
            }
        )
