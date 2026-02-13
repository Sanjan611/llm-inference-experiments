"""JSON result storage and loading."""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_inf_bench.config.schema import ExperimentConfig
from llm_inf_bench.metrics.aggregator import AggregatedMetrics, PercentileStats
from llm_inf_bench.metrics.collector import RequestResult, RunMetadata
from llm_inf_bench.metrics.gpu import GpuSummary, GpuTimeSeries

logger = logging.getLogger(__name__)


def generate_run_id(experiment_name: str, now: datetime | None = None) -> str:
    """Generate a run ID in the format ``{name}-{YYYYMMDD-HHmmss}``."""
    if now is None:
        now = datetime.now()
    return f"{experiment_name}-{now.strftime('%Y%m%d-%H%M%S')}"


def _serialize(obj: Any) -> Any:
    """JSON serializer for dataclasses and datetimes."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_results(
    output_dir: str | Path,
    metadata: RunMetadata,
    config: ExperimentConfig,
    results: list[RequestResult],
    aggregated: AggregatedMetrics,
    gpu_time_series: GpuTimeSeries | None = None,
) -> Path:
    """Write results JSON to ``output_dir/{run_id}.json``.

    Creates the output directory if it doesn't exist. Returns the file path.
    When *gpu_time_series* is provided, a ``gpu_metrics`` key is added to the
    payload containing the full time-series data.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    file_path = out_path / f"{metadata.run_id}.json"

    payload: dict[str, Any] = {
        "run_id": metadata.run_id,
        "status": metadata.status,
        "experiment": config.model_dump(mode="json"),
        "metadata": {
            "experiment_name": metadata.experiment_name,
            "started_at": metadata.started_at.isoformat() if metadata.started_at else None,
            "finished_at": metadata.finished_at.isoformat() if metadata.finished_at else None,
            "server_url": metadata.server_url,
            "pod_id": metadata.pod_id,
            "gpu_type": metadata.gpu_type,
            "gpu_count": metadata.gpu_count,
            "cost_per_hr": metadata.cost_per_hr,
        },
        "summary": dataclasses.asdict(aggregated),
        "requests": [
            {k: v for k, v in dataclasses.asdict(r).items() if k != "completion_text"}
            for r in results
        ],
    }

    if gpu_time_series is not None:
        payload["gpu_metrics"] = {
            "framework": gpu_time_series.framework,
            "sample_interval_ms": gpu_time_series.sample_interval_ms,
            "total_scrapes": gpu_time_series.total_scrapes,
            "scrape_errors": gpu_time_series.scrape_errors,
            "time_series": [dataclasses.asdict(s) for s in gpu_time_series.samples],
        }

    with open(file_path, "w") as f:
        json.dump(payload, f, indent=2, default=_serialize)

    return file_path


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------


@dataclass
class StoredResult:
    """Lightweight container for a loaded result file.

    Uses raw dicts rather than Pydantic models so that results from older
    (or newer) schema versions can still be loaded.
    """

    run_id: str
    status: str
    experiment: dict
    metadata: dict
    summary: dict
    requests: list[dict] = field(default_factory=list)
    gpu_metrics: dict | None = None
    file_path: Path = field(default_factory=lambda: Path())


def list_results(output_dir: str | Path) -> list[StoredResult]:
    """Load all result files from *output_dir*, newest first.

    Malformed files are skipped with a logged warning.
    """
    out_path = Path(output_dir)
    if not out_path.is_dir():
        return []

    results: list[StoredResult] = []
    for fp in sorted(out_path.glob("*.json")):
        try:
            data = json.loads(fp.read_text())
            results.append(
                StoredResult(
                    run_id=data["run_id"],
                    status=data.get("status", "unknown"),
                    experiment=data.get("experiment", {}),
                    metadata=data.get("metadata", {}),
                    summary=data.get("summary", {}),
                    requests=data.get("requests", []),
                    gpu_metrics=data.get("gpu_metrics"),
                    file_path=fp,
                )
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Skipping malformed result file %s: %s", fp, exc)

    # Sort newest first by started_at (fall back to file name)
    def _sort_key(r: StoredResult) -> str:
        return r.metadata.get("started_at", "") or ""

    results.sort(key=_sort_key, reverse=True)
    return results


def load_result(output_dir: str | Path, run_id: str) -> StoredResult:
    """Load a single result by run ID (exact or partial match).

    Raises ``FileNotFoundError`` when no match is found, or ``ValueError``
    when the *run_id* prefix matches multiple files.
    """
    out_path = Path(output_dir)

    # Exact match first
    exact = out_path / f"{run_id}.json"
    if exact.is_file():
        data = json.loads(exact.read_text())
        return StoredResult(
            run_id=data["run_id"],
            status=data.get("status", "unknown"),
            experiment=data.get("experiment", {}),
            metadata=data.get("metadata", {}),
            summary=data.get("summary", {}),
            requests=data.get("requests", []),
            gpu_metrics=data.get("gpu_metrics"),
            file_path=exact,
        )

    # Partial / prefix match
    matches = list(out_path.glob(f"*{run_id}*.json"))
    if len(matches) == 0:
        raise FileNotFoundError(f"No result found for {run_id!r} in {out_path}")
    if len(matches) > 1:
        ids = [m.stem for m in matches]
        raise ValueError(f"Ambiguous run ID {run_id!r} — matches: {', '.join(ids)}")

    fp = matches[0]
    data = json.loads(fp.read_text())
    return StoredResult(
        run_id=data["run_id"],
        status=data.get("status", "unknown"),
        experiment=data.get("experiment", {}),
        metadata=data.get("metadata", {}),
        summary=data.get("summary", {}),
        requests=data.get("requests", []),
        gpu_metrics=data.get("gpu_metrics"),
        file_path=fp,
    )


def _reconstruct_percentile_stats(d: dict | None) -> PercentileStats | None:
    """Rebuild a ``PercentileStats`` from a stored dict."""
    if d is None:
        return None
    return PercentileStats(
        p50=d["p50"],
        p95=d["p95"],
        p99=d["p99"],
        mean=d["mean"],
        min=d["min"],
        max=d["max"],
    )


def _reconstruct_gpu_summary(d: dict | None) -> GpuSummary | None:
    """Rebuild a ``GpuSummary`` from a stored dict."""
    if d is None:
        return None
    return GpuSummary(
        kv_cache_usage_peak=d.get("kv_cache_usage_peak"),
        kv_cache_usage_mean=d.get("kv_cache_usage_mean"),
        active_requests_peak=d.get("active_requests_peak"),
        active_requests_mean=d.get("active_requests_mean"),
        prefix_cache_hit_rate=d.get("prefix_cache_hit_rate"),
        generation_throughput=d.get("generation_throughput"),
        total_samples=d.get("total_samples", 0),
        scrape_errors=d.get("scrape_errors", 0),
    )


def reconstruct_aggregated_metrics(summary: dict) -> AggregatedMetrics:
    """Reconstruct ``AggregatedMetrics`` from the JSON summary dict."""
    return AggregatedMetrics(
        total_requests=summary["total_requests"],
        successful_requests=summary["successful_requests"],
        failed_requests=summary["failed_requests"],
        total_duration_s=summary["total_duration_s"],
        requests_per_second=summary["requests_per_second"],
        total_prompt_tokens=summary["total_prompt_tokens"],
        total_completion_tokens=summary["total_completion_tokens"],
        tokens_per_second=summary["tokens_per_second"],
        ttft=_reconstruct_percentile_stats(summary.get("ttft")),
        e2e_latency=_reconstruct_percentile_stats(summary.get("e2e_latency")),
        tbt=_reconstruct_percentile_stats(summary.get("tbt")),
        gpu_summary=_reconstruct_gpu_summary(summary.get("gpu_summary")),
    )
