"""JSON result storage."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_inf_bench.config.schema import ExperimentConfig
from llm_inf_bench.metrics.aggregator import AggregatedMetrics
from llm_inf_bench.metrics.collector import RequestResult, RunMetadata


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
) -> Path:
    """Write results JSON to ``output_dir/{run_id}.json``.

    Creates the output directory if it doesn't exist. Returns the file path.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    file_path = out_path / f"{metadata.run_id}.json"

    payload = {
        "run_id": metadata.run_id,
        "status": metadata.status,
        "experiment": config.model_dump(mode="json"),
        "metadata": {
            "experiment_name": metadata.experiment_name,
            "started_at": metadata.started_at.isoformat() if metadata.started_at else None,
            "finished_at": metadata.finished_at.isoformat() if metadata.finished_at else None,
            "server_url": metadata.server_url,
            "pod_id": metadata.pod_id,
        },
        "summary": dataclasses.asdict(aggregated),
        "requests": [dataclasses.asdict(r) for r in results],
    }

    with open(file_path, "w") as f:
        json.dump(payload, f, indent=2, default=_serialize)

    return file_path
