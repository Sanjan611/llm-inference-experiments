"""Pydantic models for all WebSocket message types."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Server -> Client events
# ---------------------------------------------------------------------------


class PhaseEvent(BaseModel):
    """Experiment phase transition."""

    type: str = "phase"
    run_id: str
    phase: str
    data: dict[str, Any] = {}


class GpuSampleEvent(BaseModel):
    """A single GPU metrics scrape forwarded in real time."""

    type: str = "gpu_sample"
    run_id: str
    data: dict[str, Any]


class RequestCompleteEvent(BaseModel):
    """A single request has finished (success or failure)."""

    type: str = "request_complete"
    run_id: str
    data: dict[str, Any]


class LogEvent(BaseModel):
    """Log message captured from the llm_inf_bench loggers."""

    type: str = "log"
    level: str
    message: str
    timestamp: str


class SummaryEvent(BaseModel):
    """Aggregated metrics sent after an iteration completes."""

    type: str = "summary"
    run_id: str
    data: dict[str, Any]


class ErrorEvent(BaseModel):
    """Experiment-level error."""

    type: str = "error"
    message: str


# ---------------------------------------------------------------------------
# Client -> Server commands
# ---------------------------------------------------------------------------


class StartExperimentCommand(BaseModel):
    """Request to start an experiment from a config file."""

    type: str = "start_experiment"
    config_path: str
    server_url: str | None = None
    run_name: str | None = None
    confirm: bool = True


class StopExperimentCommand(BaseModel):
    """Request to stop a running experiment."""

    type: str = "stop_experiment"
    run_id: str
