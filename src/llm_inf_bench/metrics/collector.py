"""Per-request timing data and run metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RequestResult:
    """Timing and token data for a single inference request."""

    request_index: int
    ttft_ms: float | None = None
    e2e_latency_ms: float | None = None
    inter_token_latencies_ms: list[float] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    turn_index: int | None = None
    conversation_index: int | None = None
    completion_text: str | None = None

    @property
    def tbt_ms(self) -> float | None:
        """Mean time between tokens (inter-token latency)."""
        if not self.inter_token_latencies_ms:
            return None
        return sum(self.inter_token_latencies_ms) / len(self.inter_token_latencies_ms)

    @property
    def generation_tokens_per_second(self) -> float | None:
        """Token generation throughput for this request."""
        if not self.completion_tokens or not self.e2e_latency_ms:
            return None
        if self.e2e_latency_ms <= 0:
            return None
        # Exclude TTFT — generation time is e2e minus time-to-first-token
        gen_time_ms = self.e2e_latency_ms - (self.ttft_ms or 0)
        if gen_time_ms <= 0:
            return None
        return self.completion_tokens / (gen_time_ms / 1000.0)


@dataclass
class RunMetadata:
    """Run-level metadata attached to results output."""

    run_id: str
    experiment_name: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    server_url: str | None = None
    pod_id: str | None = None
    gpu_type: str | None = None
    gpu_count: int | None = None
    cost_per_hr: float | None = None
    status: str = "pending"
