"""Tests for RequestResult and RunMetadata dataclasses."""

from __future__ import annotations

from datetime import datetime, timezone

from llm_inf_bench.metrics.collector import RequestResult, RunMetadata


class TestRequestResult:
    def test_defaults(self):
        r = RequestResult(request_index=0)
        assert r.ttft_ms is None
        assert r.e2e_latency_ms is None
        assert r.inter_token_latencies_ms == []
        assert r.prompt_tokens is None
        assert r.completion_tokens is None
        assert r.error is None
        assert r.started_at is None
        assert r.finished_at is None

    def test_tbt_ms_with_values(self):
        r = RequestResult(request_index=0, inter_token_latencies_ms=[10.0, 20.0, 30.0])
        assert r.tbt_ms == 20.0

    def test_tbt_ms_empty_list(self):
        r = RequestResult(request_index=0, inter_token_latencies_ms=[])
        assert r.tbt_ms is None

    def test_tbt_ms_single_value(self):
        r = RequestResult(request_index=0, inter_token_latencies_ms=[15.0])
        assert r.tbt_ms == 15.0

    def test_generation_tokens_per_second(self):
        r = RequestResult(
            request_index=0,
            ttft_ms=100.0,
            e2e_latency_ms=1100.0,
            completion_tokens=50,
        )
        # gen_time = 1100 - 100 = 1000ms = 1s -> 50 tok/s
        assert r.generation_tokens_per_second == 50.0

    def test_generation_tokens_per_second_no_ttft(self):
        r = RequestResult(
            request_index=0,
            e2e_latency_ms=1000.0,
            completion_tokens=50,
        )
        # gen_time = 1000 - 0 = 1000ms = 1s -> 50 tok/s
        assert r.generation_tokens_per_second == 50.0

    def test_generation_tokens_per_second_no_tokens(self):
        r = RequestResult(request_index=0, e2e_latency_ms=1000.0)
        assert r.generation_tokens_per_second is None

    def test_generation_tokens_per_second_no_latency(self):
        r = RequestResult(request_index=0, completion_tokens=50)
        assert r.generation_tokens_per_second is None

    def test_generation_tokens_per_second_zero_latency(self):
        r = RequestResult(request_index=0, e2e_latency_ms=0, completion_tokens=50)
        assert r.generation_tokens_per_second is None

    def test_generation_tokens_per_second_zero_gen_time(self):
        # ttft equals e2e -> gen_time = 0
        r = RequestResult(
            request_index=0,
            ttft_ms=500.0,
            e2e_latency_ms=500.0,
            completion_tokens=50,
        )
        assert r.generation_tokens_per_second is None

    def test_error_result(self):
        r = RequestResult(request_index=5, error="connection refused")
        assert r.error == "connection refused"
        assert r.tbt_ms is None
        assert r.generation_tokens_per_second is None


class TestRunMetadata:
    def test_defaults(self):
        m = RunMetadata(run_id="test-123", experiment_name="test")
        assert m.status == "pending"
        assert m.server_url is None
        assert m.pod_id is None

    def test_full_metadata(self):
        now = datetime.now(timezone.utc)
        m = RunMetadata(
            run_id="test-123",
            experiment_name="baseline",
            started_at=now,
            server_url="https://example.com",
            pod_id="pod-abc",
            status="completed",
        )
        assert m.started_at == now
        assert m.status == "completed"
