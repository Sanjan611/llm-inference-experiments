"""Tests for prompt loading and SingleWorkload execution."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_inf_bench.metrics.collector import RequestResult
from llm_inf_bench.workloads.single import SingleWorkload, load_prompts


class TestLoadPrompts:
    def test_valid_jsonl(self, tmp_path):
        path = tmp_path / "prompts.jsonl"
        path.write_text(
            '{"messages": [{"role": "user", "content": "Hello"}]}\n'
            '{"messages": [{"role": "user", "content": "World"}]}\n'
        )
        prompts = load_prompts(path, count=2)
        assert len(prompts) == 2
        assert prompts[0][0]["content"] == "Hello"
        assert prompts[1][0]["content"] == "World"

    def test_cycles_prompts(self, tmp_path):
        path = tmp_path / "prompts.jsonl"
        path.write_text('{"messages": [{"role": "user", "content": "A"}]}\n')
        prompts = load_prompts(path, count=3)
        assert len(prompts) == 3
        assert all(p[0]["content"] == "A" for p in prompts)

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_prompts("/nonexistent/file.jsonl", count=1)

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text("not json\n")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_prompts(path, count=1)

    def test_missing_messages_key(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"prompt": "hello"}\n')
        with pytest.raises(ValueError, match="Missing 'messages'"):
            load_prompts(path, count=1)

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        with pytest.raises(ValueError, match="No prompts found"):
            load_prompts(path, count=1)

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "prompts.jsonl"
        path.write_text(
            '\n'
            '{"messages": [{"role": "user", "content": "A"}]}\n'
            '\n'
        )
        prompts = load_prompts(path, count=1)
        assert len(prompts) == 1


class TestSingleWorkload:
    def _make_mock_runner(self, results: list[RequestResult]) -> MagicMock:
        runner = MagicMock()
        runner.chat_completion = AsyncMock(side_effect=results)
        return runner

    @pytest.mark.asyncio
    async def test_execute_all_success(self):
        prompts = [
            [{"role": "user", "content": "A"}],
            [{"role": "user", "content": "B"}],
        ]
        expected = [
            RequestResult(request_index=0, ttft_ms=50.0, e2e_latency_ms=300.0),
            RequestResult(request_index=1, ttft_ms=60.0, e2e_latency_ms=350.0),
        ]
        runner = self._make_mock_runner(expected)

        workload = SingleWorkload(prompts=prompts, model="test")
        results = await workload.execute(runner)

        assert len(results) == 2
        assert results[0].request_index == 0
        assert results[1].request_index == 1
        assert runner.chat_completion.call_count == 2

    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        prompts = [[{"role": "user", "content": "A"}]] * 10
        errors = [
            RequestResult(request_index=i, error="fail")
            for i in range(10)
        ]
        runner = self._make_mock_runner(errors)

        workload = SingleWorkload(
            prompts=prompts,
            model="test",
            max_consecutive_errors=3,
        )
        results = await workload.execute(runner)

        # Should stop after 3 consecutive errors
        assert len(results) == 3
        assert runner.chat_completion.call_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_success(self):
        prompts = [[{"role": "user", "content": "A"}]] * 6
        responses = [
            RequestResult(request_index=0, error="fail"),
            RequestResult(request_index=1, error="fail"),
            RequestResult(request_index=2, ttft_ms=50.0, e2e_latency_ms=300.0),  # success resets
            RequestResult(request_index=3, error="fail"),
            RequestResult(request_index=4, error="fail"),
            RequestResult(request_index=5, ttft_ms=50.0, e2e_latency_ms=300.0),
        ]
        runner = self._make_mock_runner(responses)

        workload = SingleWorkload(
            prompts=prompts,
            model="test",
            max_consecutive_errors=3,
        )
        results = await workload.execute(runner)

        assert len(results) == 6  # All executed because success resets counter

    @pytest.mark.asyncio
    async def test_callback_invoked(self):
        prompts = [[{"role": "user", "content": "A"}]]
        expected = [RequestResult(request_index=0, ttft_ms=50.0)]
        runner = self._make_mock_runner(expected)
        callback = MagicMock()

        workload = SingleWorkload(
            prompts=prompts, model="test", on_request_complete=callback
        )
        await workload.execute(runner)

        callback.assert_called_once()

    def test_total_requests(self):
        prompts = [[{"role": "user", "content": "A"}]] * 5
        workload = SingleWorkload(prompts=prompts, model="test")
        assert workload.total_requests() == 5
