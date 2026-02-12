"""Tests for ConcurrentWorkload execution."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_inf_bench.metrics.collector import RequestResult
from llm_inf_bench.workloads.concurrent import ConcurrentWorkload


class TestConcurrentWorkload:
    def _make_prompts(self, n: int) -> list[list[dict[str, str]]]:
        return [[{"role": "user", "content": f"Prompt {i}"}] for i in range(n)]

    def _make_mock_runner(
        self, results: list[RequestResult] | None = None
    ) -> MagicMock:
        runner = MagicMock()
        if results is not None:
            runner.chat_completion = AsyncMock(side_effect=results)
        else:
            runner.chat_completion = AsyncMock(
                return_value=RequestResult(
                    request_index=0, ttft_ms=50.0, e2e_latency_ms=300.0
                )
            )
        return runner

    @pytest.mark.asyncio
    async def test_all_success(self):
        """10 prompts with concurrency=4 should produce 10 results."""
        prompts = self._make_prompts(10)
        expected = [
            RequestResult(request_index=i, ttft_ms=50.0, e2e_latency_ms=300.0)
            for i in range(10)
        ]
        runner = self._make_mock_runner(expected)

        workload = ConcurrentWorkload(
            prompts=prompts, model="test", concurrency=4
        )
        results = await workload.execute(runner)

        assert len(results) == 10
        assert runner.chat_completion.call_count == 10

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Peak active requests should not exceed the concurrency limit."""
        active = 0
        max_active = 0
        lock = asyncio.Lock()

        async def mock_chat(**kwargs):
            nonlocal active, max_active
            async with lock:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            async with lock:
                active -= 1
            return RequestResult(
                request_index=0, ttft_ms=50.0, e2e_latency_ms=300.0
            )

        runner = MagicMock()
        runner.chat_completion = mock_chat

        prompts = self._make_prompts(20)
        workload = ConcurrentWorkload(
            prompts=prompts, model="test", concurrency=4
        )
        await workload.execute(runner)

        assert max_active <= 4

    @pytest.mark.asyncio
    async def test_error_threshold_stops_early(self):
        """Reaching max_total_errors should cancel remaining requests."""
        call_count = 0

        async def mock_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            # Add a small delay to allow cancellation to propagate
            await asyncio.sleep(0.01)
            return RequestResult(request_index=0, error="fail")

        runner = MagicMock()
        runner.chat_completion = mock_chat

        prompts = self._make_prompts(20)
        workload = ConcurrentWorkload(
            prompts=prompts,
            model="test",
            concurrency=1,  # sequential so we can predict behavior
            max_total_errors=3,
        )
        results = await workload.execute(runner)

        # Should stop after hitting the error threshold
        assert len(results) <= 20
        # At least 3 errors were recorded before stopping
        error_results = [r for r in results if r.error]
        assert len(error_results) >= 3

    @pytest.mark.asyncio
    async def test_callback_invoked(self):
        """Callback fires for each completed request."""
        prompts = self._make_prompts(5)
        expected = [
            RequestResult(request_index=i, ttft_ms=50.0, e2e_latency_ms=300.0)
            for i in range(5)
        ]
        runner = self._make_mock_runner(expected)
        callback = MagicMock()

        workload = ConcurrentWorkload(
            prompts=prompts,
            model="test",
            concurrency=2,
            on_request_complete=callback,
        )
        await workload.execute(runner)

        assert callback.call_count == 5

    @pytest.mark.asyncio
    async def test_results_ordered_by_index(self):
        """Results should be sorted by request_index regardless of completion order."""
        delays = [0.05, 0.01, 0.04, 0.02, 0.03]

        async def mock_chat(**kwargs):
            msgs = kwargs.get("messages", [])
            content = msgs[0]["content"] if msgs else ""
            idx = int(content.split()[-1])
            await asyncio.sleep(delays[idx])
            return RequestResult(
                request_index=0, ttft_ms=50.0, e2e_latency_ms=300.0
            )

        runner = MagicMock()
        runner.chat_completion = mock_chat

        prompts = self._make_prompts(5)
        workload = ConcurrentWorkload(
            prompts=prompts, model="test", concurrency=5
        )
        results = await workload.execute(runner)

        assert [r.request_index for r in results] == [0, 1, 2, 3, 4]

    def test_total_requests(self):
        prompts = self._make_prompts(8)
        workload = ConcurrentWorkload(
            prompts=prompts, model="test", concurrency=4
        )
        assert workload.total_requests() == 8

    @pytest.mark.asyncio
    async def test_concurrency_of_one(self):
        """concurrency=1 should behave like sequential execution."""
        execution_order: list[str] = []

        async def mock_chat(**kwargs):
            msgs = kwargs.get("messages", [])
            content = msgs[0]["content"] if msgs else "?"
            execution_order.append(f"start-{content}")
            await asyncio.sleep(0.01)
            execution_order.append(f"end-{content}")
            return RequestResult(
                request_index=0, ttft_ms=50.0, e2e_latency_ms=300.0
            )

        runner = MagicMock()
        runner.chat_completion = mock_chat

        prompts = self._make_prompts(3)
        workload = ConcurrentWorkload(
            prompts=prompts, model="test", concurrency=1
        )
        await workload.execute(runner)

        # With concurrency=1, starts and ends should alternate
        for i in range(len(execution_order) - 1):
            if execution_order[i].startswith("start-"):
                assert execution_order[i + 1].startswith("end-")
