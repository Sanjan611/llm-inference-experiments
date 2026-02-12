"""Tests for BatchWorkload execution."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_inf_bench.metrics.collector import RequestResult
from llm_inf_bench.workloads.batch import BatchWorkload


class TestBatchWorkload:
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
                return_value=RequestResult(request_index=0, ttft_ms=50.0, e2e_latency_ms=300.0)
            )
        return runner

    @pytest.mark.asyncio
    async def test_executes_in_batches(self):
        """6 prompts with batch_size=2 should produce 3 batches."""
        prompts = self._make_prompts(6)
        expected = [
            RequestResult(request_index=i, ttft_ms=50.0, e2e_latency_ms=300.0)
            for i in range(6)
        ]
        runner = self._make_mock_runner(expected)

        workload = BatchWorkload(prompts=prompts, model="test", batch_size=2)
        results = await workload.execute(runner)

        assert len(results) == 6
        assert [r.request_index for r in results] == [0, 1, 2, 3, 4, 5]
        assert runner.chat_completion.call_count == 6

    @pytest.mark.asyncio
    async def test_last_batch_smaller(self):
        """5 prompts with batch_size=3 should handle a final batch of 2."""
        prompts = self._make_prompts(5)
        expected = [
            RequestResult(request_index=i, ttft_ms=50.0, e2e_latency_ms=300.0)
            for i in range(5)
        ]
        runner = self._make_mock_runner(expected)

        workload = BatchWorkload(prompts=prompts, model="test", batch_size=3)
        results = await workload.execute(runner)

        assert len(results) == 5
        assert [r.request_index for r in results] == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_batch_concurrency(self):
        """Requests within a batch should run concurrently (overlap)."""
        active = 0
        max_active = 0

        async def mock_chat(**kwargs):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.05)
            active -= 1
            return RequestResult(request_index=0, ttft_ms=50.0, e2e_latency_ms=300.0)

        runner = MagicMock()
        runner.chat_completion = mock_chat

        prompts = self._make_prompts(4)
        workload = BatchWorkload(prompts=prompts, model="test", batch_size=4)
        await workload.execute(runner)

        assert max_active == 4

    @pytest.mark.asyncio
    async def test_batches_sequential(self):
        """Batch N+1 should not start until batch N finishes."""
        execution_order: list[str] = []

        async def mock_chat(**kwargs):
            # Extract prompt content to identify the request
            msgs = kwargs.get("messages", [])
            content = msgs[0]["content"] if msgs else "?"
            execution_order.append(f"start-{content}")
            await asyncio.sleep(0.02)
            execution_order.append(f"end-{content}")
            return RequestResult(request_index=0, ttft_ms=50.0, e2e_latency_ms=300.0)

        runner = MagicMock()
        runner.chat_completion = mock_chat

        prompts = self._make_prompts(4)  # 2 batches of 2
        workload = BatchWorkload(prompts=prompts, model="test", batch_size=2)
        await workload.execute(runner)

        # All batch 0 ends should appear before any batch 1 starts
        batch0_ends = [
            i for i, e in enumerate(execution_order) if e in ("end-Prompt 0", "end-Prompt 1")
        ]
        batch1_starts = [
            i for i, e in enumerate(execution_order) if e in ("start-Prompt 2", "start-Prompt 3")
        ]
        assert all(e < s for e in batch0_ends for s in batch1_starts)

    @pytest.mark.asyncio
    async def test_circuit_breaker_all_errors(self):
        """3 consecutive all-error batches should stop the workload."""
        prompts = self._make_prompts(12)  # 6 batches of 2
        errors = [RequestResult(request_index=i, error="fail") for i in range(12)]
        runner = self._make_mock_runner(errors)

        workload = BatchWorkload(
            prompts=prompts,
            model="test",
            batch_size=2,
            max_consecutive_failed_batches=3,
            max_total_errors=100,  # high, so only batch-level breaker triggers
        )
        results = await workload.execute(runner)

        # 3 failed batches * 2 requests = 6 requests
        assert len(results) == 6
        assert runner.chat_completion.call_count == 6

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_partial_success(self):
        """A batch with at least one success resets the consecutive counter."""
        prompts = self._make_prompts(8)  # 4 batches of 2
        responses = [
            # Batch 0: all fail
            RequestResult(request_index=0, error="fail"),
            RequestResult(request_index=1, error="fail"),
            # Batch 1: partial success — resets counter
            RequestResult(request_index=2, error="fail"),
            RequestResult(request_index=3, ttft_ms=50.0, e2e_latency_ms=300.0),
            # Batch 2: all fail
            RequestResult(request_index=4, error="fail"),
            RequestResult(request_index=5, error="fail"),
            # Batch 3: all fail (only 2nd consecutive)
            RequestResult(request_index=6, error="fail"),
            RequestResult(request_index=7, error="fail"),
        ]
        runner = self._make_mock_runner(responses)

        workload = BatchWorkload(
            prompts=prompts,
            model="test",
            batch_size=2,
            max_consecutive_failed_batches=3,
            max_total_errors=100,
        )
        results = await workload.execute(runner)

        # All 4 batches run because counter resets after batch 1
        assert len(results) == 8

    @pytest.mark.asyncio
    async def test_callback_invoked_per_request(self):
        """Callback fires per request, not per batch."""
        prompts = self._make_prompts(4)
        expected = [
            RequestResult(request_index=i, ttft_ms=50.0, e2e_latency_ms=300.0)
            for i in range(4)
        ]
        runner = self._make_mock_runner(expected)
        callback = MagicMock()

        workload = BatchWorkload(
            prompts=prompts, model="test", batch_size=2, on_request_complete=callback
        )
        await workload.execute(runner)

        assert callback.call_count == 4

    def test_total_requests(self):
        prompts = self._make_prompts(7)
        workload = BatchWorkload(prompts=prompts, model="test", batch_size=3)
        assert workload.total_requests() == 7

    @pytest.mark.asyncio
    async def test_total_error_threshold(self):
        """max_total_errors safety net stops the workload."""
        prompts = self._make_prompts(10)  # 5 batches of 2
        # All fail but only 1 per batch so consecutive batch breaker won't fire
        responses = []
        for i in range(10):
            if i % 2 == 0:
                responses.append(RequestResult(request_index=i, error="fail"))
            else:
                responses.append(RequestResult(request_index=i, ttft_ms=50.0, e2e_latency_ms=300.0))
        runner = self._make_mock_runner(responses)

        workload = BatchWorkload(
            prompts=prompts,
            model="test",
            batch_size=2,
            max_consecutive_failed_batches=100,
            max_total_errors=3,
        )
        results = await workload.execute(runner)

        # Stops after 3 total errors: that's batch 0 (1 err), batch 1 (1 err), batch 2 (1 err) = 6 requests
        assert len(results) == 6
