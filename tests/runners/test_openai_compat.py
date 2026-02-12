"""Tests for OpenAICompatibleRunner base class."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from llm_inf_bench.runners.base import HealthCheckTimeout
from llm_inf_bench.runners.openai_compat import OpenAICompatibleRunner

from .conftest import (
    async_context_manager,
    make_content_chunk,
    make_sse_lines,
    make_stream_response,
    make_usage_chunk,
)


@pytest.fixture
def runner():
    return OpenAICompatibleRunner(
        base_url="http://localhost:8000",
        model="test-model",
        health_interval=0.01,
        health_timeout=600.0,
    )


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_passes_on_200(self, runner):
        mock_resp = MagicMock(status_code=200)
        with patch.object(runner._client, "get", new_callable=AsyncMock, return_value=mock_resp):
            await runner.wait_for_health(timeout=5)

    @pytest.mark.asyncio
    async def test_timeout_raises(self, runner):
        with patch.object(
            runner._client, "get", new_callable=AsyncMock, side_effect=httpx.ConnectError("refused")
        ):
            with pytest.raises(HealthCheckTimeout, match="not healthy"):
                await runner.wait_for_health(timeout=0.05, interval=0.01)

    @pytest.mark.asyncio
    async def test_uses_custom_interval(self, runner):
        """Health check sleeps for the configured interval between attempts."""
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("refused")
            return MagicMock(status_code=200)

        with (
            patch.object(runner._client, "get", new_callable=AsyncMock, side_effect=side_effect),
            patch("llm_inf_bench.runners.openai_compat.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            await runner.wait_for_health(timeout=5, interval=0.42)

        # asyncio.sleep should have been called with the custom interval
        for call in mock_sleep.call_args_list:
            assert call.args[0] == 0.42

    @pytest.mark.asyncio
    async def test_defaults_to_instance_interval(self, runner):
        """Without explicit interval kwarg, uses self._health_interval."""
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ConnectError("refused")
            return MagicMock(status_code=200)

        with (
            patch.object(runner._client, "get", new_callable=AsyncMock, side_effect=side_effect),
            patch("llm_inf_bench.runners.openai_compat.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            await runner.wait_for_health(timeout=5)

        assert mock_sleep.call_args.args[0] == runner._health_interval


class TestStreamCompletion:
    @pytest.mark.asyncio
    async def test_parses_streaming_response(self, runner):
        sse_lines = make_sse_lines([
            make_content_chunk("Hello"),
            make_content_chunk(" world"),
            make_content_chunk("!"),
            make_usage_chunk(prompt=10, completion=3),
        ])
        mock_resp = make_stream_response(sse_lines)

        with patch.object(
            runner._client, "stream", return_value=async_context_manager(mock_resp),
        ):
            result = await runner.chat_completion(
                messages=[{"role": "user", "content": "test"}],
                model="test-model",
            )

        assert result.error is None
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 3
        assert result.ttft_ms is not None and result.ttft_ms > 0
        assert len(result.inter_token_latencies_ms) == 2

    @pytest.mark.asyncio
    async def test_error_captured_in_result(self, runner):
        with patch.object(
            runner._client, "stream", side_effect=httpx.ConnectError("refused"),
        ):
            result = await runner.chat_completion(
                messages=[{"role": "user", "content": "test"}],
                model="test-model",
            )

        assert result.error is not None
        assert "refused" in result.error

    @pytest.mark.asyncio
    async def test_token_count_fallback(self, runner):
        """Without usage in the response, count content chunks."""
        sse_lines = make_sse_lines([
            make_content_chunk("a"),
            make_content_chunk("b"),
        ])
        mock_resp = make_stream_response(sse_lines)

        with patch.object(
            runner._client, "stream", return_value=async_context_manager(mock_resp),
        ):
            result = await runner.chat_completion(
                messages=[{"role": "user", "content": "test"}],
                model="test-model",
            )

        assert result.completion_tokens == 2
        assert result.prompt_tokens is None
