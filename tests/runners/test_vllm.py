"""Tests for VLLMRunner with mocked HTTP responses."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from llm_inf_bench.runners.base import HealthCheckTimeout
from llm_inf_bench.runners.vllm import VLLMRunner


@pytest.fixture
def runner():
    return VLLMRunner(base_url="http://localhost:8000", model="test-model")


def _make_sse_lines(chunks: list[dict]) -> list[str]:
    """Build SSE-formatted lines from a list of chunk dicts."""
    lines = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}")
        lines.append("")  # blank line between events
    lines.append("data: [DONE]")
    return lines


def _make_content_chunk(content: str, index: int = 0) -> dict:
    return {
        "choices": [{"index": index, "delta": {"content": content}}],
    }


def _make_usage_chunk(prompt: int, completion: int) -> dict:
    return {
        "choices": [],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_passes_on_200(self, runner):
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch.object(runner._client, "get", new_callable=AsyncMock, return_value=mock_resp):
            await runner.wait_for_health(timeout=5, interval=0.01)

    @pytest.mark.asyncio
    async def test_timeout_raises(self, runner):
        with patch.object(
            runner._client, "get", new_callable=AsyncMock, side_effect=httpx.ConnectError("refused")
        ):
            with pytest.raises(HealthCheckTimeout, match="not healthy"):
                await runner.wait_for_health(timeout=0.05, interval=0.01)

    @pytest.mark.asyncio
    async def test_retries_on_connect_error(self, runner):
        """Should retry on connection errors and eventually succeed."""
        mock_fail = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_ok = AsyncMock(return_value=MagicMock(status_code=200))

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("refused")
            return MagicMock(status_code=200)

        with patch.object(runner._client, "get", new_callable=AsyncMock, side_effect=side_effect):
            await runner.wait_for_health(timeout=5, interval=0.01)

        assert call_count >= 3


class TestChatCompletion:
    @pytest.mark.asyncio
    async def test_parses_streaming_response(self, runner):
        sse_lines = _make_sse_lines([
            _make_content_chunk("Hello"),
            _make_content_chunk(" world"),
            _make_content_chunk("!"),
            _make_usage_chunk(prompt=10, completion=3),
        ])

        mock_resp = _make_stream_response(sse_lines)

        with patch.object(
            runner._client,
            "stream",
            return_value=_async_context_manager(mock_resp),
        ):
            result = await runner.chat_completion(
                messages=[{"role": "user", "content": "test"}],
                model="test-model",
            )

        assert result.error is None
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 3
        assert result.ttft_ms is not None
        assert result.ttft_ms > 0
        assert result.e2e_latency_ms is not None
        assert len(result.inter_token_latencies_ms) == 2  # 3 chunks -> 2 gaps

    @pytest.mark.asyncio
    async def test_handles_http_error(self, runner):
        mock_resp = AsyncMock()
        mock_resp.status_code = 500
        mock_resp.aread = AsyncMock(return_value=b"Internal Server Error")

        with patch.object(
            runner._client,
            "stream",
            return_value=_async_context_manager(mock_resp),
        ):
            result = await runner.chat_completion(
                messages=[{"role": "user", "content": "test"}],
                model="test-model",
            )

        assert result.error is not None
        assert "500" in result.error

    @pytest.mark.asyncio
    async def test_handles_connection_error(self, runner):
        with patch.object(
            runner._client,
            "stream",
            side_effect=httpx.ConnectError("refused"),
        ):
            result = await runner.chat_completion(
                messages=[{"role": "user", "content": "test"}],
                model="test-model",
            )

        assert result.error is not None
        assert "refused" in result.error

    @pytest.mark.asyncio
    async def test_fallback_token_count(self, runner):
        """Without usage in the response, count content chunks."""
        sse_lines = _make_sse_lines([
            _make_content_chunk("a"),
            _make_content_chunk("b"),
        ])

        mock_resp = _make_stream_response(sse_lines)

        with patch.object(
            runner._client,
            "stream",
            return_value=_async_context_manager(mock_resp),
        ):
            result = await runner.chat_completion(
                messages=[{"role": "user", "content": "test"}],
                model="test-model",
            )

        assert result.completion_tokens == 2
        assert result.prompt_tokens is None


class TestClose:
    @pytest.mark.asyncio
    async def test_closes_client(self, runner):
        with patch.object(runner._client, "aclose", new_callable=AsyncMock) as mock_close:
            await runner.close()
            mock_close.assert_called_once()


# --- Helpers ---

def _make_stream_response(sse_lines: list[str]):
    """Create a mock response that supports async iteration of SSE lines."""
    mock = MagicMock()
    mock.status_code = 200

    async def _aiter_lines():
        for line in sse_lines:
            yield line

    mock.aiter_lines = _aiter_lines
    return mock


class _async_context_manager:
    """Wraps a mock to work as an async context manager."""

    def __init__(self, mock_resp):
        self._resp = mock_resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *args):
        pass
