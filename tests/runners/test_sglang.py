"""Tests for SGLangRunner."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from llm_inf_bench.runners.sglang import SGLangRunner

from .conftest import (
    async_context_manager,
    make_content_chunk,
    make_sse_lines,
    make_stream_response,
    make_usage_chunk,
)


@pytest.fixture
def runner():
    return SGLangRunner(base_url="http://localhost:8000", model="test-model")


class TestDefaults:
    def test_health_interval_is_10(self, runner):
        assert runner._health_interval == 10.0

    def test_health_timeout_is_600(self, runner):
        assert runner._health_timeout == 600.0


class TestStreaming:
    @pytest.mark.asyncio
    async def test_basic_streaming(self, runner):
        sse_lines = make_sse_lines([
            make_content_chunk("Hello"),
            make_usage_chunk(prompt=5, completion=1),
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
        assert result.prompt_tokens == 5
        assert result.completion_tokens == 1
