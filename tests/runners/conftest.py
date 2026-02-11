"""Shared SSE helpers for runner tests."""

from __future__ import annotations

import json
from unittest.mock import MagicMock


def make_sse_lines(chunks: list[dict]) -> list[str]:
    """Build SSE-formatted lines from a list of chunk dicts."""
    lines = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}")
        lines.append("")  # blank line between events
    lines.append("data: [DONE]")
    return lines


def make_content_chunk(content: str, index: int = 0) -> dict:
    return {
        "choices": [{"index": index, "delta": {"content": content}}],
    }


def make_usage_chunk(prompt: int, completion: int) -> dict:
    return {
        "choices": [],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


def make_stream_response(sse_lines: list[str]):
    """Create a mock response that supports async iteration of SSE lines."""
    mock = MagicMock()
    mock.status_code = 200

    async def _aiter_lines():
        for line in sse_lines:
            yield line

    mock.aiter_lines = _aiter_lines
    return mock


class async_context_manager:
    """Wraps a mock to work as an async context manager."""

    def __init__(self, mock_resp):
        self._resp = mock_resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *args):
        pass
