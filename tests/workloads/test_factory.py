"""Tests for workload factory."""

from __future__ import annotations

import pytest

from llm_inf_bench.workloads import (
    BatchWorkload,
    ConcurrentWorkload,
    SingleWorkload,
    create_workload,
)


def _make_prompts(n: int = 4) -> list[list[dict[str, str]]]:
    return [[{"role": "user", "content": f"Prompt {i}"}] for i in range(n)]


class TestCreateWorkload:
    def test_creates_single(self):
        wl = create_workload("single", _make_prompts(), model="test")
        assert isinstance(wl, SingleWorkload)

    def test_creates_batch(self):
        wl = create_workload("batch", _make_prompts(), model="test", batch_size=2)
        assert isinstance(wl, BatchWorkload)

    def test_creates_concurrent(self):
        wl = create_workload("concurrent", _make_prompts(), model="test", concurrency=4)
        assert isinstance(wl, ConcurrentWorkload)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown workload type"):
            create_workload("unknown", _make_prompts(), model="test")

    def test_batch_without_batch_size_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            create_workload("batch", _make_prompts(), model="test")

    def test_concurrent_without_concurrency_raises(self):
        with pytest.raises(ValueError, match="concurrency"):
            create_workload("concurrent", _make_prompts(), model="test")

    def test_passes_max_tokens_and_temperature(self):
        wl = create_workload(
            "single",
            _make_prompts(),
            model="test",
            max_tokens=512,
            temperature=0.3,
        )
        assert isinstance(wl, SingleWorkload)
        assert wl._max_tokens == 512
        assert wl._temperature == 0.3

    def test_passes_callback(self):
        from unittest.mock import MagicMock

        cb = MagicMock()
        wl = create_workload(
            "single", _make_prompts(), model="test", on_request_complete=cb
        )
        assert wl._on_request_complete is cb
