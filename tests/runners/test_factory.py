"""Tests for the runner factory."""

from __future__ import annotations

import pytest

from llm_inf_bench.runners import create_runner
from llm_inf_bench.runners.sglang import SGLangRunner
from llm_inf_bench.runners.vllm import VLLMRunner


class TestCreateRunner:
    def test_returns_vllm_runner(self):
        runner = create_runner("vllm", "http://localhost:8000", "m")
        assert isinstance(runner, VLLMRunner)

    def test_returns_sglang_runner(self):
        runner = create_runner("sglang", "http://localhost:8000", "m")
        assert isinstance(runner, SGLangRunner)

    def test_unknown_framework_raises(self):
        with pytest.raises(ValueError, match="Unknown framework"):
            create_runner("unknown", "http://localhost:8000", "m")

    def test_error_message_lists_supported(self):
        with pytest.raises(ValueError, match="sglang.*vllm"):
            create_runner("bad", "http://localhost:8000", "m")
