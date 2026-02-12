"""Tests for RunPod client."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from llm_inf_bench.providers.base import PodTimeoutError, ProvisioningError
from llm_inf_bench.providers.runpod.client import RunPodProvider


class TestAPIKeyResolution:
    def test_explicit_key(self):
        with patch("llm_inf_bench.providers.runpod.client.runpod_sdk"):
            provider = RunPodProvider(api_key="explicit-key")
        # Should not raise

    def test_env_var_key(self, monkeypatch):
        monkeypatch.setenv("RUNPOD_API_KEY", "env-key")
        with patch("llm_inf_bench.providers.runpod.client.runpod_sdk"):
            provider = RunPodProvider()

    def test_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        with (
            patch(
                "llm_inf_bench.config.loader.load_app_config",
                return_value=MagicMock(runpod=MagicMock(api_key="")),
            ),
            patch("llm_inf_bench.providers.runpod.client.runpod_sdk"),
        ):
            with pytest.raises(ProvisioningError, match="No RunPod API key"):
                RunPodProvider()


class TestCheckAPIKeyConfigured:
    def test_env_var_present(self, monkeypatch):
        monkeypatch.setenv("RUNPOD_API_KEY", "test")
        assert RunPodProvider.check_api_key_configured() is True

    def test_no_key(self, monkeypatch):
        monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
        with patch(
            "llm_inf_bench.config.loader.load_app_config",
            return_value=MagicMock(runpod=MagicMock(api_key="")),
        ):
            assert RunPodProvider.check_api_key_configured() is False


class TestGetUrl:
    def test_url_format(self):
        with patch("llm_inf_bench.providers.runpod.client.runpod_sdk"):
            provider = RunPodProvider(api_key="test")
        url = provider.get_url("abc123", 8000)
        assert url == "https://abc123-8000.proxy.runpod.net"


class TestCleanupAll:
    def test_cleanup_terminates_managed_pods(self):
        with patch("llm_inf_bench.providers.runpod.client.runpod_sdk") as mock_sdk:
            provider = RunPodProvider(api_key="test")
            provider._managed_pods = {"pod1", "pod2"}
            provider.cleanup_all()
            assert mock_sdk.terminate_pod.call_count == 2

    def test_cleanup_logs_failures(self):
        with patch("llm_inf_bench.providers.runpod.client.runpod_sdk") as mock_sdk:
            mock_sdk.terminate_pod.side_effect = Exception("API error")
            provider = RunPodProvider(api_key="test")
            provider._managed_pods = {"pod1"}
            # Should not raise
            provider.cleanup_all()
