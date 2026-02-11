"""Tests for GPU pricing module."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

from llm_inf_bench.providers.runpod.pricing import (
    GPU_PRICING,
    check_pricing_staleness,
    estimate_cost,
    get_known_gpu_types,
)


class TestGetKnownGPUTypes:
    def test_returns_list(self):
        types = get_known_gpu_types()
        assert isinstance(types, list)
        assert len(types) > 0

    def test_contains_expected_gpus(self):
        types = get_known_gpu_types()
        assert "A100-80GB" in types
        assert "H100-80GB" in types
        assert "RTX 4090" in types


class TestCheckPricingStaleness:
    def test_fresh_data(self):
        with patch("llm_inf_bench.providers.runpod.pricing.date") as mock_date:
            mock_date.today.return_value = date(2026, 2, 11)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            is_stale, age = check_pricing_staleness()
            assert not is_stale
            assert age <= 30

    def test_stale_data(self):
        with patch("llm_inf_bench.providers.runpod.pricing.date") as mock_date:
            mock_date.today.return_value = date(2026, 6, 1)
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            is_stale, age = check_pricing_staleness()
            assert is_stale
            assert age > 30


class TestEstimateCost:
    def test_known_gpu(self):
        cost = estimate_cost("A100-80GB", 1, 60)
        assert cost is not None
        # Community rate for A100-80GB is $1.99/hr
        assert cost == pytest.approx(1.99)

    def test_multi_gpu(self):
        cost = estimate_cost("A100-80GB", 2, 60)
        assert cost is not None
        assert cost == pytest.approx(1.99 * 2)

    def test_partial_hour(self):
        cost = estimate_cost("A100-80GB", 1, 30)
        assert cost is not None
        assert cost == pytest.approx(1.99 / 2)

    def test_secure_tier(self):
        cost = estimate_cost("A100-80GB", 1, 60, tier="secure")
        assert cost is not None
        assert cost == pytest.approx(3.29)

    def test_unknown_gpu_returns_none(self):
        assert estimate_cost("FAKE-GPU", 1, 60) is None

    def test_all_gpus_have_pricing(self):
        for gpu_type in get_known_gpu_types():
            cost = estimate_cost(gpu_type, 1, 60)
            assert cost is not None, f"No pricing for {gpu_type}"
            assert cost > 0


# Need pytest import for pytest.approx
import pytest
