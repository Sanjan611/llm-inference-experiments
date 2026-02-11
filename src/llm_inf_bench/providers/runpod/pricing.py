"""Hardcoded GPU pricing with staleness checking.

Pricing sourced from INTERFACE.md. RunPod GPU type IDs should be verified
against ``runpod.get_gpus()`` — the exact strings may change over time.
"""

from __future__ import annotations

from datetime import date

PRICING_LAST_UPDATED: date = date(2026, 2, 11)

# gpu_type -> (community_$/hr, secure_$/hr)
GPU_PRICING: dict[str, tuple[float, float]] = {
    "H100-80GB": (3.99, 5.49),
    "A100-80GB": (1.99, 3.29),
    "A100-40GB": (1.64, 2.49),
    "L40S": (1.14, 1.84),
    "A40": (0.79, 1.24),
    "RTX 4090": (0.69, 0.99),
    "RTX 3090": (0.44, 0.69),
    "RTX 4000 Ada": (0.29, 0.49),
}

# Maps user-facing config names to RunPod SDK gpu_type_id strings.
# NOTE: Verify these against runpod.get_gpus() — IDs may drift between SDK versions.
GPU_TYPE_TO_RUNPOD_ID: dict[str, str] = {
    "H100-80GB": "NVIDIA H100 80GB HBM3",
    "A100-80GB": "NVIDIA A100 80GB PCIe",
    "A100-40GB": "NVIDIA A100-PCIE-40GB",
    "L40S": "NVIDIA L40S",
    "A40": "NVIDIA A40",
    "RTX 4090": "NVIDIA GeForce RTX 4090",
    "RTX 3090": "NVIDIA GeForce RTX 3090",
    "RTX 4000 Ada": "NVIDIA RTX 4000 Ada Generation",
}

_STALENESS_THRESHOLD_DAYS = 30


def get_known_gpu_types() -> list[str]:
    """Return the list of GPU types we have pricing for."""
    return list(GPU_PRICING.keys())


def check_pricing_staleness() -> tuple[bool, int]:
    """Check if pricing data is stale.

    Returns:
        (is_stale, age_days) — stale if older than 30 days.
    """
    age_days = (date.today() - PRICING_LAST_UPDATED).days
    return age_days > _STALENESS_THRESHOLD_DAYS, age_days


def estimate_cost(
    gpu_type: str,
    gpu_count: int,
    estimated_minutes: float,
    tier: str = "community",
) -> float | None:
    """Estimate cost for a GPU configuration.

    Returns:
        Estimated cost in USD, or None if the gpu_type is unknown.
    """
    pricing = GPU_PRICING.get(gpu_type)
    if pricing is None:
        return None
    community_rate, secure_rate = pricing
    rate = secure_rate if tier == "secure" else community_rate
    return rate * gpu_count * (estimated_minutes / 60.0)
