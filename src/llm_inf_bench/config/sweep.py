"""Sweep expansion — generates multiple experiment configs from sweep parameters."""

from __future__ import annotations

import itertools
from typing import Any

from llm_inf_bench.config.schema import ExperimentConfig


def expand_sweep(
    config: ExperimentConfig,
) -> list[tuple[ExperimentConfig, dict[str, Any]]]:
    """Expand an experiment config into a list of (config, params) tuples.

    If the config has no sweep section, returns ``[(config, {})]``.

    For each combination of sweep parameter values, produces a deep copy
    with the parameter overridden and a descriptive suffix appended to the name.

    If sweeping over concurrency with ``type: "single"``, the type is
    auto-promoted to ``"concurrent"``.
    """
    sweep = config.workload.sweep
    if sweep is None:
        return [(config, {})]

    concurrency_values = sweep.concurrency or []
    batch_size_values = sweep.batch_size or []

    if not concurrency_values and not batch_size_values:
        return [(config, {})]

    # Build axis lists for cartesian product
    axes: list[list[tuple[str, int]]] = []
    if concurrency_values:
        axes.append([("concurrency", v) for v in concurrency_values])
    if batch_size_values:
        axes.append([("batch_size", v) for v in batch_size_values])

    variations: list[tuple[ExperimentConfig, dict[str, Any]]] = []

    for combo in itertools.product(*axes):
        params: dict[str, Any] = dict(combo)
        variation = config.model_copy(deep=True)

        # Build name suffix
        suffix_parts: list[str] = []
        for key, value in params.items():
            if key == "concurrency":
                suffix_parts.append(f"c{value}")
                variation.workload.concurrency = value
                # Auto-promote single -> concurrent
                if variation.workload.type == "single":
                    variation.workload.type = "concurrent"
            elif key == "batch_size":
                suffix_parts.append(f"b{value}")
                variation.workload.batch_size = value
                # Auto-promote single -> batch
                if variation.workload.type == "single":
                    variation.workload.type = "batch"

        variation.name = f"{config.name}-{'-'.join(suffix_parts)}"
        variations.append((variation, params))

    return variations
