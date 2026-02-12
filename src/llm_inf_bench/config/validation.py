"""Semantic cross-field validation for experiment configs.

Goes beyond Pydantic's structural checks to enforce business rules
like GPU type existence and workload-type-specific field requirements.
"""

from __future__ import annotations

from llm_inf_bench.config.schema import ExperimentConfig
from llm_inf_bench.providers.runpod.pricing import get_known_gpu_types


class ConfigValidationError(Exception):
    """Raised when semantic validation fails.

    Attributes:
        errors: All validation errors found (not just the first).
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Config validation failed: {'; '.join(errors)}")


def validate_experiment(config: ExperimentConfig) -> list[str]:
    """Validate an experiment config beyond structural checks.

    Returns:
        Empty list if valid, otherwise a list of error messages.
    """
    errors: list[str] = []

    # 1. GPU type must be in the pricing table
    known = get_known_gpu_types()
    if config.infrastructure.gpu_type not in known:
        errors.append(
            f"Unknown gpu_type '{config.infrastructure.gpu_type}'. "
            f"Known types: {', '.join(known)}"
        )

    # 2. Workload type "batch" requires batch_size
    wl = config.workload
    if wl.type == "batch" and wl.batch_size is None:
        errors.append("Workload type 'batch' requires 'batch_size' to be set")

    # 3. Workload type "concurrent" requires concurrency
    if wl.type == "concurrent" and wl.concurrency is None:
        errors.append("Workload type 'concurrent' requires 'concurrency' to be set")

    # 4. Workload type "multi_turn" requires conversation
    if wl.type == "multi_turn" and wl.conversation is None:
        errors.append("Workload type 'multi_turn' requires 'conversation' to be set")

    return errors
