"""YAML loading with ``extends`` inheritance and path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llm_inf_bench.config.schema import AppConfig, ExperimentConfig
from llm_inf_bench.config.validation import ConfigValidationError, validate_experiment

APP_CONFIG_DIR: Path = Path.home() / ".llm-inf-bench"
APP_CONFIG_PATH: Path = APP_CONFIG_DIR / "config.yaml"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* on top of *base*. Override wins for leaf values."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(data).__name__}")
    return data


def resolve_extends(data: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    """Resolve single-level ``extends`` inheritance.

    Loads the parent config, deep-merges the child on top, and removes the
    ``extends`` key. Does NOT recurse into the parent's ``extends``.
    """
    extends = data.get("extends")
    if extends is None:
        return data

    parent_path = (config_dir / extends).resolve()
    if not parent_path.exists():
        raise FileNotFoundError(f"Parent config not found: {parent_path}")

    parent_data = load_yaml(parent_path)
    # Remove extends from parent so it doesn't propagate
    parent_data.pop("extends", None)

    # Child overrides parent
    child_data = {k: v for k, v in data.items() if k != "extends"}
    return deep_merge(parent_data, child_data)


def _resolve_source_path(data: dict[str, Any], config_dir: Path) -> None:
    """Resolve workload.requests.source relative to the config file's directory."""
    workload = data.get("workload")
    if not isinstance(workload, dict):
        return
    requests = workload.get("requests")
    if not isinstance(requests, dict):
        return
    source = requests.get("source")
    if source is None:
        return

    source_path = Path(source)
    if not source_path.is_absolute():
        requests["source"] = str((config_dir / source_path).resolve())


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load an experiment config: YAML -> extends -> path resolution -> validate.

    Raises:
        FileNotFoundError: If the config or parent file doesn't exist.
        ValueError: If the YAML is malformed.
        pydantic.ValidationError: If structural validation fails.
        ConfigValidationError: If semantic validation fails.
    """
    config_path = Path(path).resolve()
    config_dir = config_path.parent

    data = load_yaml(config_path)
    data = resolve_extends(data, config_dir)
    _resolve_source_path(data, config_dir)

    config = ExperimentConfig(**data)

    errors = validate_experiment(config)
    if errors:
        raise ConfigValidationError(errors)

    return config


def load_app_config() -> AppConfig:
    """Load application config from ~/.llm-inf-bench/config.yaml.

    Returns defaults if the file doesn't exist — avoids forcing ``init`` first.
    """
    if not APP_CONFIG_PATH.exists():
        return AppConfig()
    data = load_yaml(APP_CONFIG_PATH)
    return AppConfig(**data)
