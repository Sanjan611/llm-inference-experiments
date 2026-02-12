# CLAUDE.md

## Project Overview

A benchmarking and observability framework for measuring LLM inference performance using vLLM and SGLang on RunPod GPUs. The CLI tool is called `llm-inf-bench`.

The framework provisions cloud GPU instances, deploys an inference server, sends structured workloads, collects client-side and server-side metrics, and produces JSON result files for analysis.

## Design Documents

- **PROJECT_OVERVIEW.md** — Motivation, features, architecture, key concepts (KV cache, prefix caching, PagedAttention vs RadixAttention)
- **INTERFACE.md** — CLI commands, YAML config format, progress output, cost confirmation flow, directory structure
- **PROOF_OF_CONCEPTS.md** — Standalone validation scripts in `poc/` (RunPod lifecycle, vLLM/SGLang deployment, chat completions, Prometheus scraping)
- **docs/IMPLEMENTATION_PLAN.md** — Phased build plan with module breakdown and verification criteria

## Architecture

```
CLI (cli.py)
 ├─ Config loading (config/loader.py → schema.py → validation.py)
 ├─ Provider pod lifecycle (providers/runpod/)
 ├─ Runner health check + streaming chat completions (runners/)
 ├─ Background GPU metrics scraping (metrics/gpu.py)
 ├─ Workload execution (workloads/)
 ├─ Aggregation + JSON persistence (metrics/aggregator.py, storage.py)
 └─ Rich console output (output/console.py, summary.py)
```

The orchestration flow in `cli.py:_execute_run` follows four phases: provisioning → model loading/health check → benchmark execution → cleanup. Partial results are saved on interruption.

## Module Layout

```
src/llm_inf_bench/
├── cli.py          # Typer CLI entry point and orchestration
├── config/         # YAML loading, Pydantic v2 schema, cross-field validation
├── runners/        # Inference framework runners (vLLM, SGLang) via OpenAI-compatible API
├── workloads/      # Execution strategies (single-request sequential; batch/concurrent planned)
├── providers/      # Cloud GPU lifecycle management (RunPod)
├── metrics/        # Request timing, GPU metrics scraping, aggregation, JSON persistence
└── output/         # Rich console progress display and summary/comparison rendering
```

## Key Abstractions

- **Runner** (`runners/base.py`) — Framework-agnostic interface. `OpenAICompatibleRunner` handles streaming chat completions with per-token timing (TTFT, inter-token latency, e2e). vLLM/SGLang runners are thin subclasses. Factory: `create_runner(framework, base_url, model)`.
- **Workload** (`workloads/base.py`) — Execution strategy. Takes a runner, loads prompts from JSONL, returns `list[RequestResult]`. Has `on_request_complete` callback for progress updates.
- **Provider** (`providers/base.py`) — Infrastructure lifecycle. Returns a `ProviderPod` with `base_url`. RunPod implementation handles API key resolution, polling, and proxy URL construction.
- **GpuMetricsScraper** (`metrics/gpu.py`) — Background asyncio task that polls `/metrics` at configurable intervals. Framework-aware metric name mapping (vLLM vs SGLang Prometheus conventions). Returns `GpuTimeSeries` on stop.
- **ExperimentConfig** (`config/schema.py`) — Root Pydantic v2 model. Child models: `ModelConfig`, `InfrastructureConfig`, `WorkloadConfig`, `FrameworkOptions`, `MetricsConfig`.

## Patterns and Conventions

**Async-first** — All I/O uses `async`/`await`. HTTP via `httpx.AsyncClient`. Background tasks via `asyncio.Task`.

**ABC + Factory** — Abstract base classes define interfaces (Runner, Workload, Provider). Factory functions map string identifiers to concrete classes.

**Error capture, not raise** — Request-level errors are captured in `RequestResult.error` rather than raised. Module-level errors use typed exceptions (`RunnerError`, `HealthCheckTimeout`, `ProvisioningError`, `ConfigValidationError`).

**Dataclasses for data, Pydantic for config** — Runtime data (RequestResult, GpuSample, ProviderPod) uses `@dataclass`. User-facing configuration uses Pydantic v2 models with validation.

**Config inheritance** — YAML files support an `extends` field for single-level inheritance with deep merge. Child values override parent values. Paths resolve relative to the config file directory.

**Prompt format** — JSONL files where each line is `{"messages": [{"role": "...", "content": "..."}]}`. Prompts cycle if `count` exceeds file size.

**Type hints** — `from __future__ import annotations` everywhere. PEP 604 union syntax (`X | None`). Strict mypy enabled.

**Logging** — `logging.getLogger(__name__)` per module. Info for phase transitions, warning for non-fatal issues, debug for details.

**Rich output** — All CLI output through `rich.console.Console`. Panels, tables, progress bars with live field updates. Color-coded deltas in comparisons.

## Build and Test

```bash
uv sync                     # Install dependencies
uv run llm-inf-bench --help # Run the CLI
uv run pytest               # Run tests
uv run mypy src/            # Type checking (strict mode)
uv run ruff check src/      # Lint
uv run ruff format src/     # Format
```

Test fixtures live in `tests/conftest.py` — includes `minimal_experiment_dict`, `minimal_experiment_yaml`, and `base_experiment_yaml` for config testing. Tests use `pytest-asyncio` with `asyncio_mode = "auto"`.

## Directory Conventions

- `experiments/base/` — Base YAML configs for inheritance (vllm-defaults, sglang-defaults)
- `experiments/examples/` — Ready-to-use experiment configs
- `prompts/` — JSONL prompt files referenced by configs
- `results/` — JSON output files (git-ignored), one per run, named `{name}-{YYYYMMDD-HHmmss}.json`
- `poc/` — Standalone proof-of-concept scripts (not part of the framework)
- `~/.llm-inf-bench/config.yaml` — User app config (RunPod API key, defaults)
