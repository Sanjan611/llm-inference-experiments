# llm-inf-bench

A CLI tool for benchmarking LLM inference performance on cloud GPUs. Point it at a YAML config, and it provisions a GPU on [RunPod](https://www.runpod.io/), deploys an inference server ([vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/sgl-project/sglang)), runs your workload, collects detailed latency and GPU metrics, and tears everything down when it's done.

```
llm-inf-bench run experiments/examples/baseline-vllm.yaml
```

## What You Get

Every run produces a JSON results file with:

- **Latency breakdown** — Time to first token (TTFT), inter-token latency, end-to-end latency, all at p50/p95/p99
- **Throughput** — Tokens per second and requests per second
- **GPU metrics** — Utilization, memory usage, KV cache occupancy, prefix cache hit rates (sampled from the server's Prometheus endpoint)
- **Per-request detail** — Individual timings for every request, so you can analyze distributions yourself

```
Summary:
┌──────────────────────────────────────────────┐
│  Requests:     100        Errors: 0          │
│  Duration:     42.3s                         │
│  Throughput:   138.7 tok/s                   │
│                                              │
│  TTFT:         p50=42ms  p95=78ms  p99=112ms │
│  Latency:      p50=298ms p95=412ms p99=523ms │
│  TBT:          p50=12ms  p95=18ms  p99=24ms  │
└──────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- A [RunPod](https://www.runpod.io/) account and API key

### Install

```bash
git clone https://github.com/Sanjan611/llm-inference-experiments.git
cd llm-inference-experiments
uv sync
```

### Configure

```bash
uv run llm-inf-bench init          # creates ~/.llm-inf-bench/config.yaml
uv run llm-inf-bench doctor        # checks Python, API key, RunPod connectivity
```

Set your RunPod API key in `~/.llm-inf-bench/config.yaml`:

```yaml
runpod:
  api_key: "your-key-here"
```

### Run an Experiment

```bash
# Quick smoke test (~$0.10, small model, cheap GPU)
uv run llm-inf-bench run experiments/examples/quick-smoke-test.yaml

# Baseline latency for Llama 3 8B on an A100
uv run llm-inf-bench run experiments/examples/baseline-vllm.yaml

# Same model on SGLang for comparison
uv run llm-inf-bench run experiments/examples/baseline-sglang.yaml
```

The CLI shows live progress during each phase:

```
[1/4] Provisioning RunPod instance...
[2/4] Loading model...
[3/4] Running benchmark...
      Progress: ████████████████████░░░░░  80/100 (80%)
      ├─ Throughput:    142.3 tok/s
      ├─ Avg TTFT:      45.2 ms
      └─ Avg Latency:   312.5 ms
[4/4] Collecting results...
      ✓ Metrics saved to results/baseline-vllm-a100-20260212-143022.json
      ✓ Instance terminated
```

### Browse Results

```bash
uv run llm-inf-bench results list                    # list all runs
uv run llm-inf-bench results show <run-id>           # view a single run
uv run llm-inf-bench results compare <id1> <id2>     # side-by-side comparison
```

## Experiment Configuration

Experiments are defined in YAML. A minimal config:

```yaml
name: "my-experiment"

model:
  name: "meta-llama/Meta-Llama-3-8B-Instruct"

framework: "vllm"   # or "sglang"

infrastructure:
  provider: "runpod"
  gpu_type: "A100-80GB"
  gpu_count: 1

workload:
  type: "single"
  requests:
    source: "prompts/baseline.jsonl"
    count: 50
  parameters:
    max_tokens: 256
    temperature: 0.7

metrics:
  collect_gpu_metrics: true
```

Configs support inheritance — define shared defaults in a base file and override per-experiment:

```yaml
extends: "../base/vllm-defaults.yaml"
name: "my-experiment"

model:
  name: "meta-llama/Meta-Llama-3-8B-Instruct"

infrastructure:
  gpu_type: "A100-80GB"
```

See [`experiments/examples/`](experiments/examples/) for ready-to-use configs.

## Workload Types

| Type | Description | Example Use |
|------|-------------|-------------|
| `single` | Sequential, one request at a time | Baseline latency measurement |
| `batch` | Fixed batch size, dispatched concurrently | Throughput under controlled load |
| `concurrent` | Worker pool with N concurrent slots | Simulated multi-user scenarios |

### Parameter Sweeps

Vary a parameter across multiple values in a single experiment:

```yaml
workload:
  type: "concurrent"
  concurrency: 1
  requests:
    source: "prompts/baseline.jsonl"
    count: 32
  sweep:
    concurrency: [1, 2, 4, 8, 16]
```

This runs 5 iterations, one per concurrency level, and saves a separate result for each.

## Useful Options

```bash
# Validate config without provisioning anything
uv run llm-inf-bench run experiment.yaml --dry-run

# Skip the cost confirmation prompt
uv run llm-inf-bench run experiment.yaml --confirm

# Use an already-running server (no provisioning/teardown)
uv run llm-inf-bench run experiment.yaml --server-url https://your-pod-url:8000
```

## Project Structure

```
experiments/       YAML experiment configs (base defaults + examples)
prompts/           JSONL prompt files referenced by configs
results/           JSON output files, one per run (git-ignored)
src/llm_inf_bench/ Framework source code
tests/             Test suite
poc/               Standalone proof-of-concept scripts
```

## Development

```bash
uv sync                         # install dependencies
uv run pytest                    # run tests
uv run mypy src/                 # type checking (strict)
uv run ruff check src/           # lint
uv run ruff format src/          # format
```

## License

MIT
