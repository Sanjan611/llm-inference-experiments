# Interface Design

This document describes the user interface and workflow for `llm-inf-bench`, the CLI tool for running LLM inference experiments.

## CLI Commands

### Setup & Configuration

```bash
# Initialize configuration directory and check dependencies
llm-inf-bench init

# Set configuration values
llm-inf-bench config set runpod.api_key <key>

# Verify environment is correctly configured
llm-inf-bench doctor
```

### Running Experiments

```bash
# Run an experiment (auto-provisions infrastructure, auto-terminates after)
llm-inf-bench run experiments/baseline.yaml

# Keep server running after experiment completes
llm-inf-bench run experiments/baseline.yaml --keep-server

# Use an existing server instead of provisioning
llm-inf-bench run experiments/baseline.yaml --server-url <url>

# Validate configuration without running
llm-inf-bench run experiments/baseline.yaml --dry-run

# Skip cost confirmation prompt
llm-inf-bench run experiments/baseline.yaml --confirm

# Resume a failed or interrupted run
llm-inf-bench run experiments/baseline.yaml --resume <run-id>
```

### Server Management

For manual server control when not using auto-provisioning:

```bash
# Check status of running servers
llm-inf-bench server status

# Stop a specific server
llm-inf-bench server stop <pod-id>
```

### Results

```bash
# List completed experiment runs
llm-inf-bench results list

# Show summary statistics for a run
llm-inf-bench results show <run-id>

# Compare two runs side-by-side
llm-inf-bench results compare <id1> <id2>

# Export results for external analysis
llm-inf-bench results export <id> --format csv
```

### Maintenance

```bash
# Update hardcoded GPU pricing data
llm-inf-bench update-pricing
```

---

## Configuration File Format

Configuration files use YAML format. Files can inherit from base configurations using the `extends` field.

### Basic Configuration

```yaml
# experiments/baseline-llama3-8b.yaml
name: "baseline-llama3-8b-vllm"
description: "Baseline latency measurements for Llama 3 8B on vLLM"

model:
  name: "meta-llama/Meta-Llama-3-8B-Instruct"
  quantization: null  # or "awq", "gptq", etc.

framework: "vllm"  # or "sglang"
framework_options:
  max_model_len: 4096
  gpu_memory_utilization: 0.9
  enable_prefix_caching: true

infrastructure:
  provider: "runpod"
  gpu_type: "A100-80GB"
  gpu_count: 1

workload:
  type: "single"  # single | batch | concurrent | multi_turn
  requests:
    source: "prompts/baseline.jsonl"  # or inline list
    count: 100
  parameters:
    max_tokens: 256
    temperature: 0.7

metrics:
  collect_gpu_metrics: true
  sample_interval_ms: 100
  output_dir: "results/"
```

### Configuration Inheritance

Base configurations can be defined for reuse:

```yaml
# experiments/base/vllm-defaults.yaml
framework: "vllm"
framework_options:
  gpu_memory_utilization: 0.9
  enable_prefix_caching: true
```

Child configurations inherit and override:

```yaml
# experiments/my-experiment.yaml
extends: "base/vllm-defaults.yaml"
name: "my-experiment"

model:
  name: "meta-llama/Meta-Llama-3-8B-Instruct"

# framework and framework_options inherited from base
```

### Parameter Sweeps

Run multiple variations sequentially in a single experiment:

```yaml
# experiments/sweep-concurrency.yaml
extends: "base/vllm-defaults.yaml"
name: "concurrency-sweep"

workload:
  type: "concurrent"
  sweep:
    concurrency: [1, 4, 8, 16, 32]
  requests:
    source: "prompts/baseline.jsonl"
    count: 100
```

### Workload Types

**Single requests** — One request at a time, baseline latency:

```yaml
workload:
  type: "single"
  requests:
    source: "prompts/baseline.jsonl"
    count: 100
```

**Batch requests** — Controlled batch sizes:

```yaml
workload:
  type: "batch"
  batch_size: 8
  requests:
    source: "prompts/baseline.jsonl"
    count: 100
```

**Concurrent requests** — Simulated multi-user load:

```yaml
workload:
  type: "concurrent"
  concurrency: 16
  requests:
    source: "prompts/baseline.jsonl"
    count: 100
```

**Multi-turn / Agentic** — Conversation flows with context accumulation:

```yaml
workload:
  type: "multi_turn"
  conversation:
    turns: 5
    system_prompt: "prompts/agent-system.txt"
    user_messages: "prompts/agent-queries.jsonl"
    shared_prefix: true  # Test prefix caching effectiveness
```

---

## Progress Output

### Phase 1: Infrastructure Setup

```
[1/4] Provisioning RunPod instance...
      GPU: A100-80GB × 1
      Image: vllm/vllm-openai:latest
      ✓ Instance created (pod_abc123)
      ✓ Waiting for ready... (42s)
      ✓ Server URL: https://abc123-8000.proxy.runpod.net
```

### Phase 2: Model Loading

```
[2/4] Loading model...
      Model: meta-llama/Meta-Llama-3-8B-Instruct
      ✓ Model loaded (18.2s)
      ✓ Health check passed
      Memory: 15.2 GB / 80 GB
```

### Phase 3: Benchmark Execution

```
[3/4] Running benchmark...
      Workload: concurrent (16 workers)
      Requests: 100

      Progress: ████████████████████░░░░░  80/100 (80%)

      Live Metrics:
      ├─ Throughput:    142.3 tok/s
      ├─ Avg TTFT:      45.2 ms
      ├─ Avg Latency:   312.5 ms
      └─ GPU Util:      87%
```

### Phase 4: Collection & Cleanup

```
[4/4] Collecting results...
      ✓ Metrics saved to results/baseline-llama3-8b-vllm-20240115-143022.json
      ✓ GPU metrics captured (523 samples)
      ✓ Instance terminated

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

---

## Cost Confirmation

Before provisioning infrastructure, the CLI displays estimated cost and requires confirmation:

```
$ llm-inf-bench run experiments/a100-sweep.yaml

Experiment: concurrency-sweep-a100
──────────────────────────────────────────────────
  GPU:        A100-80GB × 1
  Workload:   5 variations × 100 requests each
  Est. Time:  ~15-25 minutes

  Estimated Cost: $0.82 - $1.37
  (A100-80GB: $1.99/hr community, $3.29/hr secure)
──────────────────────────────────────────────────

Proceed? [y/N]:
```

- Configs estimated under $0.50 skip the prompt automatically
- Use `--confirm` flag to always skip the prompt
- GPU pricing is hardcoded and sourced from official RunPod documentation
- Stale pricing data (>30 days) triggers a warning

```
⚠ GPU pricing data is 45 days old. Run `llm-inf-bench update-pricing` to refresh.
```

---

## Partial Results & Resume

When a run fails or is interrupted, partial results are saved with checkpoint information:

```json
{
  "run_id": "baseline-20240115-143022",
  "status": "partial",
  "completed_requests": 67,
  "total_requests": 100,
  "failure_reason": "Server connection lost",
  "checkpoint": {
    "last_request_index": 66,
    "server_pod_id": "pod_abc123"
  },
  "results": { ... }
}
```

Resume a failed run:

```
$ llm-inf-bench run experiments/baseline.yaml --resume baseline-20240115-143022

Resuming run: baseline-20240115-143022
  Previously completed: 67/100 requests
  Remaining: 33 requests

Checking server status...
  ✗ Original server terminated
  → Provisioning new instance...
```

---

## Environment Setup

### Installation

```bash
git clone <repo>
cd llm-inference-experiments
pip install -e ".[dev]"
```

### Initial Configuration

```bash
llm-inf-bench init
```

Creates `~/.llm-inf-bench/config.yaml`:

```yaml
runpod:
  api_key: ""
defaults:
  output_dir: "./results"
```

### Verification

```bash
llm-inf-bench doctor
```

Checks:
- Python 3.10+
- RunPod API key configured
- Network connectivity to RunPod
- Optional: Prometheus available

---

## Directory Structure

```
llm-inference-experiments/
├── pyproject.toml
├── src/
│   └── llm_inf_bench/
│       ├── __init__.py
│       ├── cli.py                    # Typer CLI entry point
│       ├── config/
│       │   ├── __init__.py
│       │   ├── schema.py             # Pydantic models for config
│       │   ├── loader.py             # YAML loading, inheritance
│       │   └── validation.py
│       ├── runners/
│       │   ├── __init__.py
│       │   ├── base.py               # Abstract runner interface
│       │   ├── vllm.py               # vLLM client
│       │   └── sglang.py             # SGLang client
│       ├── providers/
│       │   ├── __init__.py
│       │   ├── base.py               # Abstract provider interface
│       │   └── runpod/
│       │       ├── __init__.py
│       │       ├── client.py         # RunPod API wrapper
│       │       ├── pricing.py        # Hardcoded GPU costs
│       │       └── templates.py      # Pod config templates
│       ├── workloads/
│       │   ├── __init__.py
│       │   ├── base.py               # Abstract workload interface
│       │   ├── single.py
│       │   ├── batch.py
│       │   ├── concurrent.py
│       │   └── multi_turn.py
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── collector.py          # Latency, throughput collection
│       │   ├── gpu.py                # GPU metrics polling
│       │   ├── aggregator.py         # Percentiles, summaries
│       │   └── storage.py            # JSON/JSONL output
│       └── output/
│           ├── __init__.py
│           ├── console.py            # Rich progress display
│           └── summary.py            # Post-run summary generation
├── experiments/
│   ├── examples/
│   │   ├── baseline-vllm.yaml
│   │   ├── baseline-sglang.yaml
│   │   └── concurrency-sweep.yaml
│   └── base/
│       ├── vllm-defaults.yaml
│       └── sglang-defaults.yaml
├── prompts/
│   ├── baseline.jsonl
│   └── multi-turn/
├── results/                          # Git-ignored output
└── tests/
```

---

## Dependencies

Core dependencies:

| Package | Purpose |
|---------|---------|
| `httpx` | Async HTTP client for inference requests |
| `pyyaml` | Configuration file parsing |
| `pydantic` | Configuration validation and schema |
| `rich` | CLI output formatting and progress display |
| `typer` | CLI framework |
| `runpod` | RunPod SDK for infrastructure provisioning |

Optional dependencies:

| Group | Packages | Purpose |
|-------|----------|---------|
| `prometheus` | `prometheus-client` | Extended metrics collection |
| `analysis` | `pandas`, `matplotlib` | Results analysis and visualization |

---

## GPU Pricing Reference

Hardcoded pricing sourced from RunPod documentation:

| GPU Type | Community ($/hr) | Secure ($/hr) |
|----------|------------------|---------------|
| H100-80GB | 3.99 | 5.49 |
| A100-80GB | 1.99 | 3.29 |
| A100-40GB | 1.64 | 2.49 |
| L40S | 1.14 | 1.84 |
| A40 | 0.79 | 1.24 |
| RTX 4090 | 0.69 | 0.99 |
| RTX 3090 | 0.44 | 0.69 |

Pricing data includes a `PRICING_LAST_UPDATED` timestamp. The CLI warns when data is older than 30 days.
