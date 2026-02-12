# Implementation Plan: llm-inf-bench

## Context

The project has complete design docs (PROJECT_OVERVIEW.md, INTERFACE.md) and 4 working POC scripts that prove core capabilities (RunPod pod lifecycle, vLLM/SGLang deployment, chat completions, Prometheus scraping). The `src/llm_inf_bench/` directory is entirely empty scaffolding — no implementation exists beyond a stub CLI. The goal is to build the benchmarking framework incrementally, starting with the simplest workload (single requests) and one framework (vLLM), then expanding.

## Repo State Summary

**Implemented:** POCs 1-4, pyproject.toml, directory structure, design docs
**Empty scaffolding:** All `src/` modules, experiments/, prompts/, tests/
**Not started:** POCs 5-8 (streaming, latency measurement), all CLI commands

## Phases

Each phase delivers a usable capability. Phases are designed to be independent sub-plans.

---

### Phase 1: Config & Provider Foundation

Build the data layer and RunPod provider — everything needed before any requests can be sent.

**Modules:**
- `config/schema.py` — Pydantic models: `ExperimentConfig`, `ModelConfig`, `InfrastructureConfig`, `WorkloadConfig`, `FrameworkOptions`, `MetricsConfig`
- `config/loader.py` — YAML loading with `extends` inheritance, path resolution
- `config/validation.py` — Sanity checks (gpu_count > 0, valid framework names, etc.)
- `providers/base.py` — Abstract provider interface (create, wait, health check, terminate)
- `providers/runpod/client.py` — RunPod SDK wrapper; refactor from POC 3: `check_api_key()`, `get_proxy_url()`, `wait_for_pod_ready()`, pod create/terminate with safety-net cleanup
- `providers/runpod/pricing.py` — Hardcoded GPU pricing table with `PRICING_LAST_UPDATED` and staleness check
- `providers/runpod/templates.py` — Pod creation params per framework (image, ports, docker_args construction). Refactor `build_vllm_args()` and `build_sglang_cmd()` from POCs 3-4

**Reuse from POCs:**
- POC 3 `check_api_key()` (L62-69), `get_proxy_url()` (L101-103), `wait_for_pod_ready()` (L106-140), `build_vllm_args()` (L172-178), pod create logic (L181-208), cleanup safety net (L531-540)
- POC 4 `build_sglang_cmd()` (L173-185)

**Deliverable:** `llm-inf-bench init`, `llm-inf-bench doctor`, config loading with `--dry-run` validation

---

### Phase 2: vLLM Runner & Single-Request Workload

The minimum end-to-end path: load config, provision pod, send requests one at a time, collect results.

**Modules:**
- `runners/base.py` — Abstract runner: `start()`, `health_check()`, `chat_completion()` (streaming), `scrape_metrics()`
- `runners/vllm.py` — vLLM implementation; refactor health check polling from POC 3 (L142-169). Streaming `/v1/chat/completions` with per-token timing
- `workloads/base.py` — Abstract workload: `execute(runner) -> list[RequestMetrics]`
- `workloads/single.py` — Sequential request execution with per-request timing (TTFT, inter-token latencies, e2e latency, token counts)
- `metrics/collector.py` — `RequestMetrics` dataclass (ttft_ms, inter_token_latencies_ms, e2e_latency_ms, prompt_tokens, completion_tokens)
- `metrics/aggregator.py` — Percentile computation (p50, p95, p99), means, throughput (tok/s, req/s)
- `metrics/storage.py` — Write run results to JSON with full metadata (config, metrics, timestamps, run_id)
- `output/console.py` — Rich progress: phase indicators, progress bar, live throughput
- `output/summary.py` — Post-run summary table (matches INTERFACE.md format)

**Also:**
- `prompts/baseline.jsonl` — 20-30 varied prompts for testing
- `experiments/base/vllm-defaults.yaml` — vLLM framework defaults
- `experiments/examples/baseline-vllm.yaml` — Working single-request experiment

**CLI:** Wire up `run` command with cost confirmation, `--server-url`, `--confirm`, `--dry-run`

**Deliverable:** `llm-inf-bench run experiments/examples/baseline-vllm.yaml` works end-to-end, produces JSON results

**Note:** This phase should also complete POC 5 (streaming) and POC 6-8 (latency measurement) as part of building the runner — the runner IS the streaming client with timing instrumentation.

---

### Phase 3: SGLang Runner & Results Commands

Add the second framework and the ability to compare results.

**Modules:**
- `runners/sglang.py` — Same interface as vLLM runner; refactor from POC 4 (L143-170 health check, L173-185 cmd construction)
- CLI `results` subcommand group: `list`, `show <run-id>`, `compare <id1> <id2>`

**Also:**
- `experiments/base/sglang-defaults.yaml`
- `experiments/examples/baseline-sglang.yaml`

**Deliverable:** Run same workload on both frameworks, compare performance side-by-side

---

### Phase 4: Prometheus Metrics & Server-Side Observability

Add background GPU/cache metrics collection during experiments.

**Modules:**
- `metrics/gpu.py` — Prometheus scraper running in background thread at `sample_interval_ms`. Refactor `parse_prometheus_text()`, `get_metric_flexible()`, and curated metric lists from POCs 3-4 (L311-394 POC3, L318-400 POC4). Framework-aware (uses correct metric names for vLLM vs SGLang)

**Extend:**
- `metrics/aggregator.py` — KV cache utilization percentiles, prefix cache hit rate, peak memory
- `output/summary.py` — GPU metrics section in summary output
- `metrics/storage.py` — Include GPU time-series in result JSON

**Deliverable:** `collect_gpu_metrics: true` in config produces KV cache, prefix cache, and throughput data from server

---

### Phase 5: Batch & Concurrent Workloads

Scale up from single requests to controlled load patterns.

**Modules:**
- `workloads/batch.py` — Fixed batch sizes, all requests in batch sent concurrently via `asyncio.gather`
- `workloads/concurrent.py` — Worker pool with N concurrent slots, queue-based distribution
- `config/loader.py` — Extend with `sweep` field support (generate variations, execute sequentially)

**Also:**
- `experiments/examples/concurrency-sweep.yaml`

**Deliverable:** Throughput measurement under load, parameter sweeps

---

### Phase 6: Polish & Advanced Features (future)

Not in initial scope, but noted for completeness:
- Multi-turn workload (`workloads/multi_turn.py`)
- Resume capability (`--resume <run-id>`)
- Server management commands (`server status`, `server stop`)
- Results export (`results export <id> --format csv`)
- `update-pricing` command
- `--keep-server` flag

---

## Key Design Decisions

1. **Streaming by default for timing** — The runner always uses streaming internally (even for "single" workloads) because TTFT and inter-token latency require observing individual token arrivals. This is an implementation detail, not a user-facing config option.

2. **Async HTTP with httpx** — Use `httpx.AsyncClient` for the runner's chat completion calls. Necessary for concurrent workloads and harmless for single-request workloads.

3. **Runner owns health checking** — The runner (not the provider) waits for the inference server to be healthy, since health semantics differ between vLLM and SGLang.

4. **Provider owns pod lifecycle** — Create, wait-for-ready, terminate. Framework-agnostic.

5. **Refactor POC code, don't rewrite** — The POCs are tested and working. Extract functions into the appropriate modules with minimal modification.

## Verification

After each phase, the following should work:
- **Phase 1:** `llm-inf-bench doctor` passes, `llm-inf-bench run --dry-run experiments/examples/baseline-vllm.yaml` validates config
- **Phase 2:** `llm-inf-bench run experiments/examples/baseline-vllm.yaml --server-url <url>` completes, writes JSON results (use `--server-url` with a pre-existing pod to avoid provisioning during dev)
- **Phase 3:** Same experiment with `framework: sglang`, plus `llm-inf-bench results compare` works
- **Phase 4:** Results JSON includes GPU metrics when `collect_gpu_metrics: true`
- **Phase 5:** Concurrent workload with `concurrency: [1, 4, 8]` sweep produces 3 result sets
