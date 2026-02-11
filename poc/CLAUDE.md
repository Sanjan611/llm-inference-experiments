# POC Scripts — Notes for Claude

## RunPod `docker_args` behavior

The `docker_args` parameter in `runpod.create_pod()` is appended as arguments to the Docker image's ENTRYPOINT. It does **not** replace the entrypoint.

For images with a built-in entrypoint (like `vllm/vllm-openai` which has `ENTRYPOINT ["vllm", "serve"]`), pass only the arguments — not a full command:

```python
# Wrong — gets appended to entrypoint, producing: vllm serve python -m vllm...
docker_args="python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000"

# Right — produces: vllm serve Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000
docker_args="Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000"
```

For images without an entrypoint (like `runpod/pytorch:*`, `lmsysorg/sglang`), pass the full command since there's nothing to append to.

## Prometheus metrics: vLLM vs SGLang

### Enabling metrics

- **vLLM** exposes `/metrics` by default — no extra flags needed.
- **SGLang** requires `--enable-metrics` in the server launch command.

### Metric name separator instability

vLLM versions vary between `:` and `_` as the namespace separator (e.g. `vllm:prompt_tokens_total` vs `vllm_prompt_tokens_total`). Always try both separators when looking up a metric — see `get_metric_flexible()` in POC 3/4.

### Metric name mapping between frameworks

vLLM and SGLang expose similar concepts under different names:

| Concept | vLLM | SGLang |
|---|---|---|
| Active requests | `vllm:num_requests_running` | `sglang:num_running_reqs` |
| Queued requests | `vllm:num_requests_waiting` | `sglang:num_queue_reqs` |
| KV cache pressure | `vllm:kv_cache_usage_perc` (0-1 gauge) | `sglang:token_usage` (0-1 gauge) |
| Prefix cache | `prefix_cache_hits_total` + `prefix_cache_queries_total` | `sglang:cache_hit_rate` (pre-computed) |
| Generation throughput | Compute from `generation_tokens_total` counter deltas | `sglang:gen_throughput` (native gauge, tok/s) |
| Latency histograms | `vllm:time_to_first_token_seconds`, `vllm:inter_token_latency_seconds`, `vllm:e2e_request_latency_seconds` | Same names with `sglang:` prefix |

### Histogram values

TTFT, inter-token latency, and e2e latency are Prometheus histograms. To get averages, divide `_sum` by `_count`. For percentiles, use the `_bucket` entries with `le` labels.

### Non-fatal collection

Metrics scraping should never block pod cleanup or abort an experiment. Wrap in try/except, warn, and continue.
