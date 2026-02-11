# LLM Inference Experiments

A benchmarking and observability framework for empirically measuring LLM inference performance using vLLM and SGLang on cloud-hosted GPUs.

## Overview

This project provides a structured approach to running inference experiments on open-source LLMs hosted on RunPod. It enables systematic comparison of inference frameworks (vLLM and SGLang) across different models, GPU configurations, and workload patterns.

The primary focus is on **empirical measurement and observability**—capturing detailed metrics about latency, throughput, and GPU behavior to understand how different configurations perform in practice.

## Motivation

Serving LLMs efficiently requires understanding the interplay between:

- **Model size and quantization** — How model parameters affect memory usage and speed
- **Hardware selection** — How different GPUs handle various workloads
- **Inference framework behavior** — How vLLM and SGLang differ in their optimizations
- **Request patterns** — How batching, concurrency, and conversation structure impact performance

While documentation and benchmarks exist for these frameworks, hands-on experimentation reveals nuances that papers and blog posts don't capture. This project creates a repeatable environment for running controlled experiments and recording results.

Specific areas of exploration include:

- KV cache utilization and memory pressure under load
- Prefix caching effectiveness for shared context scenarios
- Performance differences between vLLM and SGLang for identical workloads
- Behavior under varying concurrency levels
- Multi-turn and agentic workload characteristics

## Features

### Inference Frameworks

Both frameworks can be tested independently against the same workloads:

- **vLLM** — PagedAttention, continuous batching, prefix caching
- **SGLang** — RadixAttention, constrained decoding, efficient KV cache reuse

### Workload Types

Experiments support multiple request patterns, designed to be extensible:

- **Single requests** — Baseline latency measurements
- **Batched requests** — Throughput under controlled batch sizes
- **Concurrent load** — Simulated multi-user scenarios
- **Multi-turn / Agentic** — Conversation flows with context accumulation

### Metrics Collection

Comprehensive metrics captured at multiple levels:

**Latency**
- Time to first token (TTFT)
- Time between tokens (TBT) / Inter-token latency
- End-to-end request latency

**Throughput**
- Tokens per second (generation speed)
- Requests per second (serving capacity)

**GPU**
- Utilization percentage
- Memory usage (allocated, reserved, KV cache)
- KV cache occupancy and eviction rates

### Experiment Configuration

Experiments are defined in configuration files, enabling:

- Reproducible runs with documented parameters
- Model-to-GPU mappings for appropriate instance selection
- Single-command execution of defined experiments
- Structured output for analysis and comparison

### Metrics Storage

A layered approach to metrics:

- **Primary** — Structured JSON/JSONL files per experiment run, containing all metrics and configuration
- **Optional** — Prometheus integration for real-time monitoring (both vLLM and SGLang expose compatible endpoints)
- **Analysis** — Results designed for easy loading into pandas, notebooks, or visualization tools

## Architecture

```
┌─────────────────────┐         ┌─────────────────────────────────┐
│   Local Machine     │         │           RunPod                │
│                     │         │                                 │
│  ┌───────────────┐  │  HTTP   │  ┌───────────────────────────┐  │
│  │  Benchmark    │──┼────────────▶  vLLM / SGLang Server    │  │
│  │  Client       │  │         │  │                           │  │
│  └───────────────┘  │         │  │  ┌─────────┐ ┌─────────┐  │  │
│         │           │         │  │  │  Model  │ │   GPU   │  │  │
│         ▼           │         │  │  └─────────┘ └─────────┘  │  │
│  ┌───────────────┐  │         │  └───────────────────────────┘  │
│  │  Experiment   │  │         │               │                 │
│  │  Results      │  │         │               ▼                 │
│  │  (JSON)       │  │         │  ┌───────────────────────────┐  │
│  └───────────────┘  │         │  │  Prometheus Metrics       │  │
│                     │         │  │  (optional scraping)      │  │
└─────────────────────┘         │  └───────────────────────────┘  │
                                └─────────────────────────────────┘
```

**Local machine** runs the benchmark client, sends requests to the inference server, and collects results. This simulates real-world client-server separation where network latency is part of the end-to-end experience.

**RunPod** hosts the inference server (vLLM or SGLang) with the selected model and GPU. GPU-level metrics are collected server-side, independent of network latency.

This separation allows:
- Measuring true client-perceived latency (including network)
- Isolating GPU-level metrics from network effects
- Testing against different GPU types without local hardware

## Key Concepts

Several inference optimization concepts are relevant to experiments in this project:

### KV Cache

During autoregressive generation, the key-value pairs from attention layers are cached to avoid recomputation. Observing KV cache behavior—memory usage, occupancy rates, eviction patterns—reveals how the server handles memory pressure under load.

### Prefix Caching

When multiple requests share a common prefix (e.g., a system prompt), the KV cache for that prefix can be reused. Both vLLM and SGLang implement this differently. Experiments with shared-prefix workloads can measure the effectiveness of this optimization.

### Continuous Batching

Unlike static batching, continuous batching allows new requests to join a batch as earlier requests complete. This affects both latency and throughput characteristics, particularly under variable load.

### PagedAttention (vLLM) vs RadixAttention (SGLang)

Different approaches to managing KV cache memory. PagedAttention uses paged memory allocation similar to OS virtual memory. RadixAttention uses a radix tree for efficient prefix sharing. Comparative experiments can reveal practical differences in behavior.
