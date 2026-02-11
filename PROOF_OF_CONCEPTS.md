# Proof of Concepts

Standalone Python scripts that validate individual building blocks of the project. Each POC is minimal, self-contained, and has no dependency on any other POC or on project-level code. They exist purely to confirm that a specific capability works before building the real framework.

## Suggested Order

The POCs have natural dependencies. A reasonable progression through vLLM first:

**1 → 2 → 3 → 5 → 6 → 10**

Then branch to SGLang (4) and fill in the rest (7, 8).

---

## Infrastructure

### POC 1: RunPod API Basics ✅ Done

Programmatically create, start, stop, and terminate a RunPod GPU pod using their API.

**Validates:** API key setup, billing is active, pods can be managed without the web UI.

**What the script does:**
- Create a pod with a specified GPU type
- Poll until the pod is running
- Print the pod's status and connection details
- Stop the pod
- Terminate the pod

**Success criteria:** Pod goes through the full lifecycle (create → running → stopped → terminated) without manual intervention.

### POC 2: RunPod Connectivity ✅ Done

Establish an HTTP connection from the local machine to a running RunPod pod.

**Validates:** Networking, exposed ports, and that a simple HTTP server on the pod is reachable.

**What the script does:**
- Start a pod (or reuse a running one)
- Send an HTTP request to a known port on the pod
- Print the response

**Success criteria:** Local machine receives an HTTP response from the pod.

---

## Inference Server Deployment & Metrics

### POC 3: Deploy vLLM on RunPod ✅ Done

Start a vLLM server on a RunPod GPU with a small model, confirm it serves the OpenAI-compatible API, send a chat completion request, and scrape Prometheus metrics.

**Validates:** vLLM boots on RunPod, loads a model into GPU memory, exposes its API endpoint, produces a valid chat completion response, and exposes Prometheus metrics.

**What the script does:**
- Create a pod using a vLLM Docker image with a small model (e.g., a 7B quantized model)
- Poll until the vLLM health endpoint responds
- Print the available model(s) via the `/v1/models` endpoint
- Send a non-streaming `/v1/chat/completions` request and print the response content, token usage, and finish reason
- Scrape the `/metrics` endpoint and display a curated subset of Prometheus metrics (KV cache usage, request counts, latency histograms, prefix cache stats)

**Success criteria:** `/v1/models` returns the loaded model name, a chat completion request returns a valid response with token counts, and Prometheus metrics are scraped and displayed.

### POC 4: Deploy SGLang on RunPod ✅ Done

Same as POC 3 but with SGLang.

**Validates:** SGLang boots on RunPod, loads a model, exposes its API endpoint, produces a valid chat completion response, and exposes Prometheus metrics.

**What the script does:**
- Create a pod using an SGLang Docker image with the same small model (with `--enable-metrics`)
- Poll until the SGLang health endpoint responds
- Print the available model(s)
- Send a non-streaming `/v1/chat/completions` request and print the response content, token usage, and finish reason
- Scrape the `/metrics` endpoint and display a curated subset of Prometheus metrics (cache hit rate, request counts, latency histograms, throughput)

**Success criteria:** The SGLang server responds, lists the loaded model, a chat completion request returns a valid response with token counts, and Prometheus metrics are scraped and displayed.

---

## Streaming & Client-Side Metrics

### POC 5: Streaming Response Handling

Send a streaming request and consume the SSE stream token-by-token.

**Validates:** Streaming works over the network, and individual token chunks can be processed as they arrive (required for TTFT and inter-token measurements).

**What the script does:**
- Send a streaming `/v1/chat/completions` request
- Iterate over SSE chunks, printing each delta as it arrives
- Print total chunk count when the stream ends

**Success criteria:** Tokens arrive incrementally (not all at once), and the full response is assembled correctly.

### POC 6: Measure TTFT and End-to-End Latency

Capture time-to-first-token and total request duration for a single streaming request.

**Validates:** Timing instrumentation works correctly with streaming responses.

**What the script does:**
- Record timestamp before sending a streaming request
- Record timestamp when the first content chunk arrives (TTFT)
- Record timestamp when the stream ends (end-to-end latency)
- Print both measurements

**Success criteria:** TTFT is noticeably less than end-to-end latency. Values are in a plausible range (TTFT in tens/hundreds of ms, total in seconds).

### POC 7: Measure Inter-Token Latency

Record the arrival time of each token during a streaming response and compute the distribution of time-between-tokens.

**Validates:** Fine-grained per-token timing is feasible and produces meaningful data.

**What the script does:**
- Send a streaming request and record a timestamp for each content chunk
- Compute deltas between consecutive chunk timestamps
- Print min, max, mean, median, and p99 inter-token latency

**Success criteria:** Inter-token latencies are relatively consistent (low variance), with values typically in the single-digit to tens-of-ms range.

### POC 8: Measure Tokens Per Second

Count output tokens and divide by generation time to get throughput for a single request.

**Validates:** Token counting is accurate (usage fields vs. manual counting) and throughput numbers are plausible for the model/GPU.

**What the script does:**
- Send a request (streaming or non-streaming)
- Get the completion token count from the response's `usage` field
- Divide by generation duration (end-to-end minus TTFT for generation-only, or end-to-end for overall)
- Print tokens/second

**Success criteria:** Tokens/second is in a plausible range for the hardware (roughly 30-150+ tok/s depending on model and GPU).

---

## Server-Side GPU Metrics

### POC 9: Model-to-GPU Fit Check — Skipped

~~Attempt to load a model onto a specific GPU and observe whether it fits in memory.~~

Skipped — not needed for the current benchmarking workflow.

### POC 10: GPU Utilization Snapshot via nvidia-smi

Run `nvidia-smi` on the pod during inference and capture GPU state.

**Validates:** GPU-level data (utilization %, memory allocated, memory reserved) can be collected server-side, independent of the inference framework.

**What the script does:**
- Execute `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,memory.free --format=csv` on the pod (via RunPod API exec, SSH, or a simple HTTP wrapper)
- Parse and print the results
- Optionally run it once at idle and once during a request to show the difference

**Success criteria:** GPU utilization and memory numbers are returned and change between idle and active inference.
