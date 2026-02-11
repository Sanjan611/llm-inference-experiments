# Proof of Concepts

Standalone Python scripts that validate individual building blocks of the project. Each POC is minimal, self-contained, and has no dependency on any other POC or on project-level code. They exist purely to confirm that a specific capability works before building the real framework.

## Suggested Order

The POCs have natural dependencies. A reasonable progression through vLLM first:

**1 → 2 → 3 → 6 → 8 → 9 → 12 → 14**

Then branch to SGLang (4, 7, 13) and fill in the rest (5, 10, 11).

---

## Infrastructure

### POC 1: RunPod API Basics

Programmatically create, start, stop, and terminate a RunPod GPU pod using their API.

**Validates:** API key setup, billing is active, pods can be managed without the web UI.

**What the script does:**
- Create a pod with a specified GPU type
- Poll until the pod is running
- Print the pod's status and connection details
- Stop the pod
- Terminate the pod

**Success criteria:** Pod goes through the full lifecycle (create → running → stopped → terminated) without manual intervention.

### POC 2: RunPod Connectivity

Establish an HTTP connection from the local machine to a running RunPod pod.

**Validates:** Networking, exposed ports, and that a simple HTTP server on the pod is reachable.

**What the script does:**
- Start a pod (or reuse a running one)
- Send an HTTP request to a known port on the pod
- Print the response

**Success criteria:** Local machine receives an HTTP response from the pod.

---

## Inference Server Deployment

### POC 3: Deploy vLLM on RunPod

Start a vLLM server on a RunPod GPU with a small model and confirm it serves the OpenAI-compatible API.

**Validates:** vLLM boots on RunPod, loads a model into GPU memory, and exposes its API endpoint.

**What the script does:**
- Create a pod using a vLLM Docker image with a small model (e.g., a 7B quantized model)
- Poll until the vLLM health endpoint responds
- Print the available model(s) via the `/v1/models` endpoint

**Success criteria:** `/v1/models` returns the loaded model name.

### POC 4: Deploy SGLang on RunPod

Same as POC 3 but with SGLang.

**Validates:** SGLang boots on RunPod, loads a model, and exposes its API endpoint.

**What the script does:**
- Create a pod using an SGLang Docker image with the same small model
- Poll until the SGLang health endpoint responds
- Print the available model(s)

**Success criteria:** The SGLang server responds and lists the loaded model.

### POC 5: Model-to-GPU Fit Check

Attempt to load a model onto a specific GPU and observe whether it fits in memory.

**Validates:** Baseline understanding of which model/GPU pairings work, and what failure looks like when a model doesn't fit.

**What the script does:**
- Start a pod with a chosen GPU type
- Launch vLLM (or SGLang) with a model that is expected to fit
- Check that the server becomes healthy
- Optionally repeat with a model that is too large and capture the error

**Success criteria:** Healthy server for the fitting model; clear error message for the oversized model.

---

## Single Request Round-Trip

### POC 6: Single Completion Request to vLLM

Send one chat completion request from the local machine to a running vLLM server.

**Validates:** Full network round-trip, OpenAI API compatibility, and that the model generates a coherent response.

**What the script does:**
- Send a non-streaming `/v1/chat/completions` request with a short prompt
- Print the response content, token usage, and finish reason

**Success criteria:** A complete response is returned with valid token counts.

### POC 7: Single Completion Request to SGLang

Same as POC 6 but targeting an SGLang server.

**Validates:** SGLang's API compatibility with the same request format used for vLLM.

**What the script does:**
- Send a non-streaming `/v1/chat/completions` request to SGLang
- Print the response content, token usage, and finish reason

**Success criteria:** Response format matches vLLM's (or differences are documented).

### POC 8: Streaming Response Handling

Send a streaming request and consume the SSE stream token-by-token.

**Validates:** Streaming works over the network, and individual token chunks can be processed as they arrive (required for TTFT and inter-token measurements).

**What the script does:**
- Send a streaming `/v1/chat/completions` request
- Iterate over SSE chunks, printing each delta as it arrives
- Print total chunk count when the stream ends

**Success criteria:** Tokens arrive incrementally (not all at once), and the full response is assembled correctly.

---

## Client-Side Metrics

### POC 9: Measure TTFT and End-to-End Latency

Capture time-to-first-token and total request duration for a single streaming request.

**Validates:** Timing instrumentation works correctly with streaming responses.

**What the script does:**
- Record timestamp before sending a streaming request
- Record timestamp when the first content chunk arrives (TTFT)
- Record timestamp when the stream ends (end-to-end latency)
- Print both measurements

**Success criteria:** TTFT is noticeably less than end-to-end latency. Values are in a plausible range (TTFT in tens/hundreds of ms, total in seconds).

### POC 10: Measure Inter-Token Latency

Record the arrival time of each token during a streaming response and compute the distribution of time-between-tokens.

**Validates:** Fine-grained per-token timing is feasible and produces meaningful data.

**What the script does:**
- Send a streaming request and record a timestamp for each content chunk
- Compute deltas between consecutive chunk timestamps
- Print min, max, mean, median, and p99 inter-token latency

**Success criteria:** Inter-token latencies are relatively consistent (low variance), with values typically in the single-digit to tens-of-ms range.

### POC 11: Measure Tokens Per Second

Count output tokens and divide by generation time to get throughput for a single request.

**Validates:** Token counting is accurate (usage fields vs. manual counting) and throughput numbers are plausible for the model/GPU.

**What the script does:**
- Send a request (streaming or non-streaming)
- Get the completion token count from the response's `usage` field
- Divide by generation duration (end-to-end minus TTFT for generation-only, or end-to-end for overall)
- Print tokens/second

**Success criteria:** Tokens/second is in a plausible range for the hardware (roughly 30-150+ tok/s depending on model and GPU).

---

## Server-Side / GPU Metrics

### POC 12: Read Prometheus Metrics from vLLM

Scrape the vLLM Prometheus endpoint and parse the available metrics.

**Validates:** The metrics endpoint is accessible from the local machine, and the available metrics are understood.

**What the script does:**
- Send a GET request to vLLM's `/metrics` endpoint
- Parse the Prometheus text format
- Print all metric names and their current values
- Highlight KV cache and batch-related metrics if present

**Success criteria:** Metrics are returned and parseable. KV cache utilization metrics are present.

### POC 13: Read Prometheus Metrics from SGLang

Same as POC 12 but for SGLang.

**Validates:** SGLang's metrics endpoint works and the available metrics are documented.

**What the script does:**
- Scrape SGLang's metrics endpoint
- Parse and print all metric names and values
- Note any differences from vLLM's metric set

**Success criteria:** Metrics are returned. Differences from vLLM's metrics are identified.

### POC 14: GPU Utilization Snapshot via nvidia-smi

Run `nvidia-smi` on the pod during inference and capture GPU state.

**Validates:** GPU-level data (utilization %, memory allocated, memory reserved) can be collected server-side, independent of the inference framework.

**What the script does:**
- Execute `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,memory.free --format=csv` on the pod (via RunPod API exec, SSH, or a simple HTTP wrapper)
- Parse and print the results
- Optionally run it once at idle and once during a request to show the difference

**Success criteria:** GPU utilization and memory numbers are returned and change between idle and active inference.
