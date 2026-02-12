# Plan: Experiment Dashboard with Live Metrics Visualization

## Requirements

1. **Live metrics visualization** — As an experiment runs and the Prometheus scraper polls `/metrics`, display GPU metrics (KV cache usage, active/queued requests, prefix cache hit rate, generation throughput) as updating time-series charts in a browser.

2. **Live request metrics** — As each inference request completes, update client-side metric charts (TTFT distribution, E2E latency distribution, throughput over time) in real time.

3. **Experiment selection and control** — User can browse available experiment YAML configs, configure options (e.g. `--server-url`), and start a run from the UI.

4. **Log streaming** — Display experiment logs (provisioning status, health check progress, phase transitions, errors) in a live log panel.

5. **Post-experiment persistence** — After experiment completes and the RunPod pod terminates, all metrics remain visible in the dashboard for analysis. Charts stay interactive (zoom, hover, pan).

6. **Multi-experiment support** — Dashboard persists data across sequential runs. Users can view and compare past experiments. Historical results (from JSON files in `results/`) are browsable.

7. **Localhost only** — No hosted deployment. Runs on `localhost`, started via a CLI command.

---

## Current State Analysis

### What exists today

- **`GpuMetricsScraper`** (`metrics/gpu.py`) — Background asyncio task polls `/metrics` at configurable intervals (default 100ms), parses Prometheus text format, stores `GpuSample` time-series in memory. Framework-aware metric name mapping for vLLM vs SGLang.

- **Request timing** (`runners/openai_compat.py`) — Streaming `/v1/chat/completions` with per-token timestamps. Produces `RequestResult` with TTFT, inter-token latencies, e2e latency, token counts.

- **Progress callbacks** — Workloads call `on_request_complete(result)` after each request. Currently wired to `BenchmarkProgress.on_request_complete` which updates a Rich progress bar.

- **JSON persistence** (`metrics/storage.py`) — Full results saved to `results/{run_id}.json` including experiment config, all request results, aggregated stats, and GPU time-series.

- **Console output** (`output/console.py`, `output/summary.py`) — Rich progress bar during execution (shows `tok=X avg_ttft=Yms`). Post-run summary panel with percentile tables and GPU stats.

### The gap

All metrics data flows through in-memory structures, gets aggregated at the end, and is dumped to JSON + Rich console. There is no mechanism to:
- Push live data to an external consumer (browser)
- Control experiments from outside the CLI
- Visualize time-series data graphically

---

## Architecture Decision: FastAPI + WebSocket + Single HTML File

### Chosen approach

A FastAPI server running in the same Python process as the experiment runner, communicating with the browser via WebSocket for real-time bidirectional messaging. The frontend is a single self-contained HTML file (no build step) that loads Plotly.js and Alpine.js from CDN.

```
llm-inf-bench dashboard [--port 8420]
          │
          ▼
    FastAPI (uvicorn)
    ├── GET /              → serves index.html
    ├── GET /api/configs   → lists experiment YAML files
    ├── GET /api/results   → lists past result JSON files
    ├── GET /api/results/{run_id} → loads a specific result
    ├── WS  /ws            → bidirectional real-time channel
    │   ├── Server→Client: gpu_sample, request_complete, phase, log, summary
    │   └── Client→Server: start_experiment, stop_experiment
    └── Experiment Manager
        ├── Wraps existing _execute_run orchestration
        ├── Hooks into GpuMetricsScraper to emit samples
        ├── Hooks into workload callbacks to emit request results
        └── Captures log output for forwarding
```

### Alternatives considered

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **Grafana + Prometheus** | Industry standard, rich dashboards, built-in Prometheus | Heavy infra (Docker for Prometheus + Grafana), no experiment lifecycle awareness, overkill for a dev tool | Rejected — too heavy, requires Docker, doesn't know about our experiment concept |
| **Streamlit** | Zero frontend code, Python-native | Re-run model conflicts with real-time streaming, `st.empty` loops are hacky, session isolation makes it hard to share state with runner | Rejected — wrong execution model for live streaming |
| **Panel/Bokeh** | Python-native, supports push updates | Heavy dependency, Python→JS compilation model, less mature ecosystem | Possible but over-engineered for our needs |
| **SSE (Server-Sent Events)** | Simpler than WebSocket, native browser support | Unidirectional (server→client only), would need separate REST endpoints for commands, two connections instead of one | Possible but WebSocket is cleaner for bidirectional needs |
| **FastAPI + WebSocket + React SPA** | Most flexible, best DX for large frontends | Requires Node.js build toolchain, JS dependency management, significant complexity for a single-page dashboard | Rejected — build step adds friction for a Python-focused project |

### Why this stack

1. **FastAPI** — Already uses async Python. Adds `fastapi` + `uvicorn` to deps (both lightweight, well-maintained). Natural fit with existing `httpx` async architecture.

2. **WebSocket** — True bidirectional: push metrics to browser AND receive commands (start/stop experiments). Single connection, no polling. FastAPI has first-class WebSocket support.

3. **Single HTML file** — No build step, no `node_modules`, no JS toolchain. The file is served directly by FastAPI. Libraries loaded from CDN. This is a dashboard for a Python tool, not a production web app.

4. **Plotly.js** — Best charting library for scientific/analytical dashboards. Built-in `Plotly.extendTraces()` for streaming updates. Excellent interactivity (zoom, hover, pan, export). ~3MB from CDN is fine for localhost.

5. **Alpine.js** — 15KB reactive framework with no build step. Declarative HTML attributes (`x-data`, `x-show`, `x-for`). Handles UI state (selected experiment, connection status, tab switching) without verbose vanilla JS.

---

## WebSocket Protocol

### Server → Client

```jsonc
// Phase transitions
{"type": "phase", "run_id": "exp-20260212-150000", "phase": "provisioning", "data": {"gpu_type": "A100", "gpu_count": 1}}
{"type": "phase", "run_id": "...", "phase": "provisioning_done", "data": {"pod_id": "abc123", "server_url": "https://..."}}
{"type": "phase", "run_id": "...", "phase": "model_loading", "data": {"model": "meta-llama/..."}}
{"type": "phase", "run_id": "...", "phase": "model_loading_done", "data": {"elapsed_s": 45.2}}
{"type": "phase", "run_id": "...", "phase": "execution", "data": {"total_requests": 50, "workload_type": "single"}}
{"type": "phase", "run_id": "...", "phase": "sweep_iteration", "data": {"iteration": 2, "total": 3, "params": {"concurrency": 4}}}
{"type": "phase", "run_id": "...", "phase": "cleanup", "data": {}}
{"type": "phase", "run_id": "...", "phase": "done", "data": {"result_path": "results/exp-20260212-150000.json"}}

// Live GPU metrics (forwarded from GpuMetricsScraper at ~1s intervals, downsampled from 100ms)
{"type": "gpu_sample", "run_id": "...", "data": {
  "timestamp": 12.5,
  "kv_cache_usage": 0.45,
  "active_requests": 3,
  "queued_requests": 0,
  "generation_throughput": 42.5,
  "prefix_cache_hit_rate": null
}}

// Per-request results (forwarded from workload on_request_complete callback)
{"type": "request_complete", "run_id": "...", "data": {
  "request_index": 7,
  "ttft_ms": 120.5,
  "e2e_latency_ms": 1450.3,
  "tbt_ms": 28.4,
  "prompt_tokens": 150,
  "completion_tokens": 200,
  "error": null
}}

// Log messages (captured from Python logging)
{"type": "log", "level": "info", "message": "Pod created: abc123", "timestamp": "2026-02-12T15:00:05Z"}

// Final summary (after experiment completes)
{"type": "summary", "run_id": "...", "data": {/* AggregatedMetrics serialized */}}

// Errors
{"type": "error", "message": "Health check timed out after 300s"}
```

### Client → Server

```jsonc
// Start an experiment
{"type": "start_experiment", "config_path": "experiments/examples/baseline-vllm.yaml", "options": {
  "server_url": null,  // or "https://..." to skip provisioning
  "confirm": true      // skip cost confirmation
}}

// Stop a running experiment (triggers graceful cleanup)
{"type": "stop_experiment", "run_id": "..."}

// Request data
{"type": "list_configs"}
{"type": "list_results"}
{"type": "load_result", "run_id": "..."}
```

---

## Dashboard Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  llm-inf-bench Dashboard                         ● Connected  :8420 │
├────────────────────┬─────────────────────────────────────────────────┤
│                    │                                                 │
│  New Experiment    │  [Live ✦] [History] [Compare]     ← Tab bar    │
│  ┌──────────────┐  │                                                 │
│  │ Config:      │  │  Experiment: baseline-vllm                     │
│  │ [dropdown ▾] │  │  Status: ● Running — Phase 3/4 (Execution)    │
│  │              │  │  vLLM | Llama-3.1-8B | A100 x1                │
│  │ Server URL:  │  │                                                 │
│  │ [optional  ] │  │  ┌─────────────────┐  ┌─────────────────┐      │
│  │              │  │  │  KV Cache %     │  │  Active Reqs    │      │
│  │ [▶ Start]    │  │  │  (line chart)   │  │  (line chart)   │      │
│  └──────────────┘  │  │   ╱╲            │  │      ╱╲         │      │
│                    │  │  ╱  ╲╱╲╱        │  │    ╱╱  ╲        │      │
│  Past Runs         │  └─────────────────┘  └─────────────────┘      │
│  ┌──────────────┐  │                                                 │
│  │ ● baseline   │  │  ┌─────────────────┐  ┌─────────────────┐      │
│  │   -vllm      │  │  │  TTFT (ms)      │  │  Throughput     │      │
│  │   15:00:00   │  │  │  (histogram)    │  │  (tok/s line)   │      │
│  │              │  │  │  ▇▇▇▅▃▂        │  │       ╱───      │      │
│  │ ○ baseline   │  │  └─────────────────┘  └─────────────────┘      │
│  │   -sglang    │  │                                                 │
│  │   14:30:00   │  │  ┌─────────────────┐  ┌─────────────────┐      │
│  │              │  │  │  E2E Latency    │  │  TBT (ms)       │      │
│  │ ○ sweep      │  │  │  (histogram)    │  │  (histogram)    │      │
│  │   14:00:00   │  │  └─────────────────┘  └─────────────────┘      │
│  └──────────────┘  │                                                 │
│                    │  ┌───────────────────────────────────────┐      │
│                    │  │  Logs                                 │      │
│                    │  │  15:00:05 [INFO] Pod created: abc123  │      │
│                    │  │  15:00:50 [INFO] Health check passed  │      │
│                    │  │  15:00:51 [INFO] Request 1/50 done    │      │
│                    │  │  15:00:52 [INFO] Request 2/50 done    │      │
│                    │  └───────────────────────────────────────┘      │
│                    │                                                 │
│                    │  Summary (appears after completion):             │
│                    │  ┌───────────────────────────────────────┐      │
│                    │  │ Requests: 50/50  Errors: 0            │      │
│                    │  │ Duration: 125.3s                      │      │
│                    │  │ Throughput: 85.2 tok/s (0.40 req/s)   │      │
│                    │  │ TTFT: p50=120ms p95=180ms p99=220ms   │      │
│                    │  │ E2E:  p50=1.4s  p95=2.1s  p99=2.8s   │      │
│                    │  │ KV Cache: peak=67% mean=42%           │      │
│                    │  └───────────────────────────────────────┘      │
└────────────────────┴─────────────────────────────────────────────────┘
```

**Tabs:**
- **Live** — Shows the currently running (or most recently completed) experiment with real-time charts
- **History** — Browse all past results from `results/` directory, click to view charts reconstructed from stored JSON data
- **Compare** — Select two runs for side-by-side comparison (mirrors existing `results compare` CLI)

---

## Implementation Phases

### Phase A: Dashboard Backend & Event System

**New modules:**

- **`src/llm_inf_bench/dashboard/__init__.py`**

- **`src/llm_inf_bench/dashboard/server.py`** — FastAPI app setup, route registration, static file serving, uvicorn launch.

- **`src/llm_inf_bench/dashboard/websocket.py`** — WebSocket connection manager. Handles multiple connected clients (multiple browser tabs). Broadcasts events to all connected clients. Receives and dispatches commands.

- **`src/llm_inf_bench/dashboard/events.py`** — Pydantic models for all WebSocket message types (both directions). Serialization helpers. Event types: `PhaseEvent`, `GpuSampleEvent`, `RequestCompleteEvent`, `LogEvent`, `SummaryEvent`, `ErrorEvent`, `StartExperimentCommand`, `StopExperimentCommand`.

- **`src/llm_inf_bench/dashboard/experiment_manager.py`** — Manages experiment lifecycle within the dashboard context. Wraps the existing `_execute_iteration` logic but emits events instead of updating Rich progress. Handles:
  - Starting an experiment from a config path
  - Forwarding GPU samples from the scraper to WebSocket
  - Forwarding request completions to WebSocket
  - Forwarding phase transitions to WebSocket
  - Graceful stop/cancel
  - Tracking active and completed runs

- **`src/llm_inf_bench/dashboard/log_handler.py`** — Custom `logging.Handler` subclass that captures log records from `llm_inf_bench.*` loggers and forwards them to the WebSocket as `LogEvent` messages.

**Modifications to existing modules:**

- **`metrics/gpu.py`** — Add an optional `on_sample` callback to `GpuMetricsScraper` so that each scraped sample can be forwarded in real-time (in addition to being stored in the time-series). The callback is called with the `GpuSample` after each successful scrape. No change to default behavior when callback is `None`.

- **`workloads/base.py`** — The existing `on_request_complete` callback pattern already supports this. No changes needed — the dashboard's experiment manager simply provides its own callback.

- **`cli.py`** — Add `dashboard` command: `llm-inf-bench dashboard [--port 8420] [--no-browser]`. Starts the FastAPI server and optionally opens the browser.

**New dependencies:**

```toml
[project.optional-dependencies]
dashboard = ["fastapi>=0.115", "uvicorn[standard]>=0.32"]
```

Using an optional dependency group keeps the core CLI lightweight. Users who want the dashboard install with `pip install llm-inf-bench[dashboard]` or `uv sync --extra dashboard`.

### Phase B: Frontend — Live Experiment View

**`src/llm_inf_bench/dashboard/static/index.html`** — Single self-contained HTML file.

**External libraries (CDN-loaded):**
- Plotly.js (~3MB) — Time-series charts, histograms, interactive analysis
- Alpine.js (~15KB) — Reactive UI state management

**Charts (6 panels, 2x3 grid):**

1. **KV Cache Usage** — `Plotly.extendTraces()` line chart, 0-100% y-axis, time on x-axis. Updates with each `gpu_sample` event.

2. **Active & Queued Requests** — Two-line chart (active in blue, queued in orange). From `gpu_sample` events.

3. **TTFT Distribution** — Histogram that rebuilds as requests complete. Each `request_complete` adds a data point.

4. **E2E Latency Distribution** — Same pattern as TTFT.

5. **Throughput Over Time** — Line chart showing tokens/second. Computed client-side as a rolling window over `request_complete` events (tokens / time_window).

6. **Request Timeline** — Scatter/bar chart showing each request as a horizontal segment from start to finish, colored by success/failure. Gives a visual picture of concurrency and stalls.

**Other UI elements:**
- Connection status indicator (WebSocket open/closed/reconnecting)
- Experiment status banner (phase, progress count)
- Log panel (scrollable, auto-scroll with pause-on-hover)
- Summary panel (appears after experiment completes, mirrors `print_summary` output)

### Phase C: Frontend — History & Compare Tabs

**History tab:**
- Fetches `GET /api/results` on tab switch
- Lists all past runs in a table (run ID, framework, model, date, status)
- Click a run to load its full data (`GET /api/results/{run_id}`) and render the same 6 charts using the stored time-series and request data
- GPU time-series charts are reconstructed from `gpu_metrics.time_series` in the JSON
- Request metric charts are reconstructed from the `requests` array

**Compare tab:**
- Two dropdowns to select runs A and B
- Side-by-side summary tables (mirrors existing `print_comparison`)
- Overlay charts: TTFT distributions, throughput over time for both runs on the same axes with different colors

### Phase D: Robustness & Polish

- **Auto-reconnect** — WebSocket reconnection with exponential backoff on disconnect
- **GPU sample downsampling** — The scraper runs at 100ms intervals but the WebSocket pushes at ~1s intervals (every 10th sample) to avoid flooding the browser. Full resolution data still saved to JSON.
- **Experiment queueing** — If user starts a second experiment while one is running, queue it to run after the current one completes (sequential execution only in v1)
- **Error handling** — Graceful handling of failed experiments, partial results display, cleanup confirmation
- **Responsive layout** — CSS grid that adapts to window size
- **Dark mode** — Default to dark theme (common for dashboards, easier on eyes during long experiments)

---

## File Structure

```
src/llm_inf_bench/dashboard/
├── __init__.py
├── server.py              # FastAPI app, routes, uvicorn startup
├── websocket.py           # ConnectionManager, broadcast, dispatch
├── events.py              # Pydantic event/command models
├── experiment_manager.py  # Wraps experiment orchestration with event emission
├── log_handler.py         # logging.Handler → WebSocket bridge
└── static/
    └── index.html         # Single-file SPA (HTML + CSS + JS)
```

---

## Key Design Decisions

1. **Single-process architecture** — The dashboard server runs in the same process as the experiment runner. This avoids IPC complexity and gives direct access to in-memory metrics data. The FastAPI event loop runs the experiment as an asyncio task.

2. **Optional dependency** — `fastapi` and `uvicorn` are in `[dashboard]` extras, not core dependencies. The CLI `dashboard` command checks for availability and prints an install hint if missing.

3. **No build step** — The frontend is a single HTML file served by FastAPI's `StaticFiles`. Libraries loaded from CDN. This keeps the project Python-focused and avoids `node_modules`, `package.json`, webpack, etc.

4. **WebSocket over SSE** — We need bidirectional communication (push metrics AND receive commands like "start experiment"). SSE is server→client only and would require a separate REST channel for commands.

5. **Plotly over Chart.js** — Plotly has better scientific visualization (histogram binning, axis formatting, hover info), built-in streaming support (`extendTraces`), and better zoom/pan for post-hoc analysis. The ~3MB size is irrelevant for localhost.

6. **Client-side rolling throughput** — Throughput (tok/s) is computed in the browser from `request_complete` events using a sliding window, rather than having the server compute it. This keeps the server stateless about display concerns and allows the UI to adjust the window size.

7. **GPU sample downsampling for WebSocket** — Scraper continues at 100ms for accurate data collection and JSON persistence, but WebSocket broadcasts only ~1/s to avoid overloading the browser's charting. The `on_sample` callback includes a simple counter-based throttle.

8. **Sequential experiments only** — v1 does not support concurrent experiments. Running multiple pods simultaneously has cost and resource management implications that need careful UX design. Sequential queuing is supported.

---

## Scope Boundaries

**In scope:**
- `llm-inf-bench dashboard` command
- Live 6-chart panel for running experiments
- Log streaming panel
- Experiment config browser and launcher
- History browser (loads from existing JSON result files)
- Side-by-side comparison view
- Post-experiment interactive charts (persist after pod termination)

**Out of scope (future):**
- Concurrent experiment execution
- Custom chart configuration (which metrics to display)
- Dashboard authentication
- Remote access (always localhost)
- Persistent backend storage (SQLite, etc.) — JSON files remain the source of truth
- Notification system (email/Slack on experiment complete)
