"""FastAPI application with REST + WebSocket routes for the dashboard."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from llm_inf_bench.config.loader import load_experiment
from llm_inf_bench.config.validation import ConfigValidationError
from llm_inf_bench.dashboard.experiment_manager import ExperimentManager
from llm_inf_bench.dashboard.log_handler import WebSocketLogHandler
from llm_inf_bench.dashboard.websocket import ConnectionManager
from llm_inf_bench.metrics.storage import list_results, load_result

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(title="llm-inf-bench Dashboard")

    cm = ConnectionManager()
    em = ExperimentManager(cm)

    # Install WebSocket log handler on the root package logger
    log_handler = WebSocketLogHandler(cm)
    log_handler.setFormatter(logging.Formatter("%(name)s — %(message)s"))
    pkg_logger = logging.getLogger("llm_inf_bench")
    pkg_logger.addHandler(log_handler)

    @app.on_event("shutdown")
    async def _remove_log_handler() -> None:
        pkg_logger.removeHandler(log_handler)

    # ------------------------------------------------------------------
    # Static file serving
    # ------------------------------------------------------------------

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    # ------------------------------------------------------------------
    # REST API
    # ------------------------------------------------------------------

    @app.get("/api/configs")
    async def list_configs() -> list[dict[str, str]]:
        """Scan experiments/ for YAML config files."""
        configs: list[dict[str, str]] = []
        experiments_dir = Path("experiments")
        if experiments_dir.is_dir():
            for yaml_path in sorted(experiments_dir.rglob("*.yaml")):
                configs.append(
                    {
                        "name": yaml_path.stem,
                        "path": str(yaml_path),
                        "relative": str(yaml_path.relative_to(experiments_dir)),
                    }
                )
        return configs

    @app.get("/api/configs/preview")
    async def config_preview(path: str = "") -> JSONResponse:
        """Parse a config file and return key parameters for preview."""
        if not path:
            return JSONResponse({"error": "path query parameter required"}, status_code=400)
        try:
            config = load_experiment(Path(path))
        except (ConfigValidationError, ValidationError, FileNotFoundError, ValueError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)

        preview: dict[str, Any] = {
            "name": config.name,
            "description": config.description,
            "framework": config.framework,
            "model_name": config.model.name,
            "quantization": config.model.quantization,
            "gpu_type": config.infrastructure.gpu_type,
            "gpu_count": config.infrastructure.gpu_count,
            "workload_type": config.workload.type,
            "request_count": config.workload.requests.count,
            "max_tokens": config.workload.parameters.max_tokens,
            "batch_size": config.workload.batch_size,
            "concurrency": config.workload.concurrency,
            "sweep": config.workload.sweep.model_dump() if config.workload.sweep else None,
        }
        return JSONResponse(preview)

    @app.get("/api/results")
    async def api_list_results(
        output_dir: str = "results/",
    ) -> list[dict[str, Any]]:
        """List past result files."""
        stored = list_results(output_dir)
        return [
            {
                "run_id": r.run_id,
                "status": r.status,
                "experiment": r.experiment,
                "metadata": r.metadata,
                "summary": r.summary,
            }
            for r in stored
        ]

    @app.get("/api/results/{run_id}")
    async def api_get_result(
        run_id: str,
        output_dir: str = "results/",
    ) -> JSONResponse:
        """Load a specific result with full data."""
        try:
            result = load_result(output_dir, run_id)
        except (FileNotFoundError, ValueError) as e:
            return JSONResponse({"error": str(e)}, status_code=404)
        return JSONResponse(
            {
                "run_id": result.run_id,
                "status": result.status,
                "experiment": result.experiment,
                "metadata": result.metadata,
                "summary": result.summary,
                "requests": result.requests,
                "gpu_metrics": result.gpu_metrics,
            }
        )

    @app.get("/api/status")
    async def api_status() -> dict[str, Any]:
        """Current dashboard status."""
        active = em.active_run
        return {
            "active_run": {
                "run_id": active.run_id,
                "experiment_name": active.experiment.name,
                "config_path": active.config_path,
                "started_at": active.started_at.isoformat(),
            }
            if active
            else None,
            "completed_runs": [
                {
                    "run_id": r.run_id,
                    "experiment_name": r.experiment_name,
                    "status": r.status,
                }
                for r in em.completed_runs
            ],
            "connected_clients": cm.client_count,
        }

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await cm.connect(ws)
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                msg_type = msg.get("type")

                if msg_type == "start_experiment":
                    config_path = msg.get("config_path", "")
                    server_url = msg.get("server_url") or None
                    run_name = msg.get("run_name") or None
                    try:
                        run_id = await em.start_experiment(
                            config_path=config_path,
                            server_url=server_url,
                            run_name=run_name,
                        )
                        await ws.send_json(
                            {
                                "type": "experiment_started",
                                "run_id": run_id,
                            }
                        )
                    except Exception as e:
                        await ws.send_json(
                            {
                                "type": "error",
                                "message": str(e),
                            }
                        )

                elif msg_type == "stop_experiment":
                    run_id = msg.get("run_id", "")
                    await em.stop_experiment(run_id)
                    await ws.send_json(
                        {
                            "type": "experiment_stopping",
                            "run_id": run_id,
                        }
                    )

                else:
                    await ws.send_json(
                        {
                            "type": "error",
                            "message": f"Unknown command: {msg_type}",
                        }
                    )

        except WebSocketDisconnect:
            cm.disconnect(ws)

    return app


def run_server(port: int = 8420, open_browser: bool = True) -> None:
    """Start the dashboard server with uvicorn."""
    import uvicorn

    if open_browser:
        import threading
        import time
        import webbrowser

        def _open() -> None:
            time.sleep(1.0)
            webbrowser.open(f"http://localhost:{port}")

        threading.Thread(target=_open, daemon=True).start()

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(), host="0.0.0.0", port=port, log_level="info")
