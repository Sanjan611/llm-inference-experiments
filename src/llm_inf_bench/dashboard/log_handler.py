"""Logging handler that broadcasts log records to WebSocket clients."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_inf_bench.dashboard.websocket import ConnectionManager


class WebSocketLogHandler(logging.Handler):
    """Captures log records from ``llm_inf_bench.*`` and broadcasts them.

    The handler must be installed on the root ``llm_inf_bench`` logger so it
    captures records from all sub-modules.  It schedules the broadcast as a
    fire-and-forget coroutine on the running event loop.
    """

    def __init__(self, manager: ConnectionManager) -> None:
        super().__init__(level=logging.INFO)
        self._manager = manager
        self._pending_tasks: set[asyncio.Task[None]] = set()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no event loop — nothing to do

        message = {
            "type": "log",
            "level": record.levelname.lower(),
            "message": self.format(record),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        task = loop.create_task(self._manager.broadcast(message))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
