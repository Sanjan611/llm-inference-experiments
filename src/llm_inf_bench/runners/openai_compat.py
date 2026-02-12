"""OpenAI-compatible runner for vLLM and SGLang inference servers."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone

import httpx

from llm_inf_bench.metrics.collector import RequestResult
from llm_inf_bench.runners.base import HealthCheckTimeout, Runner, RunnerError

logger = logging.getLogger(__name__)


class OpenAICompatibleRunner(Runner):
    """Shared runner for frameworks exposing the OpenAI-compatible API.

    Both vLLM and SGLang serve ``/health`` and ``/v1/chat/completions``
    with identical SSE streaming, so this base class implements the full
    protocol.  Subclasses only need to set default polling intervals.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        health_interval: float = 5.0,
        health_timeout: float = 600.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._health_interval = health_interval
        self._health_timeout = health_timeout
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            follow_redirects=True,
            timeout=httpx.Timeout(connect=10, read=120, write=10, pool=10),
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def http_client(self) -> httpx.AsyncClient:
        """Expose the HTTP client for the GPU metrics scraper."""
        return self._client

    @property
    def base_url(self) -> str:
        """Expose the base URL for external consumers."""
        return self._base_url

    async def wait_for_health(
        self,
        timeout: float | None = None,
        interval: float | None = None,
    ) -> None:
        """Poll GET /health until it returns 200."""
        timeout = timeout if timeout is not None else self._health_timeout
        interval = interval if interval is not None else self._health_interval
        start = time.monotonic()
        last_error: str | None = None

        while True:
            elapsed = time.monotonic() - start
            if elapsed > timeout:
                msg = f"Server not healthy after {timeout:.0f}s"
                if last_error:
                    msg += f" (last error: {last_error})"
                raise HealthCheckTimeout(msg)

            try:
                resp = await self._client.get("/health")
                if resp.status_code == 200:
                    logger.info("Health check passed after %.1fs", elapsed)
                    return
                last_error = f"HTTP {resp.status_code}"
            except httpx.ConnectError:
                last_error = "connection refused"
            except httpx.TimeoutException:
                last_error = "timeout"
            except httpx.RemoteProtocolError as e:
                last_error = f"protocol error: {e}"

            logger.debug(
                "Health check attempt at %.1fs: %s", elapsed, last_error
            )
            await asyncio.sleep(interval)

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> RequestResult:
        """Send a streaming chat completion and measure timing."""
        started_at = datetime.now(timezone.utc)
        t_start = time.perf_counter()

        try:
            return await self._stream_completion(
                messages, model, max_tokens, temperature, started_at, t_start
            )
        except Exception as e:
            t_end = time.perf_counter()
            logger.warning("Request failed: %s", e)
            return RequestResult(
                request_index=0,
                error=str(e),
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                e2e_latency_ms=(t_end - t_start) * 1000,
            )

    async def _stream_completion(
        self,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        started_at: datetime,
        t_start: float,
    ) -> RequestResult:
        """Internal streaming implementation."""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        content_timestamps: list[float] = []
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        content_chunk_count = 0

        async with self._client.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RunnerError(
                    f"HTTP {resp.status_code}: {body.decode(errors='replace')}"
                )

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Extract usage from final chunk
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")

                # Check for content delta
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    content_timestamps.append(time.perf_counter())
                    content_chunk_count += 1

        t_end = time.perf_counter()

        # Compute timing metrics
        ttft_ms: float | None = None
        inter_token_latencies: list[float] = []

        if content_timestamps:
            ttft_ms = (content_timestamps[0] - t_start) * 1000
            for i in range(1, len(content_timestamps)):
                delta_ms = (content_timestamps[i] - content_timestamps[i - 1]) * 1000
                inter_token_latencies.append(delta_ms)

        # Fallback: use content chunk count if server didn't report usage
        if completion_tokens is None and content_chunk_count > 0:
            completion_tokens = content_chunk_count

        return RequestResult(
            request_index=0,
            ttft_ms=ttft_ms,
            e2e_latency_ms=(t_end - t_start) * 1000,
            inter_token_latencies_ms=inter_token_latencies,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
