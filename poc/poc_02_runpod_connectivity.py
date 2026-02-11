# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "runpod>=1.7",
#     "httpx>=0.27",
# ]
# ///
"""
POC 2: RunPod Connectivity

Establish an HTTP connection from the local machine to a running RunPod pod.
Creates a pod running a simple HTTP server, sends a request through RunPod's
proxy, and verifies the response.

Prerequisites:
    export RUNPOD_API_KEY="your_api_key"

Usage:
    uv run poc/poc_02_runpod_connectivity.py

    # Use a specific GPU type
    uv run poc/poc_02_runpod_connectivity.py --gpu-type "NVIDIA GeForce RTX 3090"

    # Reuse an already-running pod (skips creation and cleanup)
    uv run poc/poc_02_runpod_connectivity.py --pod-id abc123

Success criteria:
    Local machine receives an HTTP response from the pod.
"""

import argparse
import os
import sys
import time

import httpx
import runpod


# --- Configuration ---

POD_NAME = "poc-02-connectivity-test"
DEFAULT_GPU_TYPE = "NVIDIA GeForce RTX 4090"
CONTAINER_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CONTAINER_DISK_GB = 10
VOLUME_DISK_GB = 10
HTTP_PORT = 8000
DOCKER_START_CMD = f"python -m http.server {HTTP_PORT}"
POLL_INTERVAL_SECONDS = 5
READY_TIMEOUT_SECONDS = 300
HTTP_RETRY_INTERVAL_SECONDS = 5
HTTP_TIMEOUT_SECONDS = 120


def check_api_key() -> str:
    """Ensure the RunPod API key is configured."""
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        print("Error: RUNPOD_API_KEY environment variable is not set.")
        print("Get your API key from: https://www.runpod.io/console/user/settings")
        sys.exit(1)
    return api_key


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_pod_info(pod: dict) -> None:
    """Print key pod details."""
    print(f"  Pod ID:          {pod.get('id', 'N/A')}")
    print(f"  Name:            {pod.get('name', 'N/A')}")
    print(f"  Image:           {pod.get('imageName', 'N/A')}")
    print(f"  Desired Status:  {pod.get('desiredStatus', 'N/A')}")
    print(f"  Cost/hr:         ${pod.get('costPerHr', 'N/A')}")

    runtime = pod.get("runtime")
    if runtime and runtime.get("ports"):
        print("  Ports:")
        for port_info in runtime["ports"]:
            public = "public" if port_info.get("isIpPublic") else "private"
            print(
                f"    {port_info.get('privatePort')} -> "
                f"{port_info.get('ip')}:{port_info.get('publicPort')} "
                f"({port_info.get('type')}, {public})"
            )
        uptime = runtime.get("uptimeInSeconds")
        if uptime is not None:
            print(f"  Uptime:          {uptime}s")


def wait_for_pod_ready(pod_id: str) -> dict:
    """Poll until the pod is running and has port mappings available."""
    print(f"  Polling every {POLL_INTERVAL_SECONDS}s (timeout: {READY_TIMEOUT_SECONDS}s)...")
    start = time.time()
    last_status = None

    while True:
        elapsed = time.time() - start
        if elapsed > READY_TIMEOUT_SECONDS:
            raise TimeoutError(
                f"Pod {pod_id} did not become ready within {READY_TIMEOUT_SECONDS}s"
            )

        pod = runpod.get_pod(pod_id)
        status = pod.get("desiredStatus")
        runtime = pod.get("runtime")

        if status != last_status:
            print(f"  [{elapsed:.0f}s] Status: {status}")
            last_status = status

        if status == "RUNNING" and runtime is not None:
            ports = runtime.get("ports")
            if ports:
                print(f"  [{elapsed:.0f}s] Pod is ready!")
                return pod

        time.sleep(POLL_INTERVAL_SECONDS)


def get_proxy_url(pod_id: str, port: int) -> str:
    """Construct the RunPod proxy URL for a given pod and port."""
    return f"https://{pod_id}-{port}.proxy.runpod.net/"


def wait_for_http_server(url: str) -> httpx.Response:
    """Poll the HTTP server until it responds, with retries."""
    print(f"  URL: {url}")
    print(
        f"  Polling every {HTTP_RETRY_INTERVAL_SECONDS}s "
        f"(timeout: {HTTP_TIMEOUT_SECONDS}s)..."
    )
    start = time.time()

    while True:
        elapsed = time.time() - start
        if elapsed > HTTP_TIMEOUT_SECONDS:
            raise TimeoutError(
                f"HTTP server did not respond within {HTTP_TIMEOUT_SECONDS}s"
            )

        try:
            response = httpx.get(url, timeout=10, follow_redirects=True)
            print(f"  [{elapsed:.0f}s] HTTP server responded (status {response.status_code})")
            return response
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as e:
            print(f"  [{elapsed:.0f}s] Not ready yet: {type(e).__name__}")
            time.sleep(HTTP_RETRY_INTERVAL_SECONDS)


def step_create_pod(gpu_type: str) -> dict:
    """Create a pod running a simple HTTP server."""
    print_section("Step 1: Create Pod")
    print(f"  GPU type:  {gpu_type}")
    print(f"  Image:     {CONTAINER_IMAGE}")
    print(f"  Disk:      {CONTAINER_DISK_GB} GB")
    print(f"  Command:   {DOCKER_START_CMD}")
    print()

    pod = runpod.create_pod(
        name=POD_NAME,
        image_name=CONTAINER_IMAGE,
        gpu_type_id=gpu_type,
        gpu_count=1,
        container_disk_in_gb=CONTAINER_DISK_GB,
        volume_in_gb=VOLUME_DISK_GB,
        support_public_ip=True,
        ports=f"{HTTP_PORT}/http",
        docker_args=DOCKER_START_CMD,
    )

    pod_id = pod["id"]
    print(f"  Pod created successfully!")
    print(f"  Pod ID: {pod_id}")
    return pod


def step_reuse_pod(pod_id: str) -> dict:
    """Fetch an existing pod's status."""
    print_section("Step 1: Reuse Existing Pod")
    print(f"  Pod ID: {pod_id}")
    print()

    pod = runpod.get_pod(pod_id)
    status = pod.get("desiredStatus")
    print(f"  Status: {status}")

    if status != "RUNNING":
        print(f"  Warning: pod is not running (status: {status})")
        print("  Will attempt connectivity check anyway...")
    else:
        print("  Pod is running.")

    print()
    print("  Pod details:")
    print_pod_info(pod)
    return pod


def step_wait_for_running(pod_id: str) -> dict:
    """Poll until the pod is running."""
    print_section("Step 2: Wait for Pod Ready")
    pod = wait_for_pod_ready(pod_id)
    print()
    print("  Connection details:")
    print_pod_info(pod)
    return pod


def step_test_connectivity(pod_id: str) -> None:
    """Send an HTTP request through the RunPod proxy and verify the response."""
    print_section("Step 3: Test HTTP Connectivity")

    proxy_url = get_proxy_url(pod_id, HTTP_PORT)
    print("  Waiting for HTTP server to become reachable...")
    response = wait_for_http_server(proxy_url)

    print()
    print("  Response details:")
    print(f"  Status code: {response.status_code}")
    print(f"  Headers:")
    for key, value in response.headers.items():
        print(f"    {key}: {value}")

    body_snippet = response.text[:500]
    print(f"  Body (first 500 chars):")
    for line in body_snippet.splitlines():
        print(f"    {line}")

    if response.status_code == 200:
        print("\n  HTTP connectivity verified!")
    else:
        print(f"\n  Warning: unexpected status code {response.status_code}")


def step_terminate_pod(pod_id: str) -> None:
    """Terminate the pod."""
    print_section("Step 4: Terminate Pod")
    print(f"  Terminating pod {pod_id}...")
    runpod.terminate_pod(pod_id)
    print("  Pod terminated successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="POC 2: RunPod Connectivity - HTTP connection test"
    )
    parser.add_argument(
        "--gpu-type",
        default=DEFAULT_GPU_TYPE,
        help=f"GPU type ID to use (default: {DEFAULT_GPU_TYPE})",
    )
    parser.add_argument(
        "--pod-id",
        default=None,
        help="Reuse an existing pod instead of creating a new one (skips creation and cleanup)",
    )
    args = parser.parse_args()

    api_key = check_api_key()
    runpod.api_key = api_key

    print_section("POC 2: RunPod Connectivity")
    print("  Validates HTTP connectivity from local machine to a RunPod pod.")

    reusing_pod = args.pod_id is not None
    pod_id = None

    try:
        if reusing_pod:
            # Step 1: Reuse existing pod
            pod = step_reuse_pod(args.pod_id)
            pod_id = pod["id"]
        else:
            # Step 1: Create pod
            pod = step_create_pod(args.gpu_type)
            pod_id = pod["id"]

            # Step 2: Wait for running
            step_wait_for_running(pod_id)

        # Step 3: Test HTTP connectivity
        step_test_connectivity(pod_id)

        # Step 4: Cleanup (only if we created the pod)
        if not reusing_pod:
            step_terminate_pod(pod_id)
            pod_id = None  # Already terminated, don't clean up again

        print_section("SUCCESS")
        print("  Local machine received an HTTP response from the RunPod pod.")
        print()

    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")
    except Exception as e:
        print(f"\n  Error: {e}")
        raise
    finally:
        # Safety net: always try to terminate pods we created, even on error
        if pod_id is not None and not reusing_pod:
            print(f"\n  Cleaning up: terminating pod {pod_id}...")
            try:
                runpod.terminate_pod(pod_id)
                print("  Cleanup complete.")
            except Exception as cleanup_err:
                print(f"  Cleanup failed: {cleanup_err}")
                print(f"  MANUAL ACTION REQUIRED: terminate pod {pod_id} via RunPod console.")


if __name__ == "__main__":
    main()
