# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "runpod>=1.7",
#     "httpx>=0.27",
# ]
# ///
"""
POC 4: Deploy SGLang on RunPod

Start an SGLang server on a RunPod GPU with a small model and confirm it serves
the OpenAI-compatible API. Same as POC 3 but with SGLang instead of vLLM.

Prerequisites:
    export RUNPOD_API_KEY="your_api_key"

Usage:
    uv run poc/poc_04_deploy_sglang.py

    # Use a different model
    uv run poc/poc_04_deploy_sglang.py --model "Qwen/Qwen2.5-7B-Instruct"

    # Use a specific GPU type
    uv run poc/poc_04_deploy_sglang.py --gpu-type "NVIDIA GeForce RTX 3090"

    # Reuse an already-running pod with SGLang (skips creation and cleanup)
    uv run poc/poc_04_deploy_sglang.py --pod-id abc123

Success criteria:
    The SGLang server responds and /v1/models lists the loaded model.
"""

import argparse
import json
import os
import sys
import time

import httpx
import runpod


# --- Configuration ---

POD_NAME = "poc-04-sglang-deploy"
DEFAULT_GPU_TYPE = "NVIDIA RTX 4000 Ada Generation"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
# The lmsysorg/sglang image has NO ENTRYPOINT (just CMD ["/bin/bash"]),
# so docker_args must be the full startup command.
SGLANG_IMAGE = "lmsysorg/sglang:latest"
CONTAINER_DISK_GB = 10
VOLUME_DISK_GB = 10
SGLANG_PORT = 8000
POLL_INTERVAL_SECONDS = 5
READY_TIMEOUT_SECONDS = 600
HEALTH_RETRY_INTERVAL_SECONDS = 10
HEALTH_TIMEOUT_SECONDS = 600  # Model download + load can take a while


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


def get_proxy_url(pod_id: str, port: int) -> str:
    """Construct the RunPod proxy URL for a given pod and port."""
    return f"https://{pod_id}-{port}.proxy.runpod.net"


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

        try:
            pod = runpod.get_pod(pod_id)
        except Exception as e:
            print(f"  [{elapsed:.0f}s] API call failed (will retry): {e}")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

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


def wait_for_sglang_health(base_url: str) -> None:
    """Poll the SGLang /health endpoint until it responds 200."""
    health_url = f"{base_url}/health"
    print(f"  Health URL: {health_url}")
    print(
        f"  Polling every {HEALTH_RETRY_INTERVAL_SECONDS}s "
        f"(timeout: {HEALTH_TIMEOUT_SECONDS}s)..."
    )
    print("  (Model download + loading may take several minutes)")
    start = time.time()

    while True:
        elapsed = time.time() - start
        if elapsed > HEALTH_TIMEOUT_SECONDS:
            raise TimeoutError(
                f"SGLang health endpoint did not respond within {HEALTH_TIMEOUT_SECONDS}s"
            )

        try:
            response = httpx.get(health_url, timeout=10, follow_redirects=True)
            if response.status_code == 200:
                print(f"  [{elapsed:.0f}s] SGLang server is healthy!")
                return
            print(f"  [{elapsed:.0f}s] Health check returned status {response.status_code}")
        except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as e:
            print(f"  [{elapsed:.0f}s] Not ready yet: {type(e).__name__}")

        time.sleep(HEALTH_RETRY_INTERVAL_SECONDS)


def build_sglang_cmd(model: str) -> str:
    """Build the full SGLang server startup command.

    The lmsysorg/sglang image has no ENTRYPOINT (just CMD ["/bin/bash"]),
    so docker_args must be the complete command.
    """
    return (
        f"python3 -m sglang.launch_server "
        f"--model-path {model} "
        f"--host 0.0.0.0 "
        f"--port {SGLANG_PORT}"
    )


def step_create_pod(gpu_type: str, model: str) -> dict:
    """Create a pod running an SGLang server."""
    sglang_cmd = build_sglang_cmd(model)

    print_section("Step 1: Create Pod")
    print(f"  GPU type:  {gpu_type}")
    print(f"  Image:     {SGLANG_IMAGE}")
    print(f"  Model:     {model}")
    print(f"  Disk:      {CONTAINER_DISK_GB} GB")
    print(f"  Command:   {sglang_cmd}")
    print()

    pod = runpod.create_pod(
        name=POD_NAME,
        image_name=SGLANG_IMAGE,
        gpu_type_id=gpu_type,
        gpu_count=1,
        container_disk_in_gb=CONTAINER_DISK_GB,
        volume_in_gb=VOLUME_DISK_GB,
        support_public_ip=True,
        ports=f"{SGLANG_PORT}/http",
        docker_args=sglang_cmd,
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
        print("  Will attempt health check anyway...")
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


def step_wait_for_sglang(pod_id: str) -> str:
    """Wait for SGLang to become healthy and return the base URL."""
    print_section("Step 3: Wait for SGLang Health")
    base_url = get_proxy_url(pod_id, SGLANG_PORT)
    wait_for_sglang_health(base_url)
    return base_url


def step_query_models(base_url: str) -> None:
    """Query the /v1/models endpoint and print the results."""
    print_section("Step 4: Query /v1/models")
    models_url = f"{base_url}/v1/models"
    print(f"  URL: {models_url}")
    print()

    response = httpx.get(models_url, timeout=10, follow_redirects=True)
    response.raise_for_status()

    data = response.json()
    models = data.get("data", [])

    print(f"  Found {len(models)} model(s):")
    print()
    for model in models:
        print(f"  Model ID:    {model.get('id', 'N/A')}")
        print(f"  Object:      {model.get('object', 'N/A')}")
        print(f"  Owned by:    {model.get('owned_by', 'N/A')}")
        print()

    print("  Full response:")
    print(f"  {json.dumps(data, indent=2)}")


def step_terminate_pod(pod_id: str) -> None:
    """Terminate the pod."""
    print_section("Step 5: Terminate Pod")
    print(f"  Terminating pod {pod_id}...")
    runpod.terminate_pod(pod_id)
    print("  Pod terminated successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="POC 4: Deploy SGLang on RunPod"
    )
    parser.add_argument(
        "--gpu-type",
        default=DEFAULT_GPU_TYPE,
        help=f"GPU type ID to use (default: {DEFAULT_GPU_TYPE})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model to serve (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--pod-id",
        default=None,
        help="Reuse an existing pod instead of creating a new one (skips creation and cleanup)",
    )
    args = parser.parse_args()

    api_key = check_api_key()
    runpod.api_key = api_key

    print_section("POC 4: Deploy SGLang on RunPod")
    print("  Validates that SGLang boots on RunPod, loads a model, and")
    print("  exposes the OpenAI-compatible API.")
    print(f"  Model: {args.model}")

    reusing_pod = args.pod_id is not None
    pod_id = None

    try:
        if reusing_pod:
            # Step 1: Reuse existing pod
            pod = step_reuse_pod(args.pod_id)
            pod_id = pod["id"]
        else:
            # Step 1: Create pod
            pod = step_create_pod(args.gpu_type, args.model)
            pod_id = pod["id"]

            # Step 2: Wait for running
            step_wait_for_running(pod_id)

        # Step 3: Wait for SGLang to be healthy
        base_url = step_wait_for_sglang(pod_id)

        # Step 4: Query /v1/models
        step_query_models(base_url)

        # Step 5: Cleanup (only if we created the pod)
        if not reusing_pod:
            step_terminate_pod(pod_id)
            pod_id = None  # Already terminated, don't clean up again

        print_section("SUCCESS")
        print("  SGLang server deployed and /v1/models returned the loaded model.")
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
