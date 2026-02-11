# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "runpod>=1.7",
# ]
# ///
"""
POC 1: RunPod API Basics

Programmatically create, start, stop, and terminate a RunPod GPU pod
using their Python SDK, validating the full pod lifecycle.

Prerequisites:
    export RUNPOD_API_KEY="your_api_key"

Usage:
    uv run poc/poc_01_runpod_api_basics.py

    # Use a specific GPU type (default: cheapest available)
    uv run poc/poc_01_runpod_api_basics.py --gpu-type "NVIDIA GeForce RTX 3090"

Success criteria:
    Pod goes through the full lifecycle:
    create → running → stopped → terminated
    without manual intervention.
"""

import argparse
import os
import sys
import time

import runpod


# --- Configuration ---

POD_NAME = "poc-01-lifecycle-test"
DEFAULT_GPU_TYPE = "NVIDIA GeForce RTX 4090"
CONTAINER_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CONTAINER_DISK_GB = 10
VOLUME_DISK_GB = 10
POLL_INTERVAL_SECONDS = 5
READY_TIMEOUT_SECONDS = 300


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


def step_create_pod(gpu_type: str) -> dict:
    """Step 1: Create a pod with the specified GPU type."""
    print_section("Step 1: Create Pod")
    print(f"  GPU type:  {gpu_type}")
    print(f"  Image:     {CONTAINER_IMAGE}")
    print(f"  Disk:      {CONTAINER_DISK_GB} GB")
    print()

    pod = runpod.create_pod(
        name=POD_NAME,
        image_name=CONTAINER_IMAGE,
        gpu_type_id=gpu_type,
        gpu_count=1,
        container_disk_in_gb=CONTAINER_DISK_GB,
        volume_in_gb=VOLUME_DISK_GB,
        support_public_ip=True,
        ports="8000/http",
    )

    pod_id = pod["id"]
    print(f"  Pod created successfully!")
    print(f"  Pod ID: {pod_id}")
    return pod


def step_wait_for_running(pod_id: str) -> dict:
    """Step 2: Poll until the pod is running."""
    print_section("Step 2: Wait for Pod Ready")
    pod = wait_for_pod_ready(pod_id)
    print()
    print("  Connection details:")
    print_pod_info(pod)
    return pod


def step_stop_pod(pod_id: str) -> None:
    """Step 3: Stop the pod."""
    print_section("Step 3: Stop Pod")
    print(f"  Stopping pod {pod_id}...")
    result = runpod.stop_pod(pod_id)
    status = result.get("desiredStatus", "unknown")
    print(f"  Result: desiredStatus = {status}")

    # Verify it's actually stopped
    print("  Verifying stopped state...")
    time.sleep(5)
    pod = runpod.get_pod(pod_id)
    print(f"  Confirmed status: {pod.get('desiredStatus')}")


def step_terminate_pod(pod_id: str) -> None:
    """Step 4: Terminate the pod."""
    print_section("Step 4: Terminate Pod")
    print(f"  Terminating pod {pod_id}...")
    runpod.terminate_pod(pod_id)
    print("  Pod terminated successfully.")

    # Verify it no longer appears in our pods list
    print("  Verifying termination...")
    time.sleep(5)
    pods = runpod.get_pods()
    remaining_ids = [p["id"] for p in pods]
    if pod_id in remaining_ids:
        print(f"  Warning: pod {pod_id} still appears in pod list.")
    else:
        print(f"  Confirmed: pod {pod_id} is no longer in pod list.")


def main():
    parser = argparse.ArgumentParser(description="POC 1: RunPod API Basics - Pod Lifecycle")
    parser.add_argument(
        "--gpu-type",
        default=DEFAULT_GPU_TYPE,
        help=f"GPU type ID to use (default: {DEFAULT_GPU_TYPE})",
    )
    args = parser.parse_args()

    api_key = check_api_key()
    runpod.api_key = api_key

    print_section("POC 1: RunPod API Basics")
    print("  Validates the full pod lifecycle:")
    print("  create → running → stopped → terminated")

    pod_id = None
    try:
        # Step 1: Create
        pod = step_create_pod(args.gpu_type)
        pod_id = pod["id"]

        # Step 2: Wait for running
        step_wait_for_running(pod_id)

        # Step 3: Stop
        step_stop_pod(pod_id)

        # Step 4: Terminate
        step_terminate_pod(pod_id)
        pod_id = None  # Already terminated, don't clean up again

        print_section("SUCCESS")
        print("  Pod completed full lifecycle:")
        print("  create → running → stopped → terminated")
        print()

    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")
    except Exception as e:
        print(f"\n  Error: {e}")
        raise
    finally:
        # Safety net: always try to terminate the pod if it still exists
        if pod_id is not None:
            print(f"\n  Cleaning up: terminating pod {pod_id}...")
            try:
                runpod.terminate_pod(pod_id)
                print("  Cleanup complete.")
            except Exception as cleanup_err:
                print(f"  Cleanup failed: {cleanup_err}")
                print(f"  MANUAL ACTION REQUIRED: terminate pod {pod_id} via RunPod console.")


if __name__ == "__main__":
    main()
