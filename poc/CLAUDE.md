# POC Scripts — Notes for Claude

## RunPod `docker_args` behavior

The `docker_args` parameter in `runpod.create_pod()` is appended as arguments to the Docker image's ENTRYPOINT. It does **not** replace the entrypoint.

For images with a built-in entrypoint (like `vllm/vllm-openai` which has `ENTRYPOINT ["vllm", "serve"]`), pass only the arguments — not a full command:

```python
# Wrong — gets appended to entrypoint, producing: vllm serve python -m vllm...
docker_args="python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000"

# Right — produces: vllm serve Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000
docker_args="Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000"
```

For images without an entrypoint (like `runpod/pytorch:*`, `lmsysorg/sglang`), pass the full command since there's nothing to append to.
