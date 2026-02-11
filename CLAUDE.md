# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A benchmarking and observability framework for measuring LLM inference performance using vLLM and SGLang on RunPod GPUs. The CLI tool is called `llm-inf-bench`.

**Status:** Design phase — no implementation yet.

## Key Design Documents

- **PROJECT_OVERVIEW.md** — Motivation, features, architecture diagram, key concepts (KV cache, prefix caching, continuous batching, PagedAttention vs RadixAttention)
- **INTERFACE.md** — CLI commands, YAML config format, progress output, cost confirmation flow, directory structure
- **PROOF_OF_CONCEPTS.md** — Standalone validation scripts ordered by dependency (start with POC 1 → 2 → 3 → 6 → 8 → 9 → 12 → 14)
