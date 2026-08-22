#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hf_cache="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$hf_cache"

docker run --rm --gpus all --ipc=host -it \
  -v "$project_dir:/workspace" \
  -v "$hf_cache:/root/.cache/huggingface" \
  -w /workspace \
  nvcr.io/nvidia/pytorch:25.11-py3 \
  bash
