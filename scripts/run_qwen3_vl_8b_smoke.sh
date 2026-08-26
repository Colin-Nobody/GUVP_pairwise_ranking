#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/colin/miniconda3/envs/guvp-vlm/bin/python}"

cd "$project_dir"

"$python_bin" vlm_pairwise.py \
  --images-dir data/street_images \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --cpu-offload \
  --max-pairs 1 \
  --max-pixels 401408 \
  --progress-every 1 \
  --overwrite \
  --output data/vlm_pairwise_qwen3_vl_8b_smoke.csv
