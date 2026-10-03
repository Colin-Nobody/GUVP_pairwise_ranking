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
  --mec-samples 3 \
  --temperature 1.0 \
  --max-new-tokens 192 \
  --progress-every 1 \
  --overwrite \
  --audit-output output/vlm_qwen3_vl_8b_smoke/pair_audit.csv \
  --output data/vlm_pairwise_qwen3_vl_8b_smoke.csv
