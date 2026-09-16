#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/colin/miniconda3/envs/guvp-vlm/bin/python}"
interval="${1:-10}"

cd "$project_dir"
"$python_bin" scripts/monitor_formal_abba.py \
  --audit output/vlm_7b_definition_v8_3000_prefix_cache/pair_audit.csv \
  --total 3000 \
  --mec-samples 3 \
  --model-fragment Qwen2.5-VL-7B-Instruct \
  --interval "$interval"
