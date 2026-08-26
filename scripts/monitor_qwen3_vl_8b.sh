#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/colin/miniconda3/envs/guvp-vlm/bin/python}"
interval="${1:-10}"

cd "$project_dir"
"$python_bin" scripts/monitor_vlm_progress.py \
  --votes data/vlm_pairwise_qwen3_vl_8b_formal_v2.csv \
  --total 15562 \
  --interval "$interval"
