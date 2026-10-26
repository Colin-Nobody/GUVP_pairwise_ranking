#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
interval="${1:-10}"

cd "$project_dir"
source "$project_dir/jobs/lib/q10_python.sh"
q10_load_config "$project_dir"
q10_resolve_python
python_bin="$PYTHON_BIN"
"$python_bin" scripts/monitor_formal_abba.py \
  --audit output/vlm_qwen3_vl_8b_formal_mec3_bpc_v6_3000_prefix_cache/pair_audit.csv \
  --total 3000 \
  --mec-samples 3 \
  --model-fragment Qwen3-VL-8B-Instruct \
  --interval "$interval"
