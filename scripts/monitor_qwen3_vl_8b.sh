#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
interval="${1:-10}"

cd "$project_dir"
source "$project_dir/jobs/lib/q10_python.sh"
q10_load_config "$project_dir"
q10_resolve_python
python_bin="$PYTHON_BIN"
"$python_bin" scripts/monitor_vlm_progress.py \
  --votes data/vlm_pairwise_qwen3_vl_8b_formal_v2.csv \
  --total 15562 \
  --interval "$interval"
