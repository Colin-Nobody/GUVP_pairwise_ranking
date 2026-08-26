#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/colin/miniconda3/envs/guvp-vlm/bin/python}"
votes_file="$project_dir/data/vlm_pairwise_qwen3_vl_8b_formal_v2.csv"
progress_dir="$project_dir/output/vlm_qwen3_vl_8b_formal_v2"

cd "$project_dir"
mkdir -p "$progress_dir"

resume_args=()
if [[ -f "$votes_file" ]]; then
  resume_args=(--resume)
fi

"$python_bin" vlm_pairwise.py \
  --images-dir data/street_images \
  --pairs data/vlm_pairwise_formal_v2.csv \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --cpu-offload \
  --max-pixels 401408 \
  --progress-every 100 \
  --progress-file "$progress_dir/progress.log" \
  --output "$votes_file" \
  "${resume_args[@]}"

"$python_bin" main.py \
  --input "$votes_file" \
  --output-dir "$progress_dir" \
  --dimension visual_pollution
