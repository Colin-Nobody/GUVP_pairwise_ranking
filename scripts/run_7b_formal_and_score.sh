#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
votes_file="$project_dir/data/vlm_pairwise_7b_formal_v2.csv"
progress_dir="$project_dir/output/vlm_7b_formal_v2"

cd "$project_dir"
mkdir -p "$progress_dir"

resume_args=()
if [[ -f "$votes_file" ]]; then
  resume_args=(--resume)
fi

"$python_bin" vlm_pairwise.py \
  --images-dir data/street_images \
  --pairs data/vlm_pairwise_formal_v2.csv \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --max-pixels 401408 \
  --progress-every 100 \
  --progress-file "$progress_dir/progress.log" \
  --output "$votes_file" \
  "${resume_args[@]}"

"$python_bin" main.py \
  --input "$votes_file" \
  --output-dir "$progress_dir" \
  --dimension visual_pollution
