#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/colin/miniconda3/envs/guvp-vlm/bin/python}"
votes_file="$project_dir/data/vlm_pairwise_qwen3_vl_8b_formal_mec3_bpc_v6_3000.csv"
progress_dir="$project_dir/output/vlm_qwen3_vl_8b_formal_mec3_bpc_v6_3000"
audit_file="$progress_dir/pair_audit.csv"

cd "$project_dir"
mkdir -p "$progress_dir"

resume_args=()
if [[ -f "$audit_file" ]]; then
  resume_args=(--resume)
fi

"$python_bin" vlm_pairwise.py \
  --images-dir data/street_images \
  --pairs data/vlm_pairwise_formal_balanced24_v6_3000.csv \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --cpu-offload \
  --max-pixels 401408 \
  --mec-samples 3 \
  --temperature 1.0 \
  --max-new-tokens 192 \
  --progress-every 100 \
  --progress-file "$progress_dir/progress.log" \
  --audit-output "$audit_file" \
  --output "$votes_file" \
  "${resume_args[@]}"

"$python_bin" scripts/summarize_formal_abba.py \
  --audit "$audit_file" \
  --output-dir "$progress_dir"

"$python_bin" main.py \
  --input "$votes_file" \
  --output-dir "$progress_dir" \
  --dimension visual_pollution
