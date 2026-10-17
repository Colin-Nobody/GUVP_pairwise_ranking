#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/colin/miniconda3/envs/guvp-vlm/bin/python}"
votes_file="$project_dir/data/vlm_pairwise_3b_formal_mec3_bpc_v6_3000.csv"
progress_dir="$project_dir/output/vlm_3b_formal_mec3_bpc_v6_3000"
audit_file="$progress_dir/pair_audit.csv"

cd "$project_dir"
mkdir -p "$progress_dir"

exec 9>"$progress_dir/run.lock"
if ! flock -n 9; then
  echo "Another 3B formal run already holds $progress_dir/run.lock" >&2
  exit 1
fi

if pgrep -af '[v]lm_pairwise.py' >/dev/null; then
  echo "Another VLM inference process is already running; refusing to share the GPU." >&2
  pgrep -af '[v]lm_pairwise.py' >&2
  exit 1
fi

"$python_bin" scripts/preflight_formal_abba.py \
  --images-dir data/street_images \
  --pairs data/vlm_pairwise_formal_balanced24_v6_3000.csv \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --expected-images 250 \
  --expected-pairs 3000 \
  --mec-samples 3

resume_args=()
if [[ -f "$audit_file" ]]; then
  resume_args=(--resume)
fi

"$python_bin" vlm_pairwise.py \
  --images-dir data/street_images \
  --pairs data/vlm_pairwise_formal_balanced24_v6_3000.csv \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --max-pixels 401408 \
  --mec-samples 3 \
  --temperature 1.0 \
  --max-new-tokens 192 \
  --progress-every 30 \
  --progress-file "$progress_dir/progress.log" \
  --audit-output "$audit_file" \
  --output "$votes_file" \
  "${resume_args[@]}" 2>&1 | tee -a "$progress_dir/run.log"

"$python_bin" scripts/summarize_formal_abba.py \
  --audit "$audit_file" \
  --output-dir "$progress_dir"

"$python_bin" main.py \
  --input "$votes_file" \
  --output-dir "$progress_dir" \
  --dimension visual_pollution
