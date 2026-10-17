#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/colin/miniconda3/envs/guvp-vlm/bin/python}"
data_dir="$project_dir/data/expansion_750"
image_dir="$data_dir/street_images_1000"
pair_file="$data_dir/pairs_1000_13_per_image.csv"
output_dir="$project_dir/output/vlm_expansion_1000"
audit_file="$output_dir/pair_audit.csv"
votes_file="$data_dir/vlm_pairwise_votes.csv"

cd "$project_dir"
mkdir -p "$output_dir"

exec 9>"$output_dir/run.lock"
if ! flock -n 9; then
  echo "Another expansion run already holds $output_dir/run.lock" >&2
  exit 1
fi

if pgrep -af '[v]lm_pairwise.py' >/dev/null; then
  echo "Another VLM inference process is already running; refusing to share the GPU." >&2
  pgrep -af '[v]lm_pairwise.py' >&2
  exit 1
fi

"$python_bin" faireval_mec_bpc_v6/scripts/preflight_formal_abba.py \
  --images-dir "$image_dir" \
  --pairs "$pair_file" \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --expected-images 1000 \
  --expected-pairs 6500 \
  --mec-samples 3

resume_args=()
if [[ -f "$audit_file" ]]; then
  resume_args=(--resume)
fi

"$python_bin" vlm_pairwise.py \
  --images-dir "$image_dir" \
  --pairs "$pair_file" \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --max-pixels 401408 \
  --mec-samples 3 \
  --mec-batch-size 1 \
  --reuse-prefix-cache \
  --temperature 1.0 \
  --max-new-tokens 192 \
  --progress-every 30 \
  --progress-file "$output_dir/progress.log" \
  --audit-output "$audit_file" \
  --output "$votes_file" \
  --timing-output "$output_dir/timing.json" \
  --seed 42 \
  "${resume_args[@]}" 2>&1 | tee -a "$output_dir/run.log"

"$python_bin" faireval_mec_bpc_v6/scripts/summarize_formal_abba.py \
  --audit "$audit_file" \
  --output-dir "$output_dir"

"$python_bin" main.py \
  --input "$votes_file" \
  --output-dir "$output_dir" \
  --dimension visual_pollution
