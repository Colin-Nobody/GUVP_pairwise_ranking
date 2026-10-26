#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/colin/miniconda3/envs/guvp-vlm/bin/python}"
data_dir="$project_dir/data/external_vp_4city_200"
image_dir="$data_dir/images"
pair_file="$data_dir/pairs_200_13_per_image.csv"
output_dir="$project_dir/output/vlm_external_4city_200_v6"
audit_file="$output_dir/pair_audit.csv"
votes_file="$output_dir/vlm_pairwise_votes.csv"

cd "$project_dir"
mkdir -p "$output_dir"
exec 9>"$output_dir/run.lock"
if ! flock -n 9; then
  echo "Another external V6 run holds $output_dir/run.lock" >&2
  exit 1
fi
if [[ -f "$votes_file" && ! -f "$audit_file" ]]; then
  echo "Votes exist without an audit file; inspect outputs before continuing." >&2
  exit 1
fi
if pgrep -af '[v]lm_pairwise.py' >/dev/null; then
  echo "Another VLM inference process is running; refusing to share the GPU." >&2
  pgrep -af '[v]lm_pairwise.py' >&2
  exit 1
fi

"$python_bin" faireval_mec_bpc_v6/scripts/preflight_formal_abba.py \
  --images-dir "$image_dir" \
  --pairs "$pair_file" \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --expected-images 200 \
  --expected-pairs 1300 \
  --mec-samples 3

resume_args=()
if [[ -f "$audit_file" ]]; then
  resume_args=(--resume)
fi

"$python_bin" vlm_pairwise.py \
  --images-dir "$image_dir" \
  --pairs "$pair_file" \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --prompt-file prompts/vp_pairwise_faireval_mec_bpc_v6.txt \
  --prompt-version vp-faireval-mec-bpc-score-v6 \
  --score-direction higher_is_cleaner \
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

"$python_bin" main.py \
  --input "$votes_file" \
  --output-dir "$output_dir" \
  --dimension visual_pollution
