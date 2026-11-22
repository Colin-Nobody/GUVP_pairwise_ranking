#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/share/home/sgxcw3/GUVP_pairwise_ranking}"
data_root="$project_root/data/vlm_hpc_10k"
images_dir="$data_root/../e4b5k/images"
manifest="$data_root/../e4b5k/m.csv"
pairs="$data_root/../e4b5k/p.csv"
prompt="$project_root/prompts/vp_pairwise_definition_v8.txt"
output_dir="$project_root/output/vlm_qwen3_5k_v8_pairbatch50_mec3"
python_bin="${PYTHON_BIN:-python}"
model="Qwen/Qwen3-VL-8B-Instruct"
pilot_pairs="${PILOT_PAIRS:-100}"
full_mec_samples="3"
full_mec_batch="1"
full_pair_batch="50"

cd "$project_root"
mkdir -p "$output_dir" "$output_dir/benchmark"
exec 9>"$output_dir/run.lock"
if ! flock -n 9; then
  echo "Another 10k V8 run holds $output_dir/run.lock" >&2
  exit 1
fi

for item in "$images_dir" "$manifest" "$pairs" "$prompt" vlm_pairwise.py main.py scripts/preflight_formal_abba.py; do
  [[ -e "$item" ]] || { echo "Required path not found: $item" >&2; exit 1; }
done

vlm_help="$("$python_bin" vlm_pairwise.py --help 2>&1)"
if ! grep -q -- '--gpu-only' <<< "$vlm_help"; then
  echo "The HPC vlm_pairwise.py is outdated and lacks --gpu-only enforcement." >&2
  echo "Sync the current project vlm_pairwise.py before submitting this workflow." >&2
  exit 1
fi

"$python_bin" scripts/preflight_formal_abba.py \
  --images-dir "$images_dir" --pairs "$pairs" \
  --model "$model" --prompt-file "$prompt" \
  --prompt-version vp-definition-score-v8 \
  --score-direction lower_is_cleaner \
  --expected-images 5000 --expected-pairs 32500 --mec-samples "$full_mec_samples"

run_vlm() {
  local label="$1" mec_batch="$2" pair_batch="$3" pair_limit="$4" run_out="$5"
  local resume_args=()
  mkdir -p "$run_out"
  if [[ -f "$run_out/pair_audit.csv" ]]; then
    resume_args=(--resume)
  elif [[ -e "$run_out/pair_audit.csv" || -e "$run_out/vlm_pairwise_votes.csv" ]]; then
    echo "Incomplete output without a resumable audit in $run_out; inspect it first." >&2
    exit 1
  fi
  "$python_bin" hpc_vlm_10k/vlm_qwen3_local.py \
    --images-dir "$images_dir" --pairs "$pairs" --limit-pairs "$pair_limit" \
    --model "$model" --prompt-file "$prompt" \
    --prompt-version vp-definition-score-v8 \
    --score-direction lower_is_cleaner \
    --max-pixels 401408 --mec-samples "$full_mec_samples" --mec-batch-size "$mec_batch" \
    --pair-batch-size "$pair_batch" \
    --temperature 1.0 --max-new-tokens 192 --gpu-only \
    --progress-every 10 --progress-file "$run_out/progress.log" \
    --audit-output "$run_out/pair_audit.csv" \
    --output "$run_out/vlm_pairwise_votes.csv" \
    --timing-output "$run_out/timing.json" --seed 42 \
    "${resume_args[@]}" 2>&1 | tee -a "$run_out/run.log"
  echo "Completed $label pilot/full run."
}

full_audit="$output_dir/pair_audit.csv"
if [[ ! -e "$full_audit" ]]; then
  run_vlm "serial pair baseline mec3" 1 1 "$pilot_pairs" "$output_dir/benchmark/pair_batch1_mec3"
  run_vlm "pair batch50 mec3" 1 50 "$pilot_pairs" "$output_dir/benchmark/pair_batch50_mec3"
  "$python_bin" hpc_vlm_10k/qwen3_padding_guard.py --check "$output_dir/benchmark"
  "$python_bin" hpc_vlm_10k/estimate_runtime.py \
    --batch1 "$output_dir/benchmark/pair_batch1_mec3/timing.json" \
    --batch-n "$output_dir/benchmark/pair_batch50_mec3/timing.json" \
    --pair-batch-size 50 --total-pairs 32500 --max-hours 158 \
    | tee "$output_dir/benchmark/runtime_projection.json"
fi

run_vlm "pair batch50 full" "$full_mec_batch" "$full_pair_batch" 32500 "$output_dir"

"$python_bin" main.py \
  --input "$output_dir/vlm_pairwise_votes.csv" \
  --output-dir "$output_dir" \
  --dimension visual_pollution_v8
