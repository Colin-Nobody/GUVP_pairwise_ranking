#!/usr/bin/env bash
set -euo pipefail
project=/share/home/sgxcw3/GUVP_pairwise_ranking
model=/share/home/sgxcw3/qwen3-nf4
exec 9>"$model/qwen3_pipeline.lock"
flock -n 9 || { echo 'Pipeline already running'; exit 1; }
if [[ -e "$model/qwen3_submitted.jobid" ]]; then echo 'Job already submitted:'; cat "$model/qwen3_submitted.jobid"; exit 1; fi
cd "$model"
bash "$project/hpc_vlm_10k/fetch_qwen3_nf4_objects.sh"
cd "$project"
test ! -e output/vlm_qwen3_10k_v8_pairbatch50_mec3/pair_audit.csv
job=$(sbatch --parsable --time=7-00:00:00 hpc_vlm_10k/submit_qwen3_b50_5d.sbatch)
printf '%s\n' "$job" > "$model/qwen3_submitted.jobid"
printf 'Submitted GPU job: %s\n' "$job"
