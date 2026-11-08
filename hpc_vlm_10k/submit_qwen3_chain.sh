#!/usr/bin/env bash
set -euo pipefail
cd /share/home/sgxcw3/GUVP_pairwise_ranking
test -f hpc_vlm_10k/wait_qwen3_files.sbatch
test -f hpc_vlm_10k/submit_qwen3_b50_5d.sbatch
test -f hpc_vlm_10k/run_qwen3_b50_5d.sh
test ! -e output/vlm_qwen3_10k_v8_pairbatch50_mec3/pair_audit.csv
sbatch --test-only hpc_vlm_10k/wait_qwen3_files.sbatch
sbatch --test-only --time=7-00:00:00 hpc_vlm_10k/submit_qwen3_b50_5d.sbatch
files_job=$(sbatch --parsable hpc_vlm_10k/wait_qwen3_files.sbatch)
echo "Model verification job: $files_job"
full_job=$(sbatch --parsable --dependency="afterok:${files_job}" --time=7-00:00:00 hpc_vlm_10k/submit_qwen3_b50_5d.sbatch)
echo "Seven-day Qwen3 GPU job: $full_job; dependency: afterok:$files_job"
