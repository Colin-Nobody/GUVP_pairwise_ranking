#!/usr/bin/env bash
#SBATCH --job-name=qwen25-5k
#SBATCH --partition=Q10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l20:1
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=qwen25-5k-%j.out
#SBATCH --export=ALL

set -euo pipefail
project_root="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-/share/home/sgxcw3/GUVP_pairwise_ranking}}"
source_root="$project_root/oecd_europe_50city_10000"
cd "$project_root"

source "$project_root/jobs/lib/q10_python.sh"
q10_load_config "$project_root"
q10_resolve_python
python_bin="$PYTHON_BIN"
export PROJECT_ROOT="$project_root"

echo "Job: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
bash hpc_vlm_10k/q25.sh
