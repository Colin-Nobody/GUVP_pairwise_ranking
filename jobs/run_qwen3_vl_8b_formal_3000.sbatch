#!/usr/bin/env bash
#SBATCH --job-name=guvp-qwen3-8b-3000
#SBATCH --partition=Q10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=guvp-qwen3-8b-3000-%j.out
#SBATCH --export=ALL

set -euo pipefail

project_dir="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
cd "$project_dir"

source "$project_dir/jobs/lib/q10_python.sh"
q10_load_config "$project_dir"
q10_resolve_python

if [[ ! -x "$project_dir/scripts/run_qwen3_vl_8b_formal_and_score.sh" ]]; then
  echo "Qwen3-VL runner not found under PROJECT_DIR=$project_dir" >&2
  exit 1
fi

echo "Starting Qwen3-VL 8B formal 3,000-pair run on partition Q10"
echo "Project: $project_dir"
bash "$project_dir/scripts/run_qwen3_vl_8b_formal_and_score.sh"
