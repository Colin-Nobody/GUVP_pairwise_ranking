#!/usr/bin/env bash
#SBATCH --job-name=vit10k-final
#SBATCH --partition=Q10
#SBATCH --gres=gpu:l20:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from the project root}"
source jobs/lib/q10_python.sh
q10_load_config "$PWD"
q10_resolve_python
export PYTHONPATH="$PWD/output/vit_vp_10k_v8/deps${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" vit_vp/train_final_10k.py
