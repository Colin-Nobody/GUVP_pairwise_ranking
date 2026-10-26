#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  jobs/submit_q10.sh --check JOB.sbatch
  jobs/submit_q10.sh --submit JOB.sbatch

--check runs `sbatch --test-only`; it does not enqueue the job.
--submit validates first, then submits the job to Q10.
Configure resources and Python in jobs/q10.env (copy q10.env.example first).
EOF
}

if [[ $# -ne 2 ]]; then usage; exit 2; fi
mode="$1"
job_file="$2"
case "$mode" in --check|--submit) ;; *) usage; exit 2 ;; esac

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_file="${Q10_CONFIG:-$project_dir/jobs/q10.env}"
if [[ ! -f "$config_file" ]]; then
  echo "Missing Q10 configuration: $config_file" >&2
  echo "Copy jobs/q10.env.example to jobs/q10.env and fill it from the Q10 probe." >&2
  exit 1
fi
if [[ ! -f "$job_file" ]]; then
  echo "Slurm job file not found: $job_file" >&2
  exit 1
fi
if [[ "$job_file" != /* ]]; then job_file="$project_dir/$job_file"; fi

# The config is a local, gitignored shell file maintained by the user on HPC.
set -a
# shellcheck disable=SC1090
source "$config_file"
set +a
export PROJECT_DIR="$project_dir"

partition="${Q10_PARTITION:-Q10}"
if [[ "$partition" != "Q10" ]]; then
  echo "This account is authorized for Q10 only; refusing partition '$partition'." >&2
  exit 1
fi
gres="${Q10_GRES:-gpu:1}"
cpus="${Q10_CPUS:-8}"
memory="${Q10_MEM:-32G}"
walltime="${Q10_TIME:-1-00:00:00}"
args=(--partition="$partition" --gres="$gres" --cpus-per-task="$cpus" --mem="$memory" --time="$walltime" --chdir="$project_dir" --export=ALL)
if [[ -n "${Q10_ACCOUNT:-}" ]]; then args+=(--account="$Q10_ACCOUNT"); fi

echo "Validating Q10 job: $job_file"
echo "partition=$partition gres=$gres cpus=$cpus memory=$memory time=$walltime account=${Q10_ACCOUNT:-default}"
sbatch --test-only "${args[@]}" "$job_file"
if [[ "$mode" == "--submit" ]]; then
  echo "Submitting validated job to Q10..."
  sbatch "${args[@]}" "$job_file"
fi
