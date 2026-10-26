#!/usr/bin/env bash
set -uo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
report_dir="$project_dir/output/hpc_q10"
mkdir -p "$report_dir"
report_file="$report_dir/q10_probe_$(date +%Y%m%d_%H%M%S).txt"

{
  echo "Q10 cluster probe"
  echo "timestamp=$(date --iso-8601=seconds 2>/dev/null || date)"
  echo "hostname=$(hostname)"
  echo "user=$(id -un)"
  echo "home=$HOME"
  echo "working_directory=$PWD"
  echo
  echo "=== Slurm version ==="
  scontrol --version 2>&1 || true
  echo
  echo "=== Q10 partition summary ==="
  sinfo -h -p Q10 -o '%P|%a|%l|%D|%t|%C|%m|%G' 2>&1 || true
  echo
  echo "=== Q10 node resources ==="
  sinfo -N -h -p Q10 -o '%N|%T|%c|%m|%G|%f' 2>&1 || true
  echo
  echo "=== Q10 partition configuration ==="
  scontrol show partition Q10 2>&1 || true
  echo
  echo "=== Current user Slurm associations ==="
  sacctmgr -nP show assoc where user="$(id -un)" format=User,Account,Partition,QOS 2>&1 || true
  echo
  echo "=== Relevant software modules ==="
  if ! type module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
      if [[ -r "$init" ]]; then
        # shellcheck disable=SC1090
        source "$init"
        break
      fi
    done
  fi
  if type module >/dev/null 2>&1; then
    module avail 2>&1 | grep -Ei 'anaconda|conda|python|cuda|pytorch' || true
  else
    echo "module command unavailable on this shell"
  fi
  echo
  echo "=== Resource limits ==="
  ulimit -a 2>&1 || true
} 2>&1 | tee "$report_file"

echo
echo "Probe saved to: $report_file"
echo "Review it before editing jobs/q10.env and running sbatch --test-only."
