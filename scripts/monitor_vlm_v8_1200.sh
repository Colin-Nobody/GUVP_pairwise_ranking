#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$project_dir/output/vlm_v8_1200_gpuonly"
audit_file="$output_dir/pair_audit.csv"
votes_file="$output_dir/vlm_pairwise_votes.csv"
progress_file="$output_dir/progress.log"
total_pairs=7800
interval="${1:-15}"

if ! [[ "$interval" =~ ^[0-9]+$ ]] || (( interval < 1 )); then
  echo "Usage: bash scripts/monitor_vlm_v8_1200.sh [refresh_seconds]" >&2
  exit 2
fi

while true; do
  clear
  printf 'VLM V8 1,200-image comparison | %s\n\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"

  if [[ -f "$audit_file" ]]; then
    python3 - "$audit_file" "$votes_file" "$total_pairs" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path

audit_path, votes_path, total_text = sys.argv[1:]
total = int(total_text)
audit = Path(audit_path)
with audit.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
statuses = Counter(row.get("status", "unknown") for row in rows)
votes = 0
votes_file = Path(votes_path)
if votes_file.exists():
    with votes_file.open(newline="", encoding="utf-8") as handle:
        votes = sum(1 for _ in csv.DictReader(handle))
done = len(rows)
print(f"Completed pairs: {done:,} / {total:,} ({done / total:.1%})")
print(f"Valid votes:     {votes:,}")
print("Pair statuses:   " + (", ".join(f"{k}={v}" for k, v in sorted(statuses.items())) or "none yet"))
PY
  else
    echo "Audit file not created yet."
  fi

  printf '\nGPU:\n'
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
  else
    echo "nvidia-smi unavailable"
  fi

  printf '\nInference process:\n'
  if pgrep -af '[v]lm_pairwise.py' >/dev/null; then
    pgrep -af '[v]lm_pairwise.py' | sed -E 's#^([0-9]+) .*#PID \1: running#'
  else
    echo "No vlm_pairwise.py process found"
  fi

  printf '\nLatest progress entries:\n'
  if [[ -f "$progress_file" ]]; then
    tail -n 3 "$progress_file"
  else
    echo "Progress log not created yet"
  fi

  printf '\nRefreshing every %s seconds; press Ctrl+C to stop.\n' "$interval"
  sleep "$interval"
done
