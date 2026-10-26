#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$project_dir/output/vlm_v9_1200_gpuonly_optimized"
audit_file="$output_dir/pair_audit.csv"
votes_file="$output_dir/vlm_pairwise_votes.csv"
total_pairs=7800
interval="${1:-15}"

if ! [[ "$interval" =~ ^[0-9]+$ ]] || (( interval < 1 )); then
  echo "Usage: bash scripts/monitor_vlm_v9_1200.sh [refresh_seconds]" >&2
  exit 2
fi

while true; do
  printf '\033[2J\033[H'
  printf 'VLM V9 1,200-image run monitor  |  %s\n\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"

  if [[ -f "$audit_file" ]]; then
    python3 - "$audit_file" "$votes_file" "$total_pairs" <<'PY'
import csv
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

audit_path, votes_path, total_text = sys.argv[1:]
total = int(total_text)
rows = list(csv.DictReader(Path(audit_path).open(newline="", encoding="utf-8")))
counts = Counter(row.get("status", "unknown") for row in rows)
completed = len(rows)
votes = max(0, sum(1 for _ in Path(votes_path).open(encoding="utf-8")) - 1) if Path(votes_path).exists() else 0
print(f"Completed pairs: {completed:,} / {total:,} ({completed / total:.2%})")
print(f"Scored votes:    {votes:,}")
print("Pair statuses:   " + ", ".join(f"{name}={count}" for name, count in sorted(counts.items())))

timestamps = []
for row in rows[-31:]:
    try:
        timestamps.append(datetime.fromisoformat(row["timestamp_utc"]))
    except (KeyError, ValueError):
        pass
if len(timestamps) >= 2:
    elapsed = (timestamps[-1] - timestamps[0]).total_seconds()
    if elapsed > 0:
        rate = (len(timestamps) - 1) / elapsed
        remaining = max(0, total - completed) / rate
        hours, remainder = divmod(int(remaining), 3600)
        minutes = remainder // 60
        print(f"Recent rate:     {rate * 60:.2f} pairs/min (last {len(timestamps)} records)")
        print(f"Estimated left:  {hours} h {minutes} min (rough estimate)")
elif completed < total:
    print("Estimated left:  waiting for more completed-pair timestamps")
PY
  else
    echo "Audit file not created yet: $audit_file"
  fi

  printf '\nGPU:\n'
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
  else
    echo "nvidia-smi unavailable"
  fi

  printf '\nInference process:\n'
  if pgrep -af '[v]lm_pairwise.py' >/dev/null; then
    pgrep -af '[v]lm_pairwise.py' | sed -E 's#^([0-9]+) .*vlm_pairwise.py.*#PID \1: running#'
  else
    echo "No vlm_pairwise.py process found"
  fi

  printf '\nLatest progress log entries:\n'
  if [[ -f "$output_dir/progress.log" ]]; then
    tail -n 3 "$output_dir/progress.log"
  else
    echo "Progress log not created yet"
  fi

  printf '\nRefreshing every %s seconds; press Ctrl+C to exit monitor.\n' "$interval"
  sleep "$interval"
done
