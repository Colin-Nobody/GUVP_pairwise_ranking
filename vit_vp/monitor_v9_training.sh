#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_DIR="$PROJECT_ROOT/output/vit_vp_v9_holdout_1000_200"
PROCESS_PATTERN='[t]rain_v9_1000_holdout.py --device cuda'

while true; do
  clear
  date '+V9 ViT training monitor | %Y-%m-%d %H:%M:%S %Z'
  echo
  PIDS="$(pgrep -f "$PROCESS_PATTERN" || true)"
  if [[ -n "$PIDS" ]]; then
    ps -o pid,etime,%cpu,%mem,stat,args -p "$(echo "$PIDS" | paste -sd, -)"
  else
    echo 'Training process: not running'
  fi
  echo
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
  fi
  echo
  PARTIAL="$EXPERIMENT_DIR/internal_oof_predictions_partial.csv"
  if [[ -f "$PARTIAL" ]]; then
    python - "$PARTIAL" <<'PY'
import csv, sys
path = sys.argv[1]
with open(path, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
folds = {(r['repeat'], r['outer_fold']) for r in rows}
print(f'Internal city folds complete: {len(folds)}/15; OOF rows: {len(rows)}/3000')
PY
  else
    echo 'Internal city folds complete: 0/15 (first fold in progress or initializing)'
  fi
  if [[ -f "$EXPERIMENT_DIR/deployment/finetuned_last2_v9/model.json" ]]; then
    echo 'Final 1,000-image ensemble: complete'
  else
    echo 'Final 1,000-image ensemble: pending'
  fi
  sleep 5
done
