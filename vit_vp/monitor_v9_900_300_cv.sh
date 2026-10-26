#!/usr/bin/env bash
set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_DIR="$PROJECT_ROOT/output/vit_vp_v9_holdout_900_300_cv"
PROCESS_PATTERN='[t]rain_v9_900_300_cv.py'

while true; do
  clear
  date '+V9 900/300 CV training monitor | %Y-%m-%d %H:%M:%S %Z'
  echo
  PIDS="$(pgrep -f "$PROCESS_PATTERN" || true)"
  if [[ -n "$PIDS" ]]; then
    ps -o pid,etime,%cpu,%mem,stat,args -p "$(echo "$PIDS" | paste -sd, -)"
  else
    echo 'Training process: not running'
  fi
  echo
  nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
  echo
  PARTIAL="$EXPERIMENT_DIR/internal_oof_predictions_partial.csv"
  if [[ -f "$PARTIAL" ]]; then
    python - "$PARTIAL" <<'PY'
import csv, sys
with open(sys.argv[1], newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
folds = {(r['repeat'], r['outer_fold']) for r in rows}
print(f'Internal city folds complete: {len(folds)}/15; OOF rows: {len(rows)}/2700')
PY
  else
    echo 'Internal city folds complete: 0/15'
  fi
  if [[ -f "$EXPERIMENT_DIR/deployment/finetuned_last2_v9/model.json" ]]; then
    echo 'Final 900-image ensemble: complete'
  else
    echo 'Final 900-image ensemble: pending'
  fi
  if [[ -f "$EXPERIMENT_DIR/test_metrics.json" ]]; then
    echo '300-image test evaluation: complete'
  else
    echo '300-image test evaluation: pending'
  fi
  sleep 5
done
