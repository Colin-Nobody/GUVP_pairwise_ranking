#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${PROJECT_DIR}/output/vit_vp_v8_1200_citycv/finetune_last2"
PID_FILE="${OUT}/run.pid"

if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  if kill -0 "${PID}" 2>/dev/null; then
    printf 'Training process: RUNNING (PID %s)\n' "${PID}"
  else
    printf 'Training process: not active (PID %s)\n' "${PID}"
  fi
else
  printf 'Training process: PID file not found\n'
fi

if [[ -f "${OUT}/inner_candidate_scores.csv" ]]; then
  CANDIDATES="$(( $(wc -l < "${OUT}/inner_candidate_scores.csv") - 1 ))"
else
  CANDIDATES=0
fi
if [[ -f "${OUT}/outer_predictions_partial.csv" ]]; then
  OOF_ROWS="$(( $(wc -l < "${OUT}/outer_predictions_partial.csv") - 1 ))"
else
  OOF_ROWS=0
fi
if [[ -f "${OUT}/inner_selected_configs_partial.csv" ]]; then
  SELECTED="$(( $(wc -l < "${OUT}/inner_selected_configs_partial.csv") - 1 ))"
else
  SELECTED=0
fi
printf 'Inner fits: %s/360 | outer folds selected: %s/15 | OOF rows: %s/3600\n' \
  "${CANDIDATES}" "$(( SELECTED / 3 ))" "${OOF_ROWS}"

if [[ -f "${OUT}/inner_candidate_scores.csv" ]]; then
  printf 'Latest inner fit: '
  tail -n 1 "${OUT}/inner_candidate_scores.csv"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
fi
