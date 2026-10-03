#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"
WORK_DIR="${MAPILLARY_WORK_DIR:-$REPO_DIR/output/deeplabv3plus_r50_mapillary_v2_80k}"

bash "$PROJECT_DIR/train.sh"

shopt -s nullglob
BEST_CHECKPOINTS=("$WORK_DIR"/best_mIoU_iter_*.pth)
if (( ${#BEST_CHECKPOINTS[@]} != 1 )); then
  echo "Expected exactly one best checkpoint in $WORK_DIR; found ${#BEST_CHECKPOINTS[@]}" >&2
  exit 1
fi

bash "$PROJECT_DIR/run_post_training_audit.sh" "${BEST_CHECKPOINTS[0]}"

