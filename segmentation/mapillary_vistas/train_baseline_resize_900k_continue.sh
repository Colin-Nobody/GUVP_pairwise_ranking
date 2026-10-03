#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"
MMSEG_DIR="$REPO_DIR/segmentation/vendor/mmsegmentation"
PYTHON_BIN="/home/colin/miniconda3/envs/guvp-seg/bin/python"
CONFIG_FILE="$PROJECT_DIR/configs/deeplabv3plus_r50-d8_1xb2-900k-mapillary-v2-baseline-resize-continue.py"

export MAPILLARY_ROOT="${MAPILLARY_ROOT:-/home/colin/datasets/mapillary_vistas}"
export MAPILLARY_WORK_DIR="${MAPILLARY_WORK_DIR:-$REPO_DIR/output/deeplabv3plus_r50_mapillary_v2_baseline_resize_b1_450k}"
export PYTHONPATH="$PROJECT_DIR:$MMSEG_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [[ ! -f "$MAPILLARY_WORK_DIR/iter_450000.pth" ]]; then
  echo "Missing strict-resume checkpoint: $MAPILLARY_WORK_DIR/iter_450000.pth" >&2
  exit 1
fi

cd "$MMSEG_DIR"
exec "$PYTHON_BIN" tools/train.py "$CONFIG_FILE" \
  --work-dir "$MAPILLARY_WORK_DIR" \
  --resume \
  --cfg-options \
  "train_dataloader.dataset.data_root=$MAPILLARY_ROOT" \
  "val_dataloader.dataset.data_root=$MAPILLARY_ROOT" \
  "test_dataloader.dataset.data_root=$MAPILLARY_ROOT"
