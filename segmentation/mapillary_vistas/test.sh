#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CHECKPOINT [extra MMSeg test arguments...]" >&2
  exit 2
fi

CHECKPOINT="$1"
shift
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"
MMSEG_DIR="$REPO_DIR/segmentation/vendor/mmsegmentation"
PYTHON_BIN="/home/colin/miniconda3/envs/guvp-seg/bin/python"
CONFIG_FILE="$PROJECT_DIR/configs/deeplabv3plus_r50-d8_1xb2-80k_mapillary-v2-512x512.py"

export MAPILLARY_ROOT="${MAPILLARY_ROOT:-/home/colin/datasets/mapillary_vistas}"
export PYTHONPATH="$PROJECT_DIR:$MMSEG_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

cd "$MMSEG_DIR"
exec "$PYTHON_BIN" tools/test.py "$CONFIG_FILE" "$CHECKPOINT" \
  --cfg-options \
  "val_dataloader.dataset.data_root=$MAPILLARY_ROOT" \
  "test_dataloader.dataset.data_root=$MAPILLARY_ROOT" \
  "$@"
