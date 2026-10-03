#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"
MMSEG_DIR="$REPO_DIR/segmentation/vendor/mmsegmentation"
PYTHON_BIN="/home/colin/miniconda3/envs/guvp-seg/bin/python"
CONFIG_FILE="$PROJECT_DIR/configs/deeplabv3plus_r50-d8_1xb2-80k_mapillary-v2-512x512.py"

export MAPILLARY_ROOT="${MAPILLARY_ROOT:-/home/colin/datasets/mapillary_vistas}"
export MAPILLARY_WORK_DIR="${MAPILLARY_WORK_DIR:-$REPO_DIR/output/deeplabv3plus_r50_mapillary_v2_80k}"
export PYTHONPATH="$PROJECT_DIR:$MMSEG_DIR${PYTHONPATH:+:$PYTHONPATH}"
# PyTorch >=2.6 defaults torch.load to weights_only=True. MMEngine 0.10.x
# checkpoints include trusted local optimizer/message-hub Python objects that
# are required for exact resume.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

RESUME_CFG=()
for arg in "$@"; do
  if [[ "$arg" == "--resume" ]]; then
    # MMEngine otherwise interprets the ADE20K initializer as the checkpoint
    # to resume, which has no optimizer/message-hub state.
    RESUME_CFG+=("load_from=None")
  fi
done

cd "$MMSEG_DIR"
exec "$PYTHON_BIN" tools/train.py "$CONFIG_FILE" \
  --work-dir "$MAPILLARY_WORK_DIR" \
  "$@" \
  --cfg-options \
  "train_dataloader.dataset.data_root=$MAPILLARY_ROOT" \
  "val_dataloader.dataset.data_root=$MAPILLARY_ROOT" \
  "test_dataloader.dataset.data_root=$MAPILLARY_ROOT" \
  "${RESUME_CFG[@]}"
