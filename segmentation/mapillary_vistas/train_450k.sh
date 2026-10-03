#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"
MMSEG_DIR="$REPO_DIR/segmentation/vendor/mmsegmentation"
PYTHON_BIN="/home/colin/miniconda3/envs/guvp-seg/bin/python"
CONFIG_FILE="$PROJECT_DIR/configs/deeplabv3plus_r50-d8_1xb2-450k_mapillary-v2-512x512.py"

export MAPILLARY_ROOT="${MAPILLARY_ROOT:-/home/colin/datasets/mapillary_vistas}"
export MAPILLARY_WORK_DIR="${MAPILLARY_WORK_DIR:-$REPO_DIR/output/deeplabv3plus_r50_mapillary_v2_450k}"
export PYTHONPATH="$PROJECT_DIR:$MMSEG_DIR${PYTHONPATH:+:$PYTHONPATH}"
# The trusted MMEngine checkpoint contains optimizer/message-hub Python
# objects; PyTorch >=2.6 otherwise attempts a weights-only load.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# On first launch use the completed 96k checkpoint.  Later launches continue
# from this run's newest checkpoint, unless an explicit source is supplied.
if [[ -n "${MAPILLARY_RESUME_CHECKPOINT:-}" ]]; then
  RESUME_CHECKPOINT="$MAPILLARY_RESUME_CHECKPOINT"
elif [[ -s "$MAPILLARY_WORK_DIR/last_checkpoint" ]]; then
  RESUME_CHECKPOINT="$(<"$MAPILLARY_WORK_DIR/last_checkpoint")"
else
  RESUME_CHECKPOINT="$REPO_DIR/output/deeplabv3plus_r50_mapillary_v2_80k/iter_96000.pth"
fi

if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "Resume checkpoint not found: $RESUME_CHECKPOINT" >&2
  exit 1
fi

cd "$MMSEG_DIR"
exec "$PYTHON_BIN" tools/train.py "$CONFIG_FILE" \
  --work-dir "$MAPILLARY_WORK_DIR" \
  --resume \
  --cfg-options \
  "load_from=$RESUME_CHECKPOINT" \
  "train_dataloader.dataset.data_root=$MAPILLARY_ROOT" \
  "val_dataloader.dataset.data_root=$MAPILLARY_ROOT" \
  "test_dataloader.dataset.data_root=$MAPILLARY_ROOT"
