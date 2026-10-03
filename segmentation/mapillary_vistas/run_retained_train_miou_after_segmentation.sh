#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/colin/GUVP_pairwise_ranking"
SEGMENTATION_SERVICE="guvp-mapillary-regression-segmentation-250.service"
PYTHON_BIN="/home/colin/miniconda3/envs/guvp-seg/bin/python"
OUTPUT_CSV="$REPO_DIR/output/train_miou_eval_2000/retained_checkpoint_train_miou.csv"

while systemctl --user is-active --quiet "$SEGMENTATION_SERVICE"; do
  echo "[$(date '+%F %T')] Waiting for 250-image segmentation to release the GPU..."
  sleep 30
done

cd "$REPO_DIR"
"$PYTHON_BIN" segmentation/mapillary_vistas/evaluate_retained_train_miou.py \
  --repo "$REPO_DIR" \
  --output "$OUTPUT_CSV"

"$PYTHON_BIN" segmentation/mapillary_vistas/plot_train_validation_miou.py \
  --main-validation "$REPO_DIR/output/deeplabv3plus_r50_mapillary_v2_450k/validation_miou_complete_8k_450k.csv" \
  --finetune-validation "$REPO_DIR/output/deeplabv3plus_r50_mapillary_v2_finetune_3e-5_140k/validation_miou_complete_2k_140k.csv" \
  --train-eval "$OUTPUT_CSV" \
  --output "$REPO_DIR/output/train_miou_eval_2000/train_vs_validation_miou_retained_checkpoints.png"
