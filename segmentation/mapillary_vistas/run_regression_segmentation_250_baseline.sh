#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/colin/GUVP_pairwise_ranking"
PYTHON_BIN="/home/colin/miniconda3/envs/guvp-seg/bin/python"
MMSEG_DIR="$REPO_DIR/segmentation/vendor/mmsegmentation"
PROJECT_DIR="$REPO_DIR/segmentation/mapillary_vistas"
CONFIG_FILE="$PROJECT_DIR/configs/deeplabv3plus_r50-d8_1xb2-40k-mapillary-v2-finetune-3e-5.py"
CHECKPOINT="$REPO_DIR/output/deeplabv3plus_r50_mapillary_v2_finetune_3e-5_140k/best_mIoU_iter_136000.pth"
SOURCE_DIR="$REPO_DIR/output/segmentation_mapillary_v2_finetune_3e-5_regression_250"
OUTPUT_DIR="$REPO_DIR/output/segmentation_mapillary_v2_finetune_3e-5_regression_250_baseline"
MANIFEST="$SOURCE_DIR/segmentation_manifest.csv"
WAIT_SERVICE="guvp-mapillary-retained-train-miou.service"

export PYTHONPATH="$PROJECT_DIR:$MMSEG_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

while systemctl --user is-active --quiet "$WAIT_SERVICE"; do
  echo "[$(date '+%F %T')] Waiting for train-mIoU checkpoint evaluation to release the GPU..."
  sleep 30
done

if [[ ! -f "$MANIFEST" || ! -f "$CHECKPOINT" ]]; then
  echo "Manifest or checkpoint is missing" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
"$PYTHON_BIN" "$REPO_DIR/segmentation/run_audit.py" \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --expected-images 250 \
  --mode baseline \
  --config "$CONFIG_FILE" \
  --checkpoint "$CHECKPOINT"
"$PYTHON_BIN" "$REPO_DIR/segmentation/build_elastic_net_dataset.py" \
  --manifest "$MANIFEST" \
  --semantic-features "$OUTPUT_DIR/baseline/semantic_features_raw.csv" \
  --run-config "$OUTPUT_DIR/run_config.json" \
  --output-dir "$OUTPUT_DIR"

echo "250-image baseline regression segmentation complete: $OUTPUT_DIR"
