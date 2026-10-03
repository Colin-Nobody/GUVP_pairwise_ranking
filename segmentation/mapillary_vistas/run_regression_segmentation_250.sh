#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"
PYTHON_BIN="/home/colin/miniconda3/envs/guvp-seg/bin/python"
MMSEG_DIR="$REPO_DIR/segmentation/vendor/mmsegmentation"
CONFIG_FILE="$PROJECT_DIR/configs/deeplabv3plus_r50-d8_1xb2-40k-mapillary-v2-finetune-3e-5.py"
CHECKPOINT="$REPO_DIR/output/deeplabv3plus_r50_mapillary_v2_finetune_3e-5_140k/best_mIoU_iter_136000.pth"
OUTPUT_DIR="$REPO_DIR/output/segmentation_mapillary_v2_finetune_3e-5_regression_250"
MANIFEST="$OUTPUT_DIR/segmentation_manifest.csv"

export PYTHONPATH="$PROJECT_DIR:$MMSEG_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
"$PYTHON_BIN" "$REPO_DIR/segmentation/prepare_regression_segmentation_manifest.py" \
  --output "$MANIFEST"
"$PYTHON_BIN" "$REPO_DIR/segmentation/run_audit.py" \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --expected-images 250 \
  --mode sliding \
  --config "$CONFIG_FILE" \
  --checkpoint "$CHECKPOINT"
"$PYTHON_BIN" "$REPO_DIR/segmentation/build_elastic_net_dataset.py" \
  --manifest "$MANIFEST" \
  --semantic-features "$OUTPUT_DIR/sliding_512_o25/semantic_features_raw.csv" \
  --run-config "$OUTPUT_DIR/run_config.json" \
  --output-dir "$OUTPUT_DIR"

echo "250-image regression segmentation complete: $OUTPUT_DIR"
