#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/colin/GUVP_pairwise_ranking"
PYTHON_BIN="/home/colin/miniconda3/envs/guvp-seg/bin/python"
MMSEG_DIR="$REPO_DIR/segmentation/vendor/mmsegmentation"
PROJECT_DIR="$REPO_DIR/segmentation/mapillary_vistas"
CONFIG_FILE="$PROJECT_DIR/configs/deeplabv3plus_r50-d8_1xb2-450k-mapillary-v2-baseline-resize.py"
CHECKPOINT="$REPO_DIR/output/deeplabv3plus_r50_mapillary_v2_baseline_resize_b1_450k/best_mIoU_iter_408000.pth"
MANIFEST="$REPO_DIR/output/segmentation_mapillary_v2_finetune_3e-5_regression_250/segmentation_manifest.csv"
OUTPUT_DIR="$REPO_DIR/output/segmentation_mapillary_v2_baseline_resize_b1_450k_regression_250"
SELECTION_CSV="$REPO_DIR/output/segmentation_mapillary_v2_finetune_3e-5_regression_250_baseline/random_sample_30_visuals/selected_30.csv"
SAMPLE_DIR="$OUTPUT_DIR/random_sample_30_visuals"

export PYTHONPATH="$PROJECT_DIR:$MMSEG_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

if [[ -f "$OUTPUT_DIR/elastic_net_ready.csv" ]]; then
  echo "Refusing to overwrite completed output directory: $OUTPUT_DIR" >&2
  exit 1
fi
if [[ ! -f "$MANIFEST" || ! -f "$CHECKPOINT" || ! -f "$SELECTION_CSV" ]]; then
  echo "Manifest, checkpoint, or fixed 30-image selection is missing" >&2
  exit 1
fi

if [[ -d "$OUTPUT_DIR" ]]; then
  echo "Partial output detected; rerunning all 250 images and replacing matching files."
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
  --output-dir "$OUTPUT_DIR" \
  --inference-mode "single-scale_whole-image_aspect-ratio-preserving_resize"
"$PYTHON_BIN" "$REPO_DIR/segmentation/create_random_segmentation_triptychs.py" \
  --run-dir "$OUTPUT_DIR" \
  --output-dir "$SAMPLE_DIR" \
  --manifest "$MANIFEST" \
  --selection-csv "$SELECTION_CSV" \
  --count 30
"$PYTHON_BIN" "$REPO_DIR/segmentation/build_random_sample_element_report.py" \
  --run-dir "$OUTPUT_DIR" \
  --sample-dir "$SAMPLE_DIR"

echo "Completed: $OUTPUT_DIR"
