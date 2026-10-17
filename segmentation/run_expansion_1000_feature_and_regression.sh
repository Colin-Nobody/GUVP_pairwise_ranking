#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_SEG="/home/colin/miniconda3/envs/guvp-seg/bin/python"
PROJECT_DIR="$REPO_DIR/segmentation/mapillary_vistas"
MMSEG_DIR="$REPO_DIR/segmentation/vendor/mmsegmentation"
CONFIG_FILE="$PROJECT_DIR/configs/deeplabv3plus_r50-d8_1xb2-40k-mapillary-v2-finetune-3e-5.py"
CHECKPOINT="$REPO_DIR/output/deeplabv3plus_r50_mapillary_v2_finetune_3e-5_140k/best_mIoU_iter_136000.pth"
OUTPUT_DIR="$REPO_DIR/output/segmentation_expansion_1000"
REGRESSION_DIR="$REPO_DIR/output/vp_pollution_elastic_net_expansion_1000"

export PYTHONPATH="$PROJECT_DIR:$MMSEG_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint is unavailable: $CHECKPOINT" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR/baseline/semantic_features_raw.csv" || -e "$OUTPUT_DIR/elastic_net_ready.csv" ]]; then
  echo "Refusing to overwrite existing 1000-image features in $OUTPUT_DIR" >&2
  exit 1
fi
if [[ -e "$REGRESSION_DIR/REPORT.md" ]]; then
  echo "Refusing to overwrite completed regression in $REGRESSION_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
"$PYTHON_SEG" "$REPO_DIR/segmentation/prepare_expansion_1000_segmentation_manifest.py" \
  --output "$OUTPUT_DIR/segmentation_manifest.csv"
"$PYTHON_SEG" "$REPO_DIR/segmentation/run_audit.py" \
  --manifest "$OUTPUT_DIR/segmentation_manifest.csv" \
  --output-dir "$OUTPUT_DIR" \
  --mode baseline \
  --config "$CONFIG_FILE" \
  --checkpoint "$CHECKPOINT" \
  --expected-images 1000 2>&1 | tee "$OUTPUT_DIR/run.log"
"$PYTHON_SEG" "$REPO_DIR/segmentation/build_elastic_net_dataset.py" \
  --manifest "$OUTPUT_DIR/segmentation_manifest.csv" \
  --semantic-features "$OUTPUT_DIR/baseline/semantic_features_raw.csv" \
  --run-config "$OUTPUT_DIR/run_config.json" \
  --output-dir "$OUTPUT_DIR" \
  --inference-mode "single-scale whole-image aspect-ratio-preserving baseline; CPU nearest-neighbour mask restore"

"$PYTHON_SEG" "$REPO_DIR/scripts/fit_vp_pollution_elastic_net.py" \
  --run-dir "$OUTPUT_DIR" \
  --output-dir "$REGRESSION_DIR" \
  --outer-repeats 10 \
  --bootstrap 250 2>&1 | tee "$REGRESSION_DIR.run.log"

echo "Completed 1000-image segmentation features and secondary score regression."
