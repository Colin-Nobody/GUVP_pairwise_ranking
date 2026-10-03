#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 CHECKPOINT" >&2
  exit 2
fi

CHECKPOINT="$1"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$PROJECT_DIR/../.." && pwd)"
PYTHON_BIN="/home/colin/miniconda3/envs/guvp-seg/bin/python"
MMSEG_DIR="$REPO_DIR/segmentation/vendor/mmsegmentation"
CONFIG_FILE="$PROJECT_DIR/configs/deeplabv3plus_r50-d8_1xb2-80k_mapillary-v2-512x512.py"
AUDIT_DIR="$REPO_DIR/output/segmentation_mapillary_v2_audit"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_DIR:$MMSEG_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
mkdir -p "$AUDIT_DIR"

"$PYTHON_BIN" "$REPO_DIR/segmentation/prepare_audit_manifest.py" \
  --output "$AUDIT_DIR/audit_manifest.csv"

"$PYTHON_BIN" "$REPO_DIR/segmentation/run_audit.py" \
  --manifest "$AUDIT_DIR/audit_manifest.csv" \
  --output-dir "$AUDIT_DIR" \
  --mode both \
  --config "$CONFIG_FILE" \
  --checkpoint "$CHECKPOINT"

"$PYTHON_BIN" "$REPO_DIR/segmentation/build_audit_gallery.py" \
  --audit-dir "$AUDIT_DIR"

"$PYTHON_BIN" "$PROJECT_DIR/summarize_audit.py" \
  --audit-dir "$AUDIT_DIR"

echo "Audit complete: $AUDIT_DIR"
