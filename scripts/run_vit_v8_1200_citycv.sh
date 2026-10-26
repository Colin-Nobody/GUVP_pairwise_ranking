#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/colin/miniconda3/envs/guvp-vit/bin/python"
AUDIT_DIR="${PROJECT_DIR}/output/vit_vp_v8_1200_citycv"
PREFIX_DIR="${AUDIT_DIR}/prefix_after_block10"
OUTPUT_DIR="${AUDIT_DIR}/finetune_last2"

cd "${PROJECT_DIR}"
mkdir -p "${OUTPUT_DIR}"

exec "${PYTHON}" vit_vp/train_finetune.py \
  --audit-dir "${AUDIT_DIR}" \
  --prefix-dir "${PREFIX_DIR}" \
  --checkpoint "${PROJECT_DIR}/models/checkpoints/dinov3_vitb.safetensors" \
  --output-dir "${OUTPUT_DIR}" \
  --repeats 3 \
  --max-epochs 60 \
  --patience 20 \
  --device cuda \
  --require-cuda \
  --skip-ridge-comparison
