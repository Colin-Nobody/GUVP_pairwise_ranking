#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

python main.py \
  --input data/vlm_pairwise_formal_v2.csv \
  --output-dir output/vlm_formal_v2_reproduced \
  --dimension visual_pollution

python scripts/verify_bundle.py
