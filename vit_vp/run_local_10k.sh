#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
audit_pid="${1:?Pass the running image-verification PID}"
while kill -0 "$audit_pid" 2>/dev/null; do sleep 10; done
test -s output/vit_vp_10k_v8/image_manifest.csv
test -s output/vit_vp_10k_v8/label_provenance.json
python -c 'import json; assert json.load(open("output/vit_vp_10k_v8/label_provenance.json"))["image_sha256_verified"] == 10000'
CONDA=/home/colin/miniconda3/bin/conda
"$CONDA" run --no-capture-output -n guvp-vit python -u vit_vp/cache_frozen_prefix.py --audit-dir output/vit_vp_10k_v8 --output-dir output/vit_vp_10k_v8/prefix --edges 512
"$CONDA" run --no-capture-output -n guvp-vit python -u vit_vp/train_final_10k.py
