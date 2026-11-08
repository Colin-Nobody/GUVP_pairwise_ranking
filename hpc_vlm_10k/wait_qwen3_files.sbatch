#!/usr/bin/env bash
#SBATCH --job-name=qwen3-files
#SBATCH --partition=Q10
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --output=qwen3-files-%j.out
set -euo pipefail
cd /share/home/sgxcw3/qwen3-nf4/qwen3_vl_8b_nf4_export
for attempt in $(seq 1 340); do
  if [[ -f model-00001-of-00006.safetensors && -f model-00002-of-00006.safetensors && -f model-00003-of-00006.safetensors && -f model-00004-of-00006.safetensors && -f model-00005-of-00006.safetensors && -f model-00006-of-00006.safetensors ]]; then
    sha256sum -c <<'HASHES'
df479c840799ef1f12808c770b4a4a06b9348736c502baf4ba3f06744e8db636  model-00001-of-00006.safetensors
6f235a5fa0e37abe229bbee24ca4a8c47d4a33a2831339177db5c0a32cab0bd2  model-00002-of-00006.safetensors
e9778b16f59a254727526c5363f8e706c245fae2c6f6b47c26ac51f7c598b7a9  model-00003-of-00006.safetensors
ca85ccfe25fbb0c9d5090d1fcbdc83b619c03790ffd59cc52ce1c42673fbd121  model-00004-of-00006.safetensors
3884b78112eec9beed98e1608b321b6a1a51118ba359c5ec616d0f4b6f9956d1  model-00005-of-00006.safetensors
b68d833f45023e36f2185699d78f721791a6881f3f3517205a3c54b9ef34767e  model-00006-of-00006.safetensors
HASHES
    echo 'All six model shards verified.'
    exit 0
  fi
  echo "Waiting for model shards: check $attempt"
  sleep 120
done
echo 'Timed out waiting for model download.' >&2
exit 1
