#!/usr/bin/env bash
#SBATCH -J e4b-v8-5k
#SBATCH -p Q10
#SBATCH --gres=gpu:l20:1
#SBATCH -t 7-00:00:00
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o e4b-v8-5k-%j.out
set -euo pipefail
cd "$HOME/GUVP_pairwise_ranking"
py="$HOME/miniforge3/envs/guvp-vlm/bin/python"
weights="$HOME/gemma-4-E4B-it-qat-q4_0-gguf"
data=data/vlm_hpc_10k
out="output/e4b_v8_5k_${SLURM_JOB_ID:?}"
mkdir -p "$out"
"$py" gemma_e4b.py --help > "$out/adapter-help.txt"
grep -q -- '--pair-batch-size' "$out/adapter-help.txt"
nvidia-smi
export GEMMA_PORT=18087
LD_LIBRARY_PATH="$HOME/miniforge3/envs/guvp-gemma124/lib:${LD_LIBRARY_PATH:-}" \
"$HOME/software/llama.cpp/build-gcc11/bin/llama-server" \
  --model "$weights/gemma-4-E4B_q4_0-it.gguf" --mmproj "$weights/gemma-4-E4B-it-mmproj.gguf" \
  --n-gpu-layers 999 --host 127.0.0.1 --port "$GEMMA_PORT" --parallel 7 --ctx-size 28672 \
  > "$out/server.log" 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true' EXIT
for ((i=0; i<180; i++)); do
  if curl -fsS "http://127.0.0.1:$GEMMA_PORT/health" >/dev/null 2>&1; then break; fi
  kill -0 "$pid" || { tail -40 "$out/server.log"; exit 1; }
  sleep 2
done
curl -fsS "http://127.0.0.1:$GEMMA_PORT/health"
printf '\nChecking the previously failing pair before the 5k\n'
"$py" -u - "$data/images" <<'PY'
from pathlib import Path
import sys
from types import SimpleNamespace

from gemma_e4b import complete, image_part, pipeline

images_dir = Path(sys.argv[1])
a = images_dir / "14ba6354-37ec-56e1-89bc-9113e96f2c74.jpg"
b = images_dir / "903a12fa-5487-59eb-be57-df761c8d3311.jpg"
args = SimpleNamespace(
    prompt=Path("prompts/vp_pairwise_definition_v8.txt").read_text(encoding="utf-8").strip(),
    temperature=1,
    max_new_tokens=2048,
)
for seed, paths in enumerate(((a, b), (b, a)), start=42):
    answer = complete([image_part(path) for path in paths], paths, args, seed)
    score = pipeline.parse_evaluation(answer)
    if score is None:
        raise SystemExit(f"Gemma preflight produced no parseable V8 scores: {answer[:400]!r}")
    print(f"Gemma preflight seed {seed}: parsed scores {score}")
PY
for batch in 7; do
  dest="$out/batch_$batch"
  mkdir -p "$dest"
  "$py" -u gemma_e4b.py --images-dir "$data/images" \
    --pairs "$data/../e4b5k/p.csv" --limit-pairs 32500 \
    --prompt-file prompts/vp_pairwise_definition_v8.txt \
    --prompt-version vp-definition-score-v8 --score-direction lower_is_cleaner \
    --model google/gemma-4-E4B-it-qat-q4_0-gguf \
    --mec-samples 3 --mec-batch-size 1 --pair-batch-size "$batch" \
    --seed 42 --temperature 1 --max-new-tokens 2048 \
    --audit-output "$dest/audit.csv" --output "$dest/votes.csv" \
    --timing-output "$dest/timing.json" --progress-every 10
done
