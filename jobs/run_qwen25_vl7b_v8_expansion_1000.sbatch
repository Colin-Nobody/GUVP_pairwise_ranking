#!/usr/bin/env bash
#SBATCH --job-name=guvp-v8-1000
#SBATCH --partition=Q10
#SBATCH --gres=gpu:l20:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=13:00:00
#SBATCH --output=guvp-v8-1000-%j.out
#SBATCH --export=ALL

set -euo pipefail

# Submit through jobs/submit_q10.sh so jobs/q10.env controls the Q10 resources.

# Slurm executes a staged copy of this file from /var/spool/slurm, so
# BASH_SOURCE is not a reliable way to locate the project. The submit helper
# exports PROJECT_DIR; direct sbatch from the project root uses SLURM_SUBMIT_DIR.
project_dir="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-/share/home/sgxcw3/GUVP_pairwise_ranking}}"
if [[ ! -f "$project_dir/jobs/lib/q10_python.sh" ]]; then
  echo "Could not find jobs/lib/q10_python.sh under PROJECT_DIR=$project_dir" >&2
  echo "Submit from the project root or set PROJECT_DIR to /share/home/sgxcw3/GUVP_pairwise_ranking." >&2
  exit 1
fi
data_dir="$project_dir/data/expansion_750"
source_image_dir="$data_dir/street_images_1000"
image_dir="$project_dir/output/vlm_expansion_1000_definition_v8/resolved_images"
pair_file="$data_dir/pairs_1000_13_per_image.csv"
prompt_file="$project_dir/prompts/vp_pairwise_definition_v8.txt"
output_dir="$project_dir/output/vlm_expansion_1000_definition_v8"
audit_file="$output_dir/pair_audit.csv"
votes_file="$output_dir/vlm_pairwise_votes.csv"

cd "$project_dir"

source "$project_dir/jobs/lib/q10_python.sh"
q10_load_config "$project_dir"
q10_resolve_python
python_bin="$PYTHON_BIN"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable not found: $python_bin" >&2
  echo "Set PYTHON_BIN to the Python executable in your guvp-vlm environment." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; this job needs a GPU node." >&2
  exit 1
fi
if [[ ! -d "$source_image_dir" || ! -f "$pair_file" || ! -f "$prompt_file" ]]; then
  echo "Missing V8 1000-image inputs. Expected:" >&2
  echo "  images: $source_image_dir" >&2
  echo "  pairs:  $pair_file" >&2
  echo "  prompt: $prompt_file" >&2
  exit 1
fi

mkdir -p "$output_dir"
exec 9>"$output_dir/run.lock"
if ! flock -n 9; then
  echo "Another V8 1000-image run holds $output_dir/run.lock" >&2
  exit 1
fi

# The tracked 1000-image directory contains absolute symlinks from the
# authoring machine. Rebuild run-local links using tracked files by basename.
"$python_bin" - "$source_image_dir" "$image_dir" \
  "$project_dir/data/street_images" "$data_dir/street_images" <<'PY'
import os
import sys
from pathlib import Path

source_dir, output_dir, *fallback_dirs = map(Path, sys.argv[1:])
image_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
entries = sorted(
    (p for p in source_dir.iterdir() if p.suffix.lower() in image_suffixes),
    key=lambda p: p.name,
)
if len(entries) != 1000:
    raise SystemExit(f"Expected 1000 image entries in {source_dir}, found {len(entries)}")

output_dir.mkdir(parents=True, exist_ok=True)
missing = []
for entry in entries:
    target = entry.resolve(strict=True) if entry.is_file() else None
    if target is None:
        for root in fallback_dirs:
            candidate = root / entry.name
            if candidate.is_file():
                target = candidate.resolve(strict=True)
                break
    if target is None:
        missing.append(entry.name)
        continue

    link = output_dir / entry.name
    if link.is_symlink():
        if link.resolve(strict=False) == target:
            continue
        raise SystemExit(f"Refusing to replace unexpected link: {link} -> {os.readlink(link)}")
    if link.exists():
        raise SystemExit(f"Refusing to replace existing non-link image: {link}")
    link.symlink_to(target)

if missing:
    raise SystemExit(f"Could not resolve {len(missing)} image(s), including: {', '.join(missing[:10])}")
staged = [p for p in output_dir.iterdir() if p.suffix.lower() in image_suffixes]
if len(staged) != 1000:
    raise SystemExit(f"Expected 1000 staged images, found {len(staged)} in {output_dir}")
print(f"Resolved {len(staged)} images into {output_dir}")
PY

if pgrep -af '[v]lm_pairwise.py' >/dev/null; then
  echo "Another VLM inference process is already running for this user; refusing to share the GPU." >&2
  pgrep -af '[v]lm_pairwise.py' >&2
  exit 1
fi

"$python_bin" scripts/preflight_formal_abba.py \
  --images-dir "$image_dir" \
  --pairs "$pair_file" \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --prompt-file "$prompt_file" \
  --prompt-version vp-definition-score-v8 \
  --score-direction lower_is_cleaner \
  --expected-images 1000 \
  --expected-pairs 6500 \
  --mec-samples 3

resume_args=()
if [[ -f "$audit_file" ]]; then
  resume_args=(--resume)
elif [[ -f "$votes_file" ]]; then
  echo "Found votes without an audit file; refusing to overwrite or mix results:" >&2
  echo "  $votes_file" >&2
  exit 1
fi

"$python_bin" vlm_pairwise.py \
  --images-dir "$image_dir" \
  --pairs "$pair_file" \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --prompt-file "$prompt_file" \
  --prompt-version vp-definition-score-v8 \
  --score-direction lower_is_cleaner \
  --max-pixels 401408 \
  --mec-samples 3 \
  --mec-batch-size 3 \
  --temperature 1.0 \
  --max-new-tokens 192 \
  --gpu-only \
  --progress-every 30 \
  --progress-file "$output_dir/progress.log" \
  --audit-output "$audit_file" \
  --output "$votes_file" \
  --timing-output "$output_dir/timing.json" \
  --seed 42 \
  "${resume_args[@]}" 2>&1 | tee -a "$output_dir/run.log"

"$python_bin" scripts/summarize_formal_abba.py \
  --audit "$audit_file" \
  --output-dir "$output_dir"

"$python_bin" main.py \
  --input "$votes_file" \
  --output-dir "$output_dir" \
  --dimension visual_pollution

echo "V8 1000-image pairwise run and scoring completed: $output_dir"
