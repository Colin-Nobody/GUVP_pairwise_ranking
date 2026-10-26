#!/usr/bin/env bash
#SBATCH --job-name=mapillary-30x200
#SBATCH --partition=Q10
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=1-00:00:00
#SBATCH --output=mapillary-30x200-%j.out
#SBATCH --export=ALL

set -euo pipefail

project_dir="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-/share/home/sgxcw3/GUVP_pairwise_ranking}}"
cd "$project_dir"

source "$project_dir/jobs/lib/q10_python.sh"
q10_load_config "$project_dir"
q10_resolve_python
python_bin="$PYTHON_BIN"

token_file="${MAPILLARY_TOKEN_FILE:-$HOME/.config/mapillary/access_token}"
if [[ ! -s "$token_file" ]]; then
  echo "Mapillary token file is missing or empty: $token_file" >&2
  exit 1
fi
MAPILLARY_ACCESS_TOKEN="$(<"$token_file")"
if [[ "$MAPILLARY_ACCESS_TOKEN" != MLY\|* ]]; then
  echo "Mapillary token file does not look like a Client Token (expected MLY|...)." >&2
  echo "Re-save the token by entering only the token at the hidden prompt." >&2
  exit 1
fi
export MAPILLARY_ACCESS_TOKEN

parquet_file="${GLOBAL_STREETSCAPES_PARQUET:-$project_dir/data/parquet/streetscapes.parquet}"
if [[ ! -f "$parquet_file" ]]; then
  cache_root="$HOME/.cache/huggingface/hub/datasets--NUS-UAL--global-streetscapes"
  if [[ -d "$cache_root" ]]; then
    parquet_file="$(find "$cache_root" -name streetscapes.parquet -type f -print -quit)"
  fi
fi
if [[ ! -f "$parquet_file" ]]; then
  echo "Global Streetscapes metadata is not cached; downloading its 1.98 GB combined Parquet from Hugging Face."
  parquet_file="$("$python_bin" - <<'PY'
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="NUS-UAL/global-streetscapes",
    filename="data/parquet/streetscapes.parquet",
    repo_type="dataset",
)
print(path)
PY
)"
fi
if [[ ! -f "$parquet_file" ]]; then
  echo "Global Streetscapes metadata download did not produce a file: $parquet_file" >&2
  exit 1
fi

manifest="$project_dir/data/mapillary_30city_200/images_manifest.csv"
image_dir="$project_dir/data/mapillary_30city_200/street_images"
download_log="$project_dir/data/mapillary_30city_200/download_log.csv"
mkdir -p "$(dirname "$manifest")" "$image_dir"

cities=(
  Amsterdam Barcelona Belgrade Berlin Brussels Budapest Copenhagen Edinburgh Helsinki Krakow
  Madrid Munich Oslo Paris Prague Rome Stockholm Tallinn Warsaw Zurich
  Athens Bucharest Hamburg Lisbon Lyon Milan Riga Sofia Vienna Vilnius
)

if [[ ! -s "$manifest" ]]; then
  "$python_bin" "$project_dir/prepare_global_streetscapes_sample.py" \
    --input "$parquet_file" \
    --output "$manifest" \
    --per-city 200 \
    --seed 20261022 \
    --grid-size-degrees 0.005 \
    --sources Mapillary \
    --cities "${cities[@]}"
else
  echo "Reusing frozen manifest: $manifest"
fi

# Validate the frozen manifest and one real image before starting the full run.
"$python_bin" - "$manifest" "$image_dir" <<'PY'
import os
import sys
from pathlib import Path

import pandas as pd
import requests

manifest_path, image_dir = map(Path, sys.argv[1:])
expected_cities = {
    "Amsterdam", "Barcelona", "Belgrade", "Berlin", "Brussels", "Budapest",
    "Copenhagen", "Edinburgh", "Helsinki", "Krakow", "Madrid", "Munich",
    "Oslo", "Paris", "Prague", "Rome", "Stockholm", "Tallinn", "Warsaw",
    "Zurich", "Athens", "Bucharest", "Hamburg", "Lisbon", "Lyon", "Milan",
    "Riga", "Sofia", "Vienna", "Vilnius",
}
df = pd.read_csv(manifest_path)
required = {"uuid", "source", "orig_id", "city"}
if required - set(df.columns):
    raise SystemExit(f"Manifest lacks columns: {sorted(required - set(df.columns))}")
counts = df.groupby("city").size().to_dict()
bad = {city: counts.get(city, 0) for city in expected_cities if counts.get(city, 0) != 200}
if len(df) != 6000 or df.uuid.nunique() != 6000 or set(df.city) != expected_cities or bad:
    raise SystemExit(f"Manifest audit failed: rows={len(df)}, unique_uuid={df.uuid.nunique()}, city_counts={bad}")
if set(df.source) != {"Mapillary"}:
    raise SystemExit(f"Expected Mapillary-only manifest, got sources={sorted(set(df.source))}")

token = os.environ["MAPILLARY_ACCESS_TOKEN"]
first = df.iloc[0]
image_id = str(int(first.orig_id))
session = requests.Session()
session.headers["Authorization"] = f"OAuth {token}"
metadata = session.get(
    f"https://graph.mapillary.com/{image_id}",
    params={"fields": "id,thumb_2048_url"}, timeout=45,
)
metadata.raise_for_status()
payload = metadata.json()
image_url = payload.get("thumb_2048_url")
if str(payload.get("id")) != image_id or not image_url:
    raise SystemExit("Mapillary preflight did not return the requested image and thumbnail URL.")

image_response = session.get(image_url, timeout=60)
image_response.raise_for_status()
image_bytes = image_response.content
if not image_bytes.startswith(b"\xff\xd8"):
    raise SystemExit("Mapillary preflight response was not a JPEG image.")
target = image_dir / f"{first.uuid}.jpg"
temporary = target.with_suffix(".part")
temporary.write_bytes(image_bytes)
temporary.replace(target)
print(f"Preflight passed: API image {image_id}; downloaded {len(image_bytes)} JPEG bytes.")
PY

"$python_bin" "$project_dir/download_global_streetscapes_images.py" \
  --manifest "$manifest" \
  --output-dir "$image_dir" \
  --log "$download_log" \
  --timeout 60 \
  --retries 5

echo "Mapillary 30-city download completed. Images: $image_dir"
echo "Manifest: $manifest"
echo "Download log: $download_log"
