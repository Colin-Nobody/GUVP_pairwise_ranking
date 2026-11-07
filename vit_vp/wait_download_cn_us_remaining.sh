#!/usr/bin/env bash
set -euo pipefail
survey_pid="${1:?survey PID required}"
download_pid="${2:?download PID required}"
cd "$(dirname "$0")/.."
while kill -0 "$survey_pid" 2>/dev/null || kill -0 "$download_pid" 2>/dev/null; do sleep 15; done
for city in Guangzhou Shenzhen Chengdu Wuhan Hangzhou Nanjing; do
  test -f "output/mapillary_cn_other_survey_12km/${city}_candidates.csv" || { echo "Missing 12 km survey for $city; stopped"; exit 1; }
done
exec /home/colin/miniconda3/bin/conda run --no-capture-output -n guvp-vit python -u vit_vp/download_cn_us_sample.py --cities Guangzhou Shenzhen Chengdu Wuhan Hangzhou Nanjing --workers 6
