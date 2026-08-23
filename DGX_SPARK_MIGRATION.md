# DGX Spark migration and reproduction

This repository is a self-contained archive of the 2026-08-22/23 GUVP
experiment. It includes all 250 selected street-view JPEGs, manifests, 15,562
formal VLM decisions, Q-score/TrueSkill outputs, plots, and mapping outputs.
Model weights are intentionally not stored in Git; Hugging Face restores them.

## 1. Clone and verify the archived experiment

```bash
git clone git@github.com:Colin-Nobody/GUVP_pairwise_ranking.git
cd GUVP_pairwise_ranking
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_bundle.py
```

Expected audit counts are 250 manifest rows/JPEGs/scored images, five cities
with 50 images each, and 15,562 valid VLM comparisons.

## 2. Recompute Q-score and TrueSkill without rerunning the VLM

```bash
source .venv/bin/activate
bash scripts/reproduce_scores.sh
```

This writes a separate `output/vlm_formal_v2_reproduced/` directory so the
archived formal output is not overwritten.

## 3. Recommended VLM environment on DGX Spark

DGX Spark is ARM64/Grace Blackwell. NVIDIA ships Docker and the NVIDIA
Container Runtime, and recommends current NGC framework containers for its
optimized software stack. The helper pins the PyTorch 25.11 container used in
NVIDIA's DGX Spark PyTorch guide:

```bash
# If NGC asks for authentication, run `docker login nvcr.io` first.
bash scripts/run_dgx_spark_container.sh
```

Inside the container:

```bash
python -m pip install -r requirements-vlm.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
python scripts/verify_bundle.py
```

The mounted Hugging Face cache survives container exits. The first VLM run
downloads `Qwen/Qwen2.5-VL-3B-Instruct`; no model token is normally required
for this public model.

## 4. Reproduce the formal VLM experiment

The archived formal run used:

- model: `Qwen/Qwen2.5-VL-3B-Instruct`
- quantization: bitsandbytes 4-bit NF4, bfloat16 compute
- prompt version: `vp-overall-minimal-v2`
- per-image visual budget: 401,408 pixels (512 x 28 x 28)
- deterministic pair selection seed: 42
- 15,562 randomly selected unordered pairs out of 31,125 possible pairs
- output: all 15,562 responses were valid `A` or `B`

Use a new output path to preserve the archived decisions:

```bash
python vlm_pairwise.py \
  --images-dir data/street_images \
  --max-pairs 15562 \
  --max-pixels 401408 \
  --seed 42 \
  --progress-every 100 \
  --progress-file output/dgx_reproduction/progress.log \
  --output data/vlm_pairwise_dgx_reproduction.csv

python main.py \
  --input data/vlm_pairwise_dgx_reproduction.csv \
  --output-dir output/dgx_reproduction \
  --dimension visual_pollution
```

Deterministic decoding is used, but exact VLM answers can still change across
GPU architectures, CUDA/PyTorch/Transformers versions, or model revisions.
The archived CSV is therefore the authoritative record of the original run.

## 5. Qwen2.5-VL-7B formal run and cross-model audit

On 2026-08-23, the same 15,562 unordered pairs, prompt, visual-token budget,
and deterministic decoding settings were evaluated with
`Qwen/Qwen2.5-VL-7B-Instruct`. The resumed run added 15,477 decisions to 85
existing decisions and completed with all 15,562 responses valid. The full run
spanned 3 h 49 min (0.884 seconds per pair based on the first and last saved
timestamps).

The 7B artifacts are:

- `data/vlm_pairwise_7b_formal_v2.csv`: all 15,562 pairwise decisions
- `output/vlm_7b_formal_v2/`: scores, summaries, plots, and run logs
- `scripts/run_7b_formal_and_score.sh`: resumable inference and scoring helper

Run the reproducible cross-model audit with:

```bash
python scripts/compare_vlm_runs.py \
  --votes-a data/vlm_pairwise_formal_v2.csv \
  --scores-a output/vlm_formal_v2/perception_scores.csv \
  --label-a 3b \
  --votes-b data/vlm_pairwise_7b_formal_v2.csv \
  --scores-b output/vlm_7b_formal_v2/perception_scores.csv \
  --label-b 7b \
  --manifest data/images_manifest.csv \
  --output-dir output/vlm_3b_vs_7b
```

The audit records a substantial presentation-order warning. The 3B model
selected position A in 15,402/15,562 comparisons (98.97%); the 7B model selected
position A in 9,317/15,562 comparisons (59.87%). Their physical-winner agreement
was 60.69%, and their aggregated VP scores had Spearman rho 0.2498. The high
Q-score/TrueSkill correlations within each run do not remove this VLM position
bias. Treat these runs as model-behavior evidence rather than VP ground truth
until an A/B-swapped audit is completed.

## 6. Re-download source images if needed

The JPEGs are already committed. For a fresh download, Mapillary rows require
a local token; never commit it:

```bash
export MAPILLARY_ACCESS_TOKEN='set-locally'
python download_global_streetscapes_images.py \
  --manifest data/images_manifest.csv \
  --metadata-kv /path/to/metadata_kv.parquet \
  --output-dir data/street_images \
  --log data/image_download_log.csv
```

## Important paths

- `data/street_images/`: 250 archived JPEGs
- `data/images_manifest.csv`: coordinates, city, source, sampling metadata
- `data/vlm_pairwise_formal_v2.csv`: authoritative formal VLM decisions
- `output/vlm_formal_v2/`: scores, summary, and charts
- `data/vlm_pairwise_7b_formal_v2.csv`: 7B formal VLM decisions
- `output/vlm_7b_formal_v2/`: 7B scores, summary, charts, and logs
- `output/vlm_3b_vs_7b/`: reproducible cross-model audit
- `output/sample_location_maps/`: five city sample-location maps
- `vlm_pairwise.py`: Qwen VLM inference
- `main.py`: Q-score and TrueSkill aggregation

No access tokens, model caches, virtual environments, or map-tile caches are
tracked by Git.
