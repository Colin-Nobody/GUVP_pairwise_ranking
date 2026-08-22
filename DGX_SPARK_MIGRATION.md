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

## 5. Re-download source images if needed

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
- `output/sample_location_maps/`: five city sample-location maps
- `vlm_pairwise.py`: Qwen VLM inference
- `main.py`: Q-score and TrueSkill aggregation

No access tokens, model caches, virtual environments, or map-tile caches are
tracked by Git.
