# Segmentation baseline

This directory contains the pinned MMSegmentation DeepLabv3+ R50-D8 ADE20K
baseline described in `SEGMENTATION_PLAN.md`.

## Installed environment

The isolated Conda environment is `/home/colin/miniconda3/envs/guvp-seg`.
It was created from Python 3.11 rather than cloned from `guvp-vlm`; the VLM
environment was not modified. Core versions are recorded in `environment.yml`.

MMSegmentation 1.2.2 imports compiled MMCV operators eagerly. MMCV 2.1.0 was
therefore built locally with its CPU extension. DeepLabv3+ does not use custom
MMCV CUDA operators: model tensors and the actual inference pipeline run on
CUDA through PyTorch 2.11.0+cu128.

For a clean rebuild after installing `environment.yml`, compile MMCV without
CUDA operators (the compiler paths are inside the new environment):

```bash
CUDA_VISIBLE_DEVICES='' \
CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" \
CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++" \
MAX_JOBS=2 MMCV_WITH_OPS=1 \
python -m pip install "mmcv==2.1.0" --no-build-isolation
```

## Verify the installation

From the repository root:

```bash
conda activate guvp-seg
python segmentation/smoke_test.py
```

The script uses the official MMSegmentation preprocessing and model inference
pipeline on three representative local images. To fit the 8 GB GPU, it takes
argmax at the official test resolution and restores the categorical mask to the
source size on the CPU with nearest-neighbor interpolation. This avoids the
default 150-channel full-resolution logits allocation (about 6.8 GiB for a
4032x3024 image). It verifies original-size masks and writes:

```text
output/segmentation_smoke/
  masks/
  overlays/
  class_counts_and_ratios.csv
  run_metadata.json
```

The model identity, source URL, source revision, file size, and checkpoint hash
are pinned in `MODEL_MANIFEST.json`.

## Thirty-image target-domain audit

The deterministic audit sample, baseline inference, 512-pixel sliding-window
comparison, structured outputs, report, and human-review page are available in:

```text
output/segmentation_audit/
```

Open the human-review page directly:

```bash
xdg-open output/segmentation_audit/visual_review/index.html
```

Or serve the repository if the browser restricts local files:

```bash
python -m http.server 8000 --directory /home/colin/GUVP_pairwise_ranking
```

Then visit `http://localhost:8000/output/segmentation_audit/visual_review/`.

The page displays original, baseline, and sliding results side by side, filters
by city and VP stratum, stores image-level ratings in the browser, and exports
them as CSV. `audit_ratings.csv` is the detailed class-level rating template.
