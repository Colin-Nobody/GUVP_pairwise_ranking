# Segmentation Plan

Last updated: 2026-09-20

## Purpose

This document is the persistent handoff for the semantic-segmentation stage of the Global Urban Visual Pollution (GUVP) project. Future Codex sessions should read this file before changing segmentation code, environments, data, or model choices.

The research objective is to discover which visible street-view elements are most strongly associated with visual-pollution judgments. Semantic segmentation supplies candidate visual-element features; it does **not** predict visual pollution directly. Visual-pollution scores and pairwise judgments are used later by Elastic Net models.

## Current project state

- Repository: `/home/colin/GUVP_pairwise_ranking`
- Street-view images: `data/street_images/`
- Image manifest: `data/images_manifest.csv`
- Images: 250 total, 50 per city across five cities.
- Image sizes vary substantially, including `2048x1536`, `2048x1024`, `4032x3024`, `3840x2160`, and portrait images.
- Current v8 score file: `output/vlm_7b_definition_v8_3000_prefix_cache/perception_scores.csv`
- The score file contains `visual_pollution_score_0_10`, TrueSkill mean/sigma, rank, wins, losses, and comparison counts.
- GPU: NVIDIA GeForce RTX 5060, approximately 7.5 GiB usable VRAM.
- Existing VLM environment: `/home/colin/miniconda3/envs/guvp-vlm` (left unchanged).
- Dedicated segmentation environment: `/home/colin/miniconda3/envs/guvp-seg`, created from scratch with Python 3.11.
- Segmentation stack: PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, MMEngine 0.10.7, MMCV 2.1.0, and MMSegmentation 1.2.2.
- GPU/model loading verified on the NVIDIA GeForce RTX 5060 (compute capability 12.0).

The worktree already contains numerous user changes and generated experiment outputs. Preserve them. Do not clean, reset, or overwrite unrelated files.

## Critical methodological rule

Do **not** train or fine-tune the segmentation model with the 250 image-level visual-pollution scores. Those scores are not pixel labels.

The correct sequence is:

```text
ADE20K-pretrained segmentation model
    -> target-domain segmentation audit
    -> frozen inference on 250 images
    -> semantic feature extraction
    -> compositional-data processing
    -> pairwise Logistic Elastic Net (primary)
       and TrueSkill-score Elastic Net (secondary)
```

Fine-tuning a segmentation model is permitted only after creating pixel-level target-domain masks.

## Phase 1 baseline model

Start with the official MMSegmentation checkpoint:

```text
Architecture: DeepLabv3+
Backbone: ResNet-50
Output stride: 8
Training dataset: ADE20K
Classes: 150
Official schedule: 80k
Official crop: 512x512
Config name: deeplabv3plus_r50-d8_4xb4-80k_ade20k-512x512
```

Official config:

<https://github.com/open-mmlab/mmsegmentation/blob/main/configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-80k_ade20k-512x512.py>

Official checkpoint:

<https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r50-d8_512x512_80k_ade20k/deeplabv3plus_r50-d8_512x512_80k_ade20k_20200614_185028-bf1400d8.pth>

Official reported ADE20K validation mIoU is 42.72 for the R50 80k model. The corresponding official training configuration reports about 10.6 GB VRAM on a V100, so reproducing full training is not the immediate task on the local 8 GB GPU. Load the pretrained checkpoint first.

Use R50 before R101. R101 has a modest benchmark gain but greater memory and compute requirements.

## Environment strategy

Do not modify the stable `guvp-vlm` environment in place. Create a separate `guvp-seg` environment.

A clean environment was created instead of cloning the VLM environment:

```bash
conda create -n guvp-seg python=3.11 pip -y
conda activate guvp-seg
```

The exact core dependency record is `segmentation/environment.yml`. Because
MMSegmentation 1.2.2 eagerly imports `mmcv.ops`, `mmcv-lite` was insufficient.
MMCV 2.1.0 was built locally with its CPU extension; DeepLabv3+ inference itself
runs on the GPU through CUDA-enabled PyTorch.

Activate the working environment with:

```bash
conda activate guvp-seg
```

## Phase 1A: three-image smoke test

Before selecting or processing an audit set, verify that the model can:

1. Download/load the pinned config and checkpoint.
2. Run on the RTX 5060 without CUDA errors or OOM.
3. Produce a 150-class integer mask.
4. Save a color overlay.
5. Restore the mask to the original image dimensions.
6. Export per-class pixel counts and ratios.

On the local 8 GB GPU, MMSegmentation's default postprocessing cannot upsample
all 150 logits to 4032x3024 at once (that allocation alone is about 6.8 GiB).
Keep the official preprocessing and network inference, take argmax at the test
resolution, and restore the categorical mask on the CPU with nearest-neighbor
interpolation. Record this low-memory postprocessing explicitly in run metadata.

Use three representative inputs:

- one `2048x1536` image;
- one `2048x1024` panorama;
- one `4032x3024` high-resolution image.

The first implementation should use the official preprocessing/inference pipeline as a reproducible baseline. Do not silently replace it with a custom resize pipeline.

## Phase 1B: 30-image target-domain audit

Computational status (2026-09-20): the deterministic manifest, baseline output,
512x512/25%-overlap sliding-logit output, structured features, comparison
statistics, and local human-review gallery have been generated under
`output/segmentation_audit/`. The human class-specific ratings remain open.

Initial visual inspection found severe tile-grid discontinuities in several
512 sliding outputs. That configuration is currently a no-go. The baseline is
more spatially coherent but must still pass the human important-class review
before use on all 250 images.

Create a deterministic audit manifest containing six images per city:

- two high-VP images;
- two middle-VP images;
- two low-VP images.

Also maximize coverage of:

- image source/platform;
- aspect ratio and resolution;
- ordinary and panoramic images;
- visually simple and cluttered scenes;
- images containing signs, poles, streetlights, vehicles, fences, and small objects;
- large v7/v8 ranking disagreements, where possible.

VP scores are used only to stratify audit coverage. They are not segmentation-training labels.

Expected audit outputs:

```text
output/segmentation_audit/
  audit_manifest.csv
  masks/<image_id>.png
  overlays/<image_id>.png
  semantic_features_audit.csv
  audit_ratings.csv
  REPORT.md
```

`semantic_features_audit.csv` should contain one row per image and one ratio column per ADE20K class, plus identifiers and non-feature metadata. Preserve raw class counts as well as ratios.

## Audit targets

Inspect the following classes particularly carefully:

```text
building, wall, road, sidewalk,
tree, grass, plant, sky,
car, bus, truck, person,
fence, signboard, poster, pole, streetlight,
awning, bench, ashcan
```

Record a simple rating for each important class/image combination:

```text
0 = unusable or seriously wrong
1 = partially usable
2 = broadly reliable
NA = class absent or impossible to assess
```

Record systematic failure modes separately, including:

- signboards/posters predicted as wall or building;
- pole/streetlight omission;
- road/sidewalk confusion;
- tree/plant/grass confusion;
- loss of people, vehicles, or other small objects;
- panorama-edge distortion;
- errors caused by aggressive downscaling;
- failure to represent litter, overhead wires, graffiti, or damaged surfaces.

ADE20K has some useful visual-pollution-adjacent labels but does not adequately cover overhead wires, scattered litter, graffiti, and damaged surfaces. Do not treat the 150 ADE20K classes as a complete visual-pollution ontology.

## Image geometry and inference policy

Never stretch all source images directly to a square and then treat the resulting pixel fractions as unbiased area ratios.

Procedure:

1. Establish results with the official preprocessing pipeline.
2. Restore every predicted mask to its source image dimensions before feature calculation.
3. On the 30-image audit set, compare the baseline against aspect-ratio-preserving overlapping sliding-window inference.
4. Candidate tiles are `512x512` initially and `768x768` if memory allows, with approximately 25% overlap.
5. Merge overlapping tiles by averaging logits before `argmax`, not by overwriting class masks.
6. Choose the final inference policy from the audit evidence and record all resize, padding, tile, overlap, and interpolation settings.

## Go/no-go decision after the audit

### If important classes are reliable

- Freeze the segmentation model.
- Run inference on all 250 images.
- Do not train the segmentation model further.
- Extract semantic features for the statistical stage.

### If broad classes are reliable but small objects are weak

Before training, compare:

- sliding-window inference;
- larger inference tiles;
- multi-scale inference;
- a modern comparison model such as SegFormer-B0/B2 or UPerNet;
- a street-specific Mapillary-pretrained model if it provides more relevant classes.

### If visual-pollution-relevant classes fail systematically

Create target-domain pixel annotations before fine-tuning. A future custom ontology may include:

```text
advertising/signage
utility_wire
pole
streetlight
vehicle
temporary_barrier
litter/debris
graffiti
damaged_surface
street_furniture
building
road/sidewalk
vegetation
sky
other
```

For an initial fine-tuning study, reserve a city-stratified frozen test set before training. Begin by freezing the backbone and training the decoder/head with mixed precision, batch size 1, and gradient accumulation. Do not decide to fine-tune solely because the aggregate ADE20K mIoU is modest; use target-domain class-specific evidence.

## Phase 2 output for all 250 images

When the audit passes, produce:

```text
output/segmentation_deeplabv3plus_r50_ade20k/
  run_config.json
  model_metadata.json
  masks/<image_id>.png
  overlays/<image_id>.png
  class_counts.csv
  semantic_features_raw.csv
  failures.csv
  run.log
```

At minimum extract:

- all retained class-area ratios;
- top/middle/bottom spatial ratios;
- semantic class count;
- semantic entropy;
- semantic boundary density;
- connected-component/fragmentation descriptors where meaningful;
- natural, transport, built-environment, street-furniture, and potential-clutter summaries, while preserving raw class features.

All outputs must remain joinable to `data/images_manifest.csv` and the VP score file through normalized image IDs.

## Later statistical stage (not Phase 1)

Do not start this stage until the segmentation audit is accepted.

Pixel-area ratios are compositional data because they sum to one. Do not interpret a plain Elastic Net fitted to all raw ratios as independent causal element weights. The later modeling plan is:

1. Remove extremely rare or demonstrably unreliable segmentation classes using rules fixed without looking at regression results.
2. Preserve a discovery-oriented set of adequately observed candidate classes.
3. Apply an appropriate log-ratio/ILR or log-contrast treatment, with zero handling documented.
4. Fit pairwise Logistic Elastic Net on feature differences as the primary model:

   ```text
   P(A is judged more polluted than B)
       = sigmoid(beta * (features_A - features_B))
   ```

5. Fit Elastic Net regression on TrueSkill/image-level VP scores as a secondary analysis.
6. Use Leave-One-City-Out outer validation and inner CV for hyperparameters.
7. Split by image/city, never randomly by pair, to prevent the same image leaking into train and test data.
8. Report standardized coefficients, bootstrap selection frequency, Spearman correlations, pairwise accuracy/AUC, ranking correlations, and cross-city stability.
9. Repeat for v7, v8, and consensus targets to quantify prompt sensitivity.

The resulting associations describe elements related to the VLM-derived visual-pollution judgments unless and until they are validated with human pairwise comparisons. Do not make causal claims from Elastic Net coefficients.

## Immediate next action for a future Codex session

Phase 1A was completed on 2026-09-20. The isolated environment, pinned source,
checkpoint, reproducible smoke-test script, original-size masks, overlays, and
class ratios are present. All three representative resolutions passed; peak
allocated GPU memory was below 0.8 GiB with low-memory postprocessing. Outputs
are in `output/segmentation_smoke/`.

The next action is Phase 1B:

1. Read this file and inspect current git status without disturbing unrelated changes.
2. Review the three smoke-test overlays and metadata.
3. Build the deterministic 30-image, city/VP/resolution-stratified audit manifest.
4. Compare the baseline against overlapping sliding-window inference.
5. Perform the class-specific visual audit before processing all 250 images.

The first milestone is complete: the ADE20K-pretrained DeepLabv3+ R50 ran
successfully on the three representative local images and produced auditable
outputs at original resolution. No new model training was performed.
