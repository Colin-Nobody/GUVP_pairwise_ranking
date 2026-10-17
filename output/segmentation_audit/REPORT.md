# DeepLabv3+ ADE20K target-domain audit: computational stage

Date: 2026-09-20

## Status

The 30-image stratified audit set and both inference experiments are complete.
The human class-specific rating stage is intentionally still open. Do not run
the selected method on all 250 images until those ratings have been reviewed.

Human-review entry point:

```text
output/segmentation_audit/visual_review/index.html
```

## Audit sample

- 30 images: six per city.
- Exactly two high-, two middle-, and two low-VP images per city.
- VP strata are city-relative and use v8 scores only for coverage, never as
  pixel labels.
- Coverage includes 23 Mapillary and 7 KartaView images, 6 panoramic images,
  4 portrait images, and resolutions up to 4032x3024.
- Selection also rewards platform, projection, resolution, aspect-ratio, and
  v7/v8 disagreement coverage.

The full deterministic selection is in `audit_manifest.csv` and can be rebuilt
with `segmentation/prepare_audit_manifest.py`.

## Compared inference policies

### Baseline

Official MMSegmentation ADE20K test resize and network inference. Argmax is
taken at test resolution and the categorical mask is restored to source size
on the CPU, avoiding an infeasible full-resolution 150-channel GPU allocation.

### Sliding 512 / 25% overlap

Aspect-ratio-preserving 512x512 source-image tiles, stride 384. Overlapping
logits are accumulated with equal weight in float16 and argmax is applied only
after all contributions are merged. This is not mask overwriting.

## Integrity checks

- Both methods produced 30 original-size masks and 30 original-size overlays.
- Each method produced 4,500 class rows (30 images x 150 ADE20K classes).
- Every image's 150 pixel ratios sums to 1 within floating-point tolerance.
- No invalid class IDs, missing outputs, CUDA errors, or OOM events occurred.
- Maximum allocated GPU memory: baseline 872.1 MiB; sliding 474.0 MiB.
- Total measured inference time: baseline 2.42 s; sliding 228.57 s.

## Method sensitivity

These comparisons measure agreement between methods, not accuracy against
ground truth:

- Mean pixel agreement: 0.7060.
- Median pixel agreement: 0.7613.
- Range of pixel agreement: 0.1768 to 0.9763.
- Mean class IoU between methods: 0.2029.
- Mean detected class count: 12.67 baseline versus 24.23 sliding.

The large difference proves that the future regression features are sensitive
to inference geometry. Method choice cannot be treated as an inconsequential
implementation detail.

## Initial visual finding

In the inspected low-agreement cases, the 512 sliding output contains obvious
rectangular tile discontinuities and context-dependent label changes. It often
finds more small regions, but many are spatially fragmented or implausible. The
current sliding configuration should therefore **not** be selected merely
because it reports more classes.

The resized baseline is spatially more coherent but visibly loses some small
objects and fine boundaries. The human audit should determine whether its
important classes are still reliable enough for area features.

## Human rating task

Use the HTML gallery to compare original, baseline, and sliding views. For the
important ADE20K classes, fill `audit_ratings.csv` using:

```text
0  unusable or seriously wrong
1  partially usable
2  broadly reliable
NA absent or impossible to assess
```

Record systematic errors such as sign/poster-to-wall confusion, missing poles
or streetlights, road/sidewalk confusion, vegetation confusion, lost small
objects, panorama distortion, and sliding tile seams.

## Current decision

- Baseline: candidate, pending human class-specific review.
- Sliding 512 / 25%: no-go in its current form due to visible tile artifacts.
- Full 250-image run: paused until the human audit is completed.

