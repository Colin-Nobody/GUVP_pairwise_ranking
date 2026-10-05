# Mapillary Vistas v2 segmentation experiment snapshot

Snapshot time: 2026-10-05 (Asia/Shanghai).

## Baseline-resize B1 training

- Model: DeepLabV3+ R50-D8, initialized from the ADE20K checkpoint with the Mapillary Vistas v2 (124-class) heads reset.
- Training geometry: aspect-ratio-preserving `Resize(scale=(2048, 512), keep_ratio=True)`, with no random crop; batch size 1.
- Optimizer/schedule: SGD; effective initial learning rate `1.25e-3`; PolyLR to `1e-4` by iteration 450,000, then strict-resume at the `1e-4` floor through iteration 900,000.
- Completed first stage: best validation mIoU **21.93%** at iteration **408,000**; final validation mIoU **21.91%** at iteration **450,000**.
- Strict continuation completed at iteration **900,000** on 2026-10-04. Final full 2,000-image validation: aAcc **86.30%**, mIoU **21.69%**, mAcc **27.80%**, mDice **30.01%**. The final checkpoint is retained locally at `output/deeplabv3plus_r50_mapillary_v2_baseline_resize_b1_450k/iter_900000.pth`; the best regression-export source checkpoint used so far is iteration **772,000**.

The runnable configuration and launchers are under `segmentation/mapillary_vistas/`.  The 450k dashboard is at `output/deeplabv3plus_r50_mapillary_v2_baseline_resize_b1_450k/convergence_loss_miou_lr_first_450k.png`.

## Regression segmentation exports and visual review

- Four 250-image exports are available: the 450k baseline, the 772k baseline continuation, the fine-tuned model (`finetune_3e-5`), and its baseline comparison. Each export contains 124 semantic-class features and the pollution-score target.
- Baseline exports use single-scale whole-image inference with aspect-ratio-preserving resize. Fine-tuned exports use 512px sliding windows with 25% overlap.
- On 2026-10-04, a random 30-image triptych set was generated for the 772k baseline continuation. Matching 30-image sets are available for the other three exports.
- The 120 review images (four sets × 30 JPGs; approximately 36 MB) are archived in the Git branch `segmentation-visual-archive-20261005`, under each export's `random_sample_30_visuals/triptychs/` directory. The sample manifests and HTML indexes remain in the corresponding local output folders.

These regression exports are exploratory inputs for the downstream statistical analysis; they do not constitute pixel-level fine-tuning labels.

## Local storage archive

- On 2026-10-05, 29 inactive segmentation checkpoints and 21 generated masks/overlays/review-preview directories (2,515 files; 11,638,058,000 bytes) were copied to `/media/colin/白/GUVP_storage/archives/GUVP_pairwise_ranking_2026-10-05/`.
- Every copied file was SHA-256 verified before the local data was replaced with a symlink. `SHA256SUMS` and `SHA256SUMS.tsv` in the archive record the checksums and original repository-relative paths.
- The original project paths continue to work while the 白 drive is mounted at `/media/colin/白`. The 136k fine-tune best checkpoint is included; copy just that checkpoint back to local storage if faster model startup is needed.
- The active VLM expansion run and its data/output directories were excluded from this archive and remain local until the run completes.

## Deliberately excluded from Git

Training checkpoints are kept on the 白 drive instead of Git because individual files exceed GitHub's standard 100 MB per-file limit. Source images and compact analysis tables remain in the project.
