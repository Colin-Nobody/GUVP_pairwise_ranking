# Mapillary Vistas v2 segmentation experiment snapshot

Snapshot time: 2026-10-03 22:28 CST.

## Baseline-resize B1 training

- Model: DeepLabV3+ R50-D8, initialized from the ADE20K checkpoint with the Mapillary Vistas v2 (124-class) heads reset.
- Training geometry: aspect-ratio-preserving `Resize(scale=(2048, 512), keep_ratio=True)`, with no random crop; batch size 1.
- Optimizer/schedule: SGD; effective initial learning rate `1.25e-3`; PolyLR to `1e-4` by iteration 450,000, then strict-resume at the `1e-4` floor through iteration 900,000.
- Completed first stage: best validation mIoU **21.93%** at iteration **408,000**; final validation mIoU **21.91%** at iteration **450,000**.
- Strict continuation state: active at iteration **558,500 / 900,000**.  Latest full 2,000-image validation (iteration 556,000): aAcc **86.39%**, mIoU **21.56%**, mAcc **27.51%**, mDice **29.69%**.  Best continuation checkpoint is mIoU **22.27%** at iteration **504,000**.

The runnable configuration and launchers are under `segmentation/mapillary_vistas/`.  The 450k dashboard is at `output/deeplabv3plus_r50_mapillary_v2_baseline_resize_b1_450k/convergence_loss_miou_lr_first_450k.png`.

## Regression segmentation export

- Source model: the 408k best first-stage checkpoint.
- Inference: single-scale, whole-image, aspect-ratio-preserving resize.
- Dataset: 250 source images.
- Export: component percentages and Elastic Net-ready feature/target tables in `output/segmentation_mapillary_v2_baseline_resize_b1_450k_regression_250/`.

## Deliberately excluded from Git

Training checkpoints and the source/mask image bulk are retained locally.  Current checkpoints are approximately 400 MB each (above GitHub's standard 100 MB per-file limit) and are actively updated by the 900k run.  They must be archived separately with Git LFS or object storage after training completes.
