# DINOv3 trained on V9 labels: 1,000 development images / 200-image holdout

## Protocol

The ViT starts from the original DINOv3 ViT-B/16 LVD-1689M pretrained checkpoint. No previously fine-tuned V6 weights are loaded. Only the final two Transformer blocks and the patch-mean MLP regressor are trained. The last 200 images were excluded from training, internal split selection, and early stopping. Hyperparameters were locked to the existing selected recipe: 512-pixel long edge, backbone LR 1e-5, head LR 1e-4, weight decay 0.01, 16 epochs, and seeds 42/137/2026.

The first 1,000 images have 3-repeat, 5-fold city-held-out OOF evaluation. The final 3-seed ensemble is fitted on all 1,000 development images and evaluated once on the four-city 200-image holdout. Predictions are evaluated on the fixed V9 0-10 target scale without holdout recalibration.

## Internal development OOF

Metrics by repeat are in internal_oof_metrics_by_repeat.csv.

## Sealed 200-image holdout

- MAE: 1.0531
- RMSE: 1.3261
- R2: 0.3343
- Spearman rho: 0.5482
- Pearson r: 0.5919
- Mean prediction minus target: 0.1043
- City macro MAE: 1.0531
- City macro Spearman rho: 0.5329

City-by-city results are in holdout_metrics_by_city.csv. Unclipped predictions are the primary metrics; clipped metrics are a display-range diagnostic only.

## Interpretation limit

The holdout images and labels are excluded from ViT fitting, model selection, and early stopping. However, the precomputed V9 TrueSkill targets were jointly estimated from a graph containing cross-partition comparisons; these fixed labels cannot be made independent without new ratings.
The holdout images and their scores were never read by the ViT fitting script. However, V9's labels were produced from one joint comparison graph, so the existing train and holdout targets share cross-partition comparison information. This evaluates image-level holdout generalization to the fixed V9 labels, not an independently generated label set or human perception.

No independent-image p-values are reported because target labels are pairwise-graph-derived and dependent.
