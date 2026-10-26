# V9 DINOv3 ViT: city-disjoint 900/300 evaluation

## Protocol

The model starts from the original DINOv3 ViT-B/16 LVD-1689M pretrained checkpoint; no previous fine-tuned V6/V9 weights are loaded. The final two Transformer blocks and patch-mean MLP head are fine-tuned. Six entire cities (the four external cities plus two original-cohort cities selected by fixed seed 20261020) form a 300-image test set. The remaining 18 original cities (900 images) are used for training.

The configuration and 16-epoch duration are fixed in advance. There is no hyperparameter selection or early stopping on the test set. Three seed models (42, 137, 2026), each reset from the same original pretrained checkpoint, are averaged for the test prediction. V9 targets were generated and globally min-max scaled from the existing joint 1,200-image pairwise graph; no test-specific recalibration is applied.

## Overall test metrics

- N: 300
- MAE: 1.0684
- RMSE: 1.3533
- R²: 0.4032
- Spearman rho: 0.6142
- Pearson r: 0.6464
- Mean prediction minus target: 0.1541

See test_metrics_by_group.csv for external-vs-original-city test breakdown and test_metrics_by_city.csv for each city.

## Interpretation limits

ViT training and model selection exclude all 300 test images and all six test cities. Existing V9 labels remain jointly graph-derived, including cross-partition comparisons.
These scores evaluate image-level and city-held-out generalization to the fixed V9 targets. They are not independent human ratings. The four external cities are all KartaView, while most training images are Mapillary, so the external subgroup also includes a source shift.

No ordinary image-independent p-values are reported because V9 labels come from a shared pairwise comparison graph and images are clustered by city.
