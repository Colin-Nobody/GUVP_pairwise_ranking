# DINOv3 ViT-B/16 direct VP prediction: first implementation results

Run date: 2026-10-19. Stage: frozen-feature Ridge and MLP, last-two-block fine-tuning, paired city-cluster uncertainty analysis, within-city spatial evaluation, comparison-edge-isolated evaluation, and fine-tuned deployment packaging are complete.

## Dataset and checkpoint

- 1,000 street-view images across 20 European cities (50 per city); all decode successfully.
- Target: `visual_pollution_score_0_10` from Qwen2.5-VL 7B, prompt v6; larger means more visual pollution. These are VLM-derived labels, not independent human ratings.
- Compared features: frozen DINOv3 ViT-B/16 LVD-1689M, CLS / patch-mean / concatenation, long edge 384 / 512. ViT weights are pinned to Hugging Face revision `ec54a6b69c8ff6066571dec86ee2b7dd26836570`, SHA-256 `9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b`.
- Three visually confirmed near-duplicate image pairs were kept in the same city partition. A fourth perceptual-hash candidate was a false positive. No exact byte duplicates or unreadable images were found.

## Evaluation

The main internal evaluation uses three repeated outer 5-fold city-grouped splits. Each outer fold trains on 16 cities (800 images) and evaluates on four unseen cities (200 images). Within each outer training set, 4-fold city-grouped CV selects model parameters. The ViT head is selected from 60 Ridge configurations; the pooled semantic Elastic Net baseline is selected from 60 configurations. All transforms are fit on the corresponding training fold.

| Model | MAE, 0–10 units | RMSE | R² | Spearman |
| --- | ---: | ---: | ---: | ---: |
| Frozen DINOv3 + Ridge | 1.245 ± 0.011 | 1.568 ± 0.013 | 0.534 ± 0.008 | 0.731 ± 0.006 |
| Semantic features + pooled Elastic Net | 1.615 ± 0.018 | 1.992 ± 0.008 | 0.248 ± 0.006 | 0.535 ± 0.023 |
| Outer training mean | 1.912 ± 0.013 | 2.323 ± 0.016 | -0.023 ± 0.014 | -0.174 ± 0.071 |

Values are the mean and standard deviation across the three repeated OOF runs. These three repeats measure sensitivity to the fixed city partitions; they are not independent samples. City-cluster bootstrap on per-image predictions averaged over the three repeats gives ViT-minus-Elastic-Net MAE difference `-0.374`, 95% interval `[-0.488, -0.279]`; RMSE difference `-0.425`, interval `[-0.549, -0.321]`; Spearman difference `+0.196`, interval `[+0.132, +0.266]`. The bootstrap samples 20 cities and is approximate at this small number of clusters.

Full results, split assignments, inner selected configurations, subgroup metrics, OOF predictions, and 2,000-replicate bootstrap outputs are in `frozen_ridge/`. In particular, `frozen_ridge/oof_predictions.csv` contains 9,000 rows (three models × 1,000 images × three repeats).

## Frozen-feature MLP comparison

A nested 48-candidate search compared the six frozen feature views (CLS, patch mean, concatenation at 384/512) and the preplanned head settings (dropout 0.2/0.5, learning rate 1e-4/1e-3, weight decay 1e-3/1e-2). Each outer fold selected by mean inner-city validation MAE; the final outer fit used the median best epoch from inner folds and never read outer-test labels. Predictions average seeds 42, 137, and 2026.

| Model | MAE, 0–10 units | RMSE | R² | Spearman |
| --- | ---: | ---: | ---: | ---: |
| Frozen DINOv3 + MLP | 1.252 ± 0.009 | 1.580 ± 0.013 | 0.527 ± 0.008 | 0.724 ± 0.005 |
| Frozen DINOv3 + Ridge | 1.245 ± 0.011 | 1.568 ± 0.013 | 0.534 ± 0.008 | 0.731 ± 0.006 |

The MLP selected patch-mean features in all 15 outer folds. On per-image predictions averaged across the three repeats, the city-cluster bootstrap (5,000 resamples of 20 cities) estimates MLP-minus-Ridge MAE +0.0003 (95% CI −0.0211 to +0.0226), RMSE +0.0061 (−0.0182 to +0.0302), and Spearman −0.0057 (−0.0180 to +0.0051). This shows no detectable MLP improvement. The frozen Ridge remains the available frozen-model deployment artifact; the later fine-tuning and spatial/edge-isolated analyses are reported below.

MLP OOF predictions, selected settings, per-repeat metrics, and bootstrap comparison are in `frozen_mlp/`.

## Limited backbone fine-tuning and paired comparison

The V3 model updates only the final two DINOv3 ViT-B/16 blocks and an MLP regression head. It uses the same three repeated, 5-fold unseen-city outer splits and 4-fold city-grouped inner selection as the frozen baselines. Each outer test fold contains four unseen cities (200 images); inner folds select input long edge (384 or 512) and backbone learning rate. The outer-test labels are used only for final scoring. Predictions average seeds 42, 137, and 2026.

| Model | MAE, 0–10 units | RMSE | R² | Spearman | Pearson |
| --- | ---: | ---: | ---: | ---: | ---: |
| DINOv3 last-two-block fine-tuning + MLP | 1.186 ± 0.010 | 1.499 ± 0.007 | 0.574 ± 0.004 | 0.753 ± 0.003 | 0.758 ± 0.003 |
| Same-fold frozen DINOv3 + Ridge | 1.245 ± 0.011 | 1.568 ± 0.013 | 0.534 ± 0.008 | 0.731 ± 0.006 | — |

Across the 15 outer folds, inner selection chose a 512-pixel long edge 11 times and 384 pixels four times; backbone learning rate was `1e-5` in all folds. The head learning rate was `1e-4`, with weight decay `0.01`. The mean fine-tuned MAE over the three repeat-level OOF runs is 1.1857 (SD 0.0100), versus 1.2447 (SD 0.0110) for same-fold Ridge.

The paired city-cluster bootstrap resampled all images within each of the 20 cities together (2,000 resamples; seed 20261019). It uses the per-image OOF predictions averaged across the three repeats for the primary summary; repeat-level intervals are also retained. For fine-tuning minus frozen Ridge, the averaged-prediction differences were MAE −0.0543 (95% CI −0.0794 to −0.0271), RMSE −0.0622 (−0.0903 to −0.0330), and Spearman +0.0169 (+0.0052 to +0.0274). The relative MAE reduction versus Ridge was 4.41% (2.22% to 6.40%). This is a consistent, modest gain over the frozen Ridge; the interval describes uncertainty over 20 city clusters and is approximate.

Against the same-fold semantic pooled Elastic Net baseline, fine-tuning reduced MAE by 0.4282 (95% CI 0.3383 to 0.5359 lower), reduced RMSE by 0.4861 (0.3833 to 0.6119 lower), and increased Spearman by 0.2147 (0.1522 to 0.2766). The relative MAE reduction was 26.68% (21.81% to 31.90%), exceeding the project’s predeclared 5% practical-improvement reference. This threshold is a project decision rule, not a universal standard.

These results estimate prediction of the fixed global Qwen/TrueSkill-derived VP labels, not independent human ratings. The three repeated splits are sensitivity checks and are not independent samples. The full paired outputs, repeat-specific metrics, city-cluster bootstrap intervals, and run metadata are in `finetune_last2/`.

## Within-city spatial validation

To test transfer to new locations within already represented cities, images were grouped into city-centered 250 m metric grid cells. Each city has 22–40 groups (50 images per city); confirmed near-duplicate pairs fall in the same group. Three repeated 5-fold outer splits and nested 4-fold inner splits keep cells intact. A 100 m exclusion buffer is a separate sensitivity protocol. Coordinates use a city-centered equirectangular approximation, which is adequate for these city-scale extents; exact source coordinates are retained in the manifest.

| Protocol | Model | MAE | RMSE | R² | Spearman |
| --- | --- | ---: | ---: | ---: | ---: |
| 250 m groups | Frozen DINOv3 + Ridge | 1.217 ± 0.009 | 1.541 ± 0.007 | 0.550 ± 0.004 | 0.741 ± 0.005 |
| 250 m groups | Semantic Elastic Net | 1.601 ± 0.017 | 2.137 ± 0.261 | 0.126 ± 0.219 | 0.567 ± 0.008 |
| 250 m + 100 m buffer | Frozen DINOv3 + Ridge | 1.217 ± 0.010 | 1.544 ± 0.012 | 0.548 ± 0.007 | 0.739 ± 0.007 |
| 250 m + 100 m buffer | Semantic Elastic Net | 1.605 ± 0.022 | 2.134 ± 0.256 | 0.128 ± 0.214 | 0.565 ± 0.006 |

Ridge remains ahead of the semantic baseline with or without the 100 m buffer. The buffer removes 21–47 outer-training images per fold (753–779 training images remain). This protocol evaluates new spatial cells in seen cities and still predicts the frozen global VLM labels; the edge-isolated analysis below addresses the latter dependency separately. OOF predictions, city metrics, selected settings, and 2,000-city-bootstrap outputs are in `within_city_spatial_250m/` and `within_city_spatial_250m_buffer100/`.

## Comparison-edge-isolated ranking evaluation

For each unseen-city outer/inner split, training VP targets were recomputed from decisive comparison edges whose two images were both in that training partition. Cross-partition edges were excluded. Inner selection used held-out validation edges only; outer scoring used edges whose two images were both in the held-out cities. No full-graph TrueSkill score was used as a model target in this analysis. VP direction was reversed within each training graph because the observed winner denotes the cleaner image.

Across three repeats, each outer test fold contributed 188–230 decisive held-out edges (about 1,025 edges per repeat); 12–28 of the 200 test images had no held-out edge in a given fold. Outer training graphs covered all 800 training images and were connected; inner training graphs had 0–1 unconnected/unrated images. Held-out subgraphs had 4–10 components, so only witnessed held-out edges are scored and no complete test-image ordering is claimed.

| Model | Held-out edge accuracy, mean ± repeat SD | Same-city edges | Cross-city edges |
| --- | ---: | ---: | ---: |
| Frozen DINOv3 + Ridge | 0.759 ± 0.013 | 0.755 | 0.760 |
| Semantic Elastic Net | 0.685 ± 0.012 | 0.705 | 0.679 |

This supports improved prediction of the held-out Qwen comparison decisions under this protocol, not independent human preference. It reports edge accuracy rather than absolute VP MAE because independently generated training-graph scores have fold-specific scales. Detailed per-edge predictions, graph coverage, and inner selected configurations are in `edge_isolated_unseen_city/`.

## Provisional frozen Ridge deployment model

Nested CV selected patch-mean features most often, with 384-pixel long edge. A separate 5-fold GroupKFold across all 20 cities selected `patch_mean`, edge 384, and Ridge alpha 1000 for the inference head. The final head was then fit on all 1,000 labeled images. This all-data fit is for prediction; do not use its training scores as a generalization estimate.

The executable predictor is `vit_vp/predict.py`; it accepts an image or image directory. Example:

```bash
/home/colin/miniconda3/envs/guvp-vit/bin/python vit_vp/predict.py data/expansion_750/street_images_1000/54352a61-c857-4bf1-afb9-23ca04309844.jpg
```

Inference smoke output for that image was 6.2212. This is an in-sample example from the 1,000-image development set and is not an accuracy measurement. The predictor verifies the checkpoint hash before loading. Raw output is not clipped; the scoring model uses the fixed dataset-relative VP scale.

## Fine-tuned deployment model

After completing nested OOF evaluation and paired comparisons, three final models were fitted on all 1,000 development images using seeds 42, 137, and 2026. The selected configuration is a 512-pixel long edge, backbone learning rate `1e-5`, head learning rate `1e-4`, weight decay `0.01`, and 16 epochs. The epoch count is the median of the 15 outer-fold `refit_epochs` values, which themselves were selected from inner-validation epochs; no full-data validation or outer-test score was used to choose it. The released predictor averages the three seed predictions.

The model files and configuration are in `deployment/finetuned_last2/`. Run a single image or a directory with:

```bash
/home/colin/miniconda3/envs/guvp-vit/bin/python vit_vp/predict_finetuned.py data/expansion_750/street_images_1000/54352a61-c857-4bf1-afb9-23ca04309844.jpg
```

The loader checks the pretrained DINOv3 checkpoint hash and each fine-tuned weight-file hash. A CUDA smoke inference completed for the example above and returned 5.6607. This image is in the training set, so this prediction demonstrates successful loading/inference only; it is not a performance estimate. Use the nested OOF results for reported generalization.

## Limits and next steps

Both frozen Ridge and last-two-block fine-tuning clear the predeclared 5% practical-improvement reference over the pooled Elastic Net. Fine-tuning also shows a paired improvement over frozen Ridge, though the relative MAE gain is modest (4.41%). For a frozen encoder, Ridge remains simpler than the MLP. The fine-tuning OOF predictions are the evaluation record; the separate all-data fine-tuned package below is for inference only.

The fixed-label city and spatial evaluations predict the Qwen/TrueSkill teacher scores; the edge-isolated experiment separately tests held-out Qwen comparison decisions. Neither establishes independent human judgments. Remaining research steps are source/panorama subgroup analysis, label-order sensitivity, and learning curves. The fine-tuned all-data package is for inference, while the nested OOF estimates remain the generalization evidence. Any independent human-perception claim still requires human-rated data.

The Mapillary Vistas v2 raw train/validation data are not needed by the ViT→VP experiments. They were checksum-verified after transfer to `/media/colin/白/GUVP_archive/mapillary_vistas`; `/home/colin/datasets/mapillary_vistas` remains a symlink for existing segmentation configs. Keep the data drive mounted for any future segmentation training or feature extraction.

## Artifacts

- Dataset and label audit: `dataset_audit.json`, `label_provenance.json`.
- Input inventory: `image_manifest.csv`.
- Duplicate review: `near_duplicate_candidates.csv`, `near_duplicate_review.csv`, `near_duplicate_groups.csv`, `near_duplicate_review.jpg`.
- Fixed splits: `splits/unseen_city_outer.csv`, `splits/unseen_city_inner.csv`, `splits/split_audit.json`.
- Frozen image features: `features/dinov3_vitb16_lvd1689m_edge384.npz`, `features/dinov3_vitb16_lvd1689m_edge512.npz`.
- Trained inference head and selection record: `deployment/model.json`, `deployment/ridge_head.npz`, `deployment/final_group_cv_candidates.csv`.
- Fine-tuning evaluation: `finetune_last2/oof_predictions.csv`, `finetune_last2/outer_fold_metrics.csv`, `finetune_last2/per_repeat_metrics.csv`, `finetune_last2/inner_selected_configs.csv`, `finetune_last2/paired_oof_predictions.csv`, `finetune_last2/model_comparison_metrics.csv`, `finetune_last2/model_comparison_city_bootstrap.csv`, `finetune_last2/model_comparison_bootstrap_metadata.json`.
- Fine-tuned deployment package: `deployment/finetuned_last2/model.json`, `seed_42.safetensors`, `seed_137.safetensors`, `seed_2026.safetensors`; predictor: `vit_vp/predict_finetuned.py`.
- Environment lock: `conda-explicit.txt`, `pip-freeze.txt`.
