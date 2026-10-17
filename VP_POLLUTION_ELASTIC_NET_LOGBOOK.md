# VP pollution score × semantic segmentation: Elastic Net logbook

## Objective

Estimate stable *associations* between Mapillary Vistas v2 semantic-element
coverage and `visual_pollution_score_0_10`.  The target is a continuous
0–10 rescaling of the 250-image visual-pollution ordering; coefficients are
not interpreted as causal effects.

## Inputs frozen before fitting

- **Sample:** 250 images, balanced across Amsterdam, Barcelona, Berlin, Paris
  and Rome (50 images each).
- **Target:** `visual_pollution_score_0_10` only.
- **Primary features:** 124 semantic area ratios extracted from
  `best_mIoU_iter_772000.pth` (DeepLabV3+ R50-D8; validation mIoU 22.67%).
- **Excluded fields:** ranks, pairwise wins/losses, `q_score_0_10`, TrueSkill
  variables and all other fields derived from the outcome.  Including them
  would leak label information.

## Why this design

Elastic Net performs shrinkage and selection in the presence of correlated
predictors (Zou & Hastie, 2005).  This has precedent in street-view studies
that use semantic segmentation followed by Elastic Net to relate visual
elements to perceived street quality.  Semantic percentages are compositional:
they sum to one, so a coefficient cannot mean an independent increase in one
element with every other element unchanged.  The raw-share model is retained
for readable element labels; a log-ratio model is its compositional robustness
check.

## Primary estimator: city-adjusted raw-share Elastic Net

1. Within every training fold, remove each city's mean from the response and
   each semantic share.  This estimates within-city association and prevents
   city-average visual differences from being mistaken for element effects.
2. Remove zero-variance classes and classes visible in fewer than 5% of that
   training fold's images.  Standardise the retained features using that fold
   only.
3. Tune `l1_ratio` and regularisation strength `lambda` in an inner five-fold
   cross-validation.  Apply the one-standard-error rule, choosing the most
   regularised candidate whose error is within one standard error of the
   minimum.
4. Assess the entire tuning protocol with ten repeated outer five-fold splits,
   stratified jointly by city and within-city score quintile.
5. Report out-of-fold MAE, RMSE, R² and Spearman correlation against both a
   city-mean baseline and the semantic model.

## Robustness and interpretation

- Refit the same nested procedure after zero replacement, closure and a CLR
  transform.  This is a compositional sensitivity model, not the source of
  named-element coefficients.
- Fit the selected raw-share model to all 250 observations, then take 250
  city-stratified bootstrap resamples.  Report standardised coefficients,
  original-share coefficients, selection frequency, sign stability and 95%
  bootstrap percentile intervals.
- Call an element *stable* only when it is selected in at least 70% of outer
  fits and bootstrap fits, with the same coefficient sign in at least 90% of
  its non-zero estimates.
- Repeat the completed analysis later with the higher-mIoU 136k fine-tune
  segmentation table.  Agreement in sign and rank is the required
  segmentation-model sensitivity check.

## Core-plus-grouped-rare sensitivity analysis

The 250-image table is highly sparse: use a prespecified **10% image
prevalence** threshold to retain individual core classes (at least 25 images).
For this sample it retains 42 core classes.  Every non-core class remains in
the composition, but is summed into one of nine theory-defined rare groups:
animals; road users; lane markings; natural elements; built/road
infrastructure; signage/traffic control; street furniture/utilities; vehicles;
and other context.  The resulting parts still close to 100%, so the same
city-adjusted raw-share and CLR nested-CV protocol applies.  This is a
sensitivity analysis, not a replacement for the 124-class main analysis.

## Exploratory pooled single-population analysis (2026-10-04)

The study objective was explicitly narrowed to explaining VP-score variation
within the current 250-image collection, treated as **one study population**.
City names are therefore not used for residualisation, fold construction,
baselines, or generalisation claims.  This version uses the 42 core classes
plus nine grouped rare classes, global VP-score quintile folds, and a
ridge-leaning Elastic Net grid (`l1_ratio` 0.10, 0.30, 0.50, 0.70).  It selects
the minimum-cross-validation-error regularisation value rather than the more
conservative one-standard-error value.

Its relaxed exploratory screen requires selection in >=30% of outer fits and
>=50% of bootstrap fits, with a common sign in >=70% of selected estimates.
Outputs belong in `output/vp_pollution_elastic_net_pooled_136k_core10_grouped`.
They are explicitly descriptive associations in this sample; city-related
visual differences may be part of the measured association and no coefficient
is a causal effect.

## Continuous non-rare element ranking (2026-10-04)

For a complete ranked display rather than sparse selection, retain only the
individual Mapillary classes observed in at least 25 of the 250 images (10%);
this leaves 42 classes and removes 82 near-absent classes.  Fit a pooled,
standardised Ridge regression to retain one continuous, jointly-shrunk
coefficient for every retained class.  City names remain unused.  The ranking
is written to
`output/vp_pollution_pooled_ridge_136k_nonrare_elements/continuous_semantic_element_weight_ranking.csv`.
Bootstrap intervals and sign consistency accompany every weight.  This is a
descriptive ranking only: the selected shrinkage level and out-of-fold metrics
must be reported alongside it, rather than treating a coefficient ranking as
evidence of predictive or causal strength.

## Interpretation rule

Weights are conditional, within-city associations after regularisation.  They
do not prove that physically adding an object would change visual pollution by
the coefficient amount.  Any practical scenario must explicitly exchange
semantic area with other elements because coverage percentages are constrained
to sum to 100%.

## References

- Zou, H. & Hastie, T. (2005). *Regularization and Variable Selection via the
  Elastic Net*. JRSS B, 67, 301–320. https://doi.org/10.1111/j.1467-9868.2005.00503.x
- Zhang et al. (2025). *The Impact of Visual Elements in Street View on Street
  Quality: A Quantitative Study Based on Deep Learning, Elastic Net Regression,
  and SHAP*. https://www.mdpi.com/2071-1050/17/8/3454
- Wang et al. (2022). *Regression analysis with compositional data using
  orthogonal log-ratio coordinates*. https://doi.org/10.1080/03610918.2019.1691224
- Vabalas et al. (2019). *Cross-validation pitfalls when selecting and
  assessing regression and classification models*. https://pmc.ncbi.nlm.nih.gov/articles/PMC3994246/
