# VP pollution Elastic Net results

- Target: `visual_pollution_score_0_10`; sample: 1000 images across 20 cities (50 images per city).
- Primary raw-share model: repeated nested CV (10 × 5 outer folds); city-adjusted within folds.
- Raw model outer-CV: MAE 1.550, RMSE 1.908, R² 0.309, Spearman 0.561.
- City-only baseline: MAE 1.756, RMSE 2.160, R² 0.115.
- CLR sensitivity outer-CV: MAE 1.584, RMSE 1.950, R² 0.279, Spearman 0.521.
- Final raw model inner-CV choice: l1_ratio=0.10, lambda=0.421697.
- Stable elements under the pre-registered rule: 28.

Coefficients are conditional within-city associations, not causal effects. See the logbook for the interpretation rule.
