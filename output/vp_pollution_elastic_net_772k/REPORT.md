# VP pollution Elastic Net results

- Target: `visual_pollution_score_0_10`; sample: 250 images across five balanced cities.
- Primary raw-share model: repeated nested CV (10 × 5 outer folds); city-adjusted within folds.
- Raw model outer-CV: MAE 1.904, RMSE 2.301, R² -0.001, Spearman 0.127.
- City-only baseline: MAE 1.898, RMSE 2.285, R² 0.016.
- CLR sensitivity outer-CV: MAE 1.899, RMSE 2.286, R² 0.016, Spearman 0.132.
- Final raw model inner-CV choice: l1_ratio=1.00, lambda=3.16228.
- Stable elements under the pre-registered rule: 0.

Coefficients are conditional within-city associations, not causal effects. See the logbook for the interpretation rule.
