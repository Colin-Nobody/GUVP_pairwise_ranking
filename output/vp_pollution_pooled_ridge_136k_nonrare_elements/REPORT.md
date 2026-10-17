# Continuous semantic-element VP weight ranking

- Scope: all 250 images treated as one pooled population; city labels are unused.
- Removed as too rare: elements occurring in fewer than 25/250 images (<10%).
- Retained: 42 individual semantic elements; no rare groups are used.
- Estimator: standardised Ridge regression, alpha=10000, selected by global 5-fold CV.
- Nested-CV: MAE 1.919; RMSE 2.313; R² -0.008; Spearman 0.031.
- Mean-score baseline: MAE 1.912; RMSE 2.306; R² -0.002.
- Positive coefficients associate a larger share with higher VP score; negative coefficients with lower VP score.
- A 10-point increase is necessarily offset by decreases in other shares; weights are descriptive associations, not causal effects.
