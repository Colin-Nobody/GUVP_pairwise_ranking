# VP pollution pooled Elastic Net results

- Target: `visual_pollution_score_0_10`; sample: 250 images treated as one pooled study population.
- City labels were deliberately not used for adjustment, splitting, baselines, or inference.
- Primary model: raw semantic shares; 10 × 5 nested CV with global VP-score quintile folds and exploratory lambda-min selection.
- Raw model outer-CV: MAE 1.924, RMSE 2.322, R² -0.017, Spearman -0.019.
- Global-mean baseline: MAE 1.912, RMSE 2.306, R² -0.002.
- CLR sensitivity: MAE 1.898, RMSE 2.276, R² 0.024, Spearman 0.146.
- Final raw model lambda-min choice: l1_ratio=0.70, lambda=0.417532.
- Elements passing the relaxed exploratory stability screen: 2.

Positive beta means that a larger element share is associated with a higher VP score in this pooled dataset; negative beta means the reverse.
These are exploratory associations, not causal effects or evidence of generalisation to other cities/datasets.
