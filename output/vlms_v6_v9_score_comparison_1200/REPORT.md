# V6 vs V9 scores: matched 1,200 images

V6 main 1,000 and external 200 were scored in separate comparison graphs. V9 uses one joint graph for all 1,200 images. These are VLM-derived labels, not human ground truth.

Because V6 used separate graphs, pooled raw V6 scores do not establish an ordering between the two cohorts. Metrics below compare each matched cohort separately; V9 is also min-max rescaled within each cohort for score-error metrics. Percentile ranks are scale-robust.

## Coverage

- V6 main: 1,000 images
- V6 external: 200 images
- V9: 1,200 images
- Matched IDs: 1,200 / 1,200

## Results

| Cohort | N | Pearson | Spearman rho | MAE after within-cohort min-max | Mean absolute percentile-rank gap | Top decile overlap |
|---|---:|---:|---:|---:|---:|---:|
| V6 main 1,000 | 1000 | 0.696 | 0.688 | 1.338 | 0.174 | 51/100 |
| V6 external 200 | 200 | 0.616 | 0.580 | 1.412 | 0.201 | 10/20 |

Pooling within-cohort percentile ranks gives Pearson r=0.670, Spearman rho=0.670; macro cohort Spearman=0.634. This is within-cohort agreement only.

Across 24 cities, median city Spearman rho=0.632, mean=0.628; lowest: Tallinn (0.354), highest: Krakow (0.795).

## Limits

- Separate V6 comparison graphs and joint V9 graph create different score scales.
- Correlation measures agreement between VLM label versions, not human perception.
- No ordinary image-level p-values are reported for graph-derived scores.

## Files

- v6_v9_scores_1200_joined.csv: aligned image scores, local scaling, percentile ranks.
- cohort_metrics.json: cohort and city summaries.
- city_metrics.csv: descriptive city-level correlations.
- largest_rank_changes_top50.csv: largest within-cohort percentile-rank changes.
