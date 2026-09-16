# Results

## V8 3,000-pair run

| Metric | Value |
|---|---:|
| Completed unordered pairs | 3,000 |
| MEC directional samples | 18,000 |
| Calibrated votes | 2,645 (88.2%) |
| Calibrated ties | 89 (3.0%) |
| Invalid pairs | 266 (8.9%) |
| Mean absolute calibrated margin | 1.4909 |
| Mean BPDE | 0.4651 |
| Mean first-position score advantage | 0.4655 |
| Wall time | 38,013.5 s (10 h 33 m 34 s) |
| Peak allocated CUDA memory | 5.878 GiB |

All 250 images entered the ranking, with 17–24 accepted comparisons per image
(median 21). Q-score and TrueSkill were strongly aligned within the run
(Spearman rho = 0.9924); this internal agreement is not external validation.

## Paired comparison with V7 3,000

The runs used the same 3,000 physical pairs and differed only in the prompt
wording. Final VP scores had Pearson r = 0.9455 and Spearman rho = 0.9452.
Among 2,429 pairs with decisive outcomes in both runs, 2,156 winners agreed
(88.76%, Cohen's kappa = 0.775). Mean absolute rank movement was 17.70 positions;
both most-polluted and least-polluted top-10 sets overlapped by 7/10.

V8 reduced the measured first-position score advantage by 13.6%, while its mean
absolute calibrated margin fell by 13.8% and BPDE rose by about 10%. The prompt
change therefore preserved relative rankings but changed score-scale behavior;
it is not evidence that V8 is more accurate without human ground truth.
