# Results

## V7 3,000-pair run

| Metric | Value |
|---|---:|
| Completed unordered pairs | 3,000 |
| MEC directional samples | 18,000 |
| Calibrated votes | 2,646 (88.2%) |
| Calibrated ties | 78 (2.6%) |
| Invalid pairs | 276 (9.2%) |
| Mean absolute calibrated margin | 1.7300 |
| Mean BPDE | 0.4229 |
| Mean first-position score advantage | 0.5389 |
| Wall time | 38,301.2 s (10 h 38 m 21 s) |
| Peak allocated CUDA memory | 5.878 GiB |

All 250 images entered the ranking, with 16–24 accepted comparisons per image
(median 21). Q-score and TrueSkill were strongly aligned within the run
(Spearman rho = 0.9923); this internal agreement is not external validation.

## V7 1,500-vs-3,000 consistency

The 3,000-pair run contains all 1,500 pairs from the smaller run plus another
1,500 balanced pairs. Across 250 images, the final scores had Pearson r = 0.911
and Spearman rho = 0.909. Mean absolute rank movement was 23.65 positions.
Among 1,150 repeated pairs with decisive outcomes in both runs, 977 agreed
(84.96%). See [`results/v7_1500_vs_v7_3000/REPORT.md`](results/v7_1500_vs_v7_3000/REPORT.md)
for the complete analysis and limitations.
