# Results

## V7 1,500-pair run

| Metric | Value |
|---|---:|
| Completed unordered pairs | 1,500 |
| MEC directional samples | 9,000 |
| Calibrated votes | 1,305 (87.0%) |
| Calibrated ties | 48 (3.2%) |
| Invalid pairs | 147 (9.8%) |
| Mean absolute calibrated margin | 1.7531 |
| Mean BPDE | 0.4168 |
| Mean first-position score advantage | 0.5299 |
| Wall time | 18,954.9 s (5 h 15 m 55 s) |
| Peak allocated CUDA memory | 5.878 GiB |

All 250 images entered the final TrueSkill ranking. After exclusions, each image
had 7–12 accepted comparisons (median 11). Q-score and TrueSkill agreed strongly
within the run (Spearman rho = 0.9827), although this internal agreement is not an
external accuracy validation.

## TrueSkill ranking comparison with V6 3,000-pair run

The final 250-image rankings were compared directly:

| Metric | V6 3,000 vs V7 1,500 |
|---|---:|
| Matched images | 250 |
| Pearson VP-score correlation | 0.8144 |
| Spearman rank correlation | 0.8077 |
| Kendall rank correlation | 0.6258 |
| Mean absolute rank difference | 31.9 positions |
| Most-polluted top-10 overlap | 3/10 |
| Least-polluted bottom-10 overlap | 5/10 |
| Most-polluted top-25 overlap | 14/25 |
| Least-polluted bottom-25 overlap | 14/25 |

All 1,500 V7 physical pairs occurred in the V6 manifest. Among the 1,028 pairs
with an accepted winner in both runs, 822 winners agreed and 206 differed, for
79.96% agreement.

The rankings are strongly positively related but not interchangeable. V7 improved
response validity and MEC consistency, while its smaller comparison graph produced
higher TrueSkill uncertainty. The prompt and scoring polarity also changed, so the
two vote sets should remain separate.
