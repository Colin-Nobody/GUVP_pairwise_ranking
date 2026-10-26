#!/usr/bin/env python3
"""Compare city-score and OECD-correlation stability for nested 200 vs 700 image samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def paired_country_bootstrap_delta(x_small, x_large, y, groups, replicates, seed):
    valid = np.isfinite(x_small) & np.isfinite(x_large) & np.isfinite(y)
    x_small, x_large, y, groups = x_small[valid], x_large[valid], y[valid], groups[valid]
    unique = np.unique(groups)
    grouped = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    delta = []
    for _ in range(replicates):
        draw = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([grouped[group] for group in draw])
        if (np.unique(x_small[indices]).size < 2 or np.unique(x_large[indices]).size < 2
                or np.unique(y[indices]).size < 2):
            continue
        rho_small = spearmanr(x_small[indices], y[indices]).statistic
        rho_large = spearmanr(x_large[indices], y[indices]).statistic
        delta.append(float(rho_large - rho_small))
    return np.quantile(delta, [0.025, 0.975]) if delta else (np.nan, np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-city-table", type=Path, required=True)
    parser.add_argument("--large-city-table", type=Path, required=True)
    parser.add_argument("--small-correlations", type=Path, required=True)
    parser.add_argument("--large-correlations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()

    small = pd.read_csv(args.small_city_table)
    large = pd.read_csv(args.large_city_table)
    if small.city.nunique() != 50 or large.city.nunique() != 50:
        raise ValueError("Both analyses must contain the same 50 cities")
    city = small.merge(large, on=["city", "country", "oecd_region", "oecd_region_code"],
                       suffixes=("_200", "_700"), validate="one_to_one")
    if len(city) != 50:
        raise ValueError("The 200- and 700-image city sets differ")
    a = city.vp_city_mean_spatial_equal_grid_200.to_numpy(float)
    b = city.vp_city_mean_spatial_equal_grid_700.to_numpy(float)
    city_out = city[["city", "country", "oecd_region", "oecd_region_code"]].copy()
    city_out["vp_mean_200"] = a
    city_out["vp_mean_700"] = b
    city_out["score_change_700_minus_200"] = b - a
    city_out["absolute_score_change"] = np.abs(b - a)
    city_out["rank_200"] = pd.Series(a).rank(ascending=False, method="average").to_numpy()
    city_out["rank_700"] = pd.Series(b).rank(ascending=False, method="average").to_numpy()
    city_out["rank_change_700_minus_200"] = city_out.rank_700 - city_out.rank_200
    city_rank_rho = float(spearmanr(a, b).statistic)

    corr_small = pd.read_csv(args.small_correlations).set_index("oecd_theme")
    corr_large = pd.read_csv(args.large_correlations).set_index("oecd_theme")
    if set(corr_small.index) != set(corr_large.index):
        raise ValueError("The OECD topic sets differ")
    groups = city.country.astype(str).to_numpy()
    themes = list(corr_small.index)
    output = []
    for i, theme in enumerate(themes):
        x0 = a
        x1 = b
        y = city[f"{theme}_200"].to_numpy(float)
        valid = np.isfinite(x0) & np.isfinite(x1) & np.isfinite(y)
        paired_ci = paired_country_bootstrap_delta(x0, x1, y, groups, args.bootstrap, args.seed + i)
        r0, r1 = corr_small.loc[theme], corr_large.loc[theme]
        output.append({
            "oecd_theme": theme,
            "is_official_oecd_theme": bool(r1.is_official_oecd_theme),
            "n_cities": int(valid.sum()),
            "spearman_rho_200": float(r0.spearman_rho),
            "spearman_rho_700": float(r1.spearman_rho),
            "spearman_delta_700_minus_200": float(r1.spearman_rho - r0.spearman_rho),
            "paired_country_bootstrap_delta_spearman_ci95_low": float(paired_ci[0]),
            "paired_country_bootstrap_delta_spearman_ci95_high": float(paired_ci[1]),
            "spearman_country_ci_width_200": float(r0.spearman_country_cluster_bootstrap_ci95_high - r0.spearman_country_cluster_bootstrap_ci95_low),
            "spearman_country_ci_width_700": float(r1.spearman_country_cluster_bootstrap_ci95_high - r1.spearman_country_cluster_bootstrap_ci95_low),
            "pearson_r_200": float(r0.pearson_r),
            "pearson_r_700": float(r1.pearson_r),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    city_out.to_csv(args.output_dir / "city_score_stability.csv", index=False, encoding="utf-8-sig")
    result = pd.DataFrame(output)
    result.to_csv(args.output_dir / "sample_size_correlation_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"city score rank Spearman(200 vs 700)={city_rank_rho:.4f}")
    print(f"median absolute city-score change={city_out.absolute_score_change.median():.4f}; "
          f"90th percentile={city_out.absolute_score_change.quantile(.9):.4f}")
    print(result[["oecd_theme", "spearman_rho_200", "spearman_rho_700", "spearman_delta_700_minus_200",
                  "paired_country_bootstrap_delta_spearman_ci95_low",
                  "paired_country_bootstrap_delta_spearman_ci95_high"]].to_string(index=False))
    print(f"Wrote sample-size comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
