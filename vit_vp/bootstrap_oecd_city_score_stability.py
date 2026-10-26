#!/usr/bin/env python3
"""Hierarchical spatial-cell bootstrap for 200 vs 700 image city-score stability."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


TOPICS = ["收入", "就业", "住房", "教育", "健康", "环境", "安全", "公民参与", "服务可及性", "社区", "生活满意度"]


def grid_key(lat: float, lon: float) -> tuple[int, int]:
    return math.floor(lat / .01), math.floor(lon * math.cos(math.radians(lat)) / .01)


def load_grid_values(manifest_path: Path, prediction_path: Path):
    manifest = pd.read_csv(manifest_path, dtype={"uuid": str})
    pred = pd.read_csv(prediction_path, dtype={"image_id": str})
    frame = manifest[["uuid", "city", "lat", "lon"]].merge(
        pred[["image_id", "predicted_vp_score"]], left_on="uuid", right_on="image_id", validate="one_to_one")
    frame["grid"] = [grid_key(float(lat), float(lon)) for lat, lon in zip(frame.lat, frame.lon)]
    cells = {}
    for city, group in frame.groupby("city", sort=False):
        cells[city] = [cell.predicted_vp_score.to_numpy(float)
                       for _, cell in group.groupby("grid", sort=False)]
    return cells


def bootstrap_city_scores(cells, cities, replicates, seed):
    rng = np.random.default_rng(seed)
    scores = np.empty((replicates, len(cities)), dtype=float)
    mean_cell_counts = []
    for j, city in enumerate(cities):
        city_cells = cells[city]
        n_cells = len(city_cells)
        mean_cell_counts.append(n_cells)
        for r in range(replicates):
            sampled_cells = rng.integers(0, n_cells, size=n_cells)
            cell_means = []
            for k in sampled_cells:
                values = city_cells[k]
                resampled = rng.choice(values, size=len(values), replace=True)
                cell_means.append(float(resampled.mean()))
            scores[r, j] = float(np.mean(cell_means))
    return scores, mean_cell_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-200", type=Path, required=True)
    parser.add_argument("--predictions-200", type=Path, required=True)
    parser.add_argument("--manifest-700", type=Path, required=True)
    parser.add_argument("--predictions-700", type=Path, required=True)
    parser.add_argument("--city-table-200", type=Path, required=True)
    parser.add_argument("--city-table-700", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()

    c200, c700 = load_grid_values(args.manifest_200, args.predictions_200), load_grid_values(args.manifest_700, args.predictions_700)
    cities = sorted(c200)
    if len(cities) != 50 or set(cities) != set(c700):
        raise ValueError("Both datasets must contain the same 50 cities")
    t200 = pd.read_csv(args.city_table_200).set_index("city").loc[cities]
    t700 = pd.read_csv(args.city_table_700).set_index("city").loc[cities]
    themes = [*TOPICS, "oecd_11_theme_equal_mean_derived"]
    y = {theme: t200[theme].to_numpy(float) for theme in themes}
    x200, ncell200 = bootstrap_city_scores(c200, cities, args.replicates, args.seed)
    x700, ncell700 = bootstrap_city_scores(c700, cities, args.replicates, args.seed + 1)

    rows = []
    for theme in themes:
        valid = np.isfinite(y[theme])
        rho200 = np.array([spearmanr(x[valid], y[theme][valid]).statistic for x in x200])
        rho700 = np.array([spearmanr(x[valid], y[theme][valid]).statistic for x in x700])
        rows.append({
            "oecd_theme": theme,
            "is_official_oecd_theme": theme in TOPICS,
            "n_cities": int(valid.sum()),
            "mean_occupied_1km_cells_200": float(np.mean(ncell200)),
            "mean_occupied_1km_cells_700": float(np.mean(ncell700)),
            "spatial_bootstrap_spearman_sd_200": float(np.std(rho200, ddof=1)),
            "spatial_bootstrap_spearman_sd_700": float(np.std(rho700, ddof=1)),
            "spatial_bootstrap_sd_ratio_700_over_200": float(np.std(rho700, ddof=1) / np.std(rho200, ddof=1)),
            "spatial_bootstrap_spearman_ci95_low_200": float(np.quantile(rho200, .025)),
            "spatial_bootstrap_spearman_ci95_high_200": float(np.quantile(rho200, .975)),
            "spatial_bootstrap_spearman_ci95_low_700": float(np.quantile(rho700, .025)),
            "spatial_bootstrap_spearman_ci95_high_700": float(np.quantile(rho700, .975)),
        })
    out = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_dir / "spatial_grid_bootstrap_stability.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "replicates": args.replicates,
        "unit": "Within each city, resample occupied approximately 1 km grid cells with replacement; within each selected cell resample its observed image predictions with replacement; recompute the equal-weight grid-mean city score and Spearman correlation to the fixed OECD values.",
        "interpretation": "This conditional spatial sampling sensitivity measures uncertainty in the sampled city VP means; it does not increase the 50-city or 26-country OECD sample size, and is not an independent validation interval.",
        "seed": args.seed,
    }
    (args.output_dir / "spatial_grid_bootstrap_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out[["oecd_theme", "spatial_bootstrap_spearman_sd_200", "spatial_bootstrap_spearman_sd_700", "spatial_bootstrap_sd_ratio_700_over_200"]].to_string(index=False))


if __name__ == "__main__":
    main()
