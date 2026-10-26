#!/usr/bin/env python3
"""Join V8 image predictions to OECD regional wellbeing and correlate city means."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from scipy.stats import pearsonr, spearmanr


TOPICS = ["收入", "就业", "住房", "教育", "健康", "环境", "安全", "公民参与", "服务可及性", "社区", "生活满意度"]
SEED = 20261024


def grid_key(lat: float, lon: float, size_degrees: float = 0.01) -> tuple[int, int]:
    lat_bin = math.floor(lat / size_degrees)
    lon_bin = math.floor(lon * math.cos(math.radians(lat)) / size_degrees)
    return lat_bin, lon_bin


def cluster_bootstrap(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_boot: int, seed: int) -> dict[str, tuple[float, float]]:
    unique = np.unique(groups)
    grouped = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    pear, spear = [], []
    for _ in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([grouped[g] for g in sampled])
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        pear.append(float(pearsonr(x[idx], y[idx]).statistic))
        spear.append(float(spearmanr(x[idx], y[idx]).statistic))
    if not pear:
        return {"pearson": (float("nan"), float("nan")), "spearman": (float("nan"), float("nan"))}
    return {
        "pearson": tuple(map(float, np.quantile(pear, [0.025, 0.975]))),
        "spearman": tuple(map(float, np.quantile(spear, [0.025, 0.975]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--oecd-xlsx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--expected-per-city", type=int, default=200)
    parser.add_argument("--expected-cities", type=int, default=50)
    parser.add_argument("--expected-regions", type=int, default=None,
                        help="Expected unique OECD regions; defaults to expected cities")
    args = parser.parse_args()
    expected_regions = args.expected_cities if args.expected_regions is None else args.expected_regions
    manifest = pd.read_csv(args.manifest, dtype={"uuid": str, "oecd_region_code": str})
    pred = pd.read_csv(args.predictions, dtype={"image_id": str, "oecd_region_code": str})
    expected_total = args.expected_cities * args.expected_per_city
    if len(manifest) != expected_total or manifest.uuid.nunique() != expected_total:
        raise ValueError(f"Expected {expected_total:,} unique manifest images")
    if len(pred) != expected_total or pred.image_id.nunique() != expected_total or set(pred.image_id) != set(manifest.uuid):
        raise ValueError(f"V8 prediction IDs do not exactly cover the {expected_total:,}-image manifest")
    if pred.predicted_vp_score.isna().any() or not np.isfinite(pred.predicted_vp_score.to_numpy()).all():
        raise ValueError("Found missing or non-finite image predictions")

    workbook = load_workbook(args.oecd_xlsx, read_only=True, data_only=True)
    ws = workbook["欧洲地区_11主题分数"]
    iterator = ws.iter_rows(values_only=True)
    header = None
    data_rows = []
    for row in iterator:
        if row and row[0] == "国家":
            header = [str(x) if x is not None else "" for x in row]
            break
    if header is None:
        raise ValueError("OECD workbook does not contain the expected header row")
    col = {name: i for i, name in enumerate(header)}
    required = ["国家", "地区（OECD region）", "地区代码", *TOPICS]
    if any(name not in col for name in required):
        raise ValueError("OECD workbook headers differ from the expected 11-topic schema")
    for row in iterator:
        if row and row[col["国家"]] and row[col["地区代码"]]:
            data_rows.append({name: row[col[name]] for name in required})
    oecd = pd.DataFrame(data_rows).rename(columns={"国家": "country", "地区（OECD region）": "oecd_region", "地区代码": "oecd_region_code"})
    oecd["oecd_region_code"] = oecd.oecd_region_code.astype(str)
    for topic in TOPICS:
        oecd[topic] = pd.to_numeric(oecd[topic], errors="coerce")
    city_key = manifest[["city", "country", "oecd_region", "oecd_region_code"]].drop_duplicates()
    if city_key.city.nunique() != args.expected_cities:
        raise ValueError(f"Expected exactly {args.expected_cities} distinct cities")
    if city_key.groupby("city")["oecd_region_code"].nunique().max() != 1:
        raise ValueError("A city maps to multiple OECD region codes")
    if city_key.oecd_region_code.nunique() != expected_regions:
        raise ValueError(f"Expected exactly {expected_regions} unique OECD regions")
    match = city_key.merge(oecd, on=["country", "oecd_region", "oecd_region_code"], how="left", validate="many_to_one", indicator=True)
    if not match._merge.eq("both").all():
        raise ValueError(f"Some city region codes are absent from the OECD workbook: {match.loc[match._merge.ne('both'), 'city'].tolist()}")

    joined = manifest.merge(pred[["image_id", "predicted_vp_score", "seed_42_score", "seed_137_score", "seed_2026_score"]],
                            left_on="uuid", right_on="image_id", validate="one_to_one")
    joined["grid"] = [f"{a}_{b}" for a, b in (grid_key(float(lat), float(lon)) for lat, lon in zip(joined.lat, joined.lon))]
    cell_means = joined.groupby(["city", "grid"], as_index=False).agg(
        grid_vp_mean=("predicted_vp_score", "mean"), images_in_grid=("uuid", "count"))
    spatial = cell_means.groupby("city", as_index=False).agg(
        vp_city_mean_spatial_equal_grid=("grid_vp_mean", "mean"), occupied_grid_cells=("grid", "nunique"))
    arithmetic = joined.groupby("city", as_index=False).agg(
        vp_city_mean_arithmetic=("predicted_vp_score", "mean"), image_count=("uuid", "count"))
    city_vp = spatial.merge(arithmetic, on="city", validate="one_to_one")
    city_table = city_vp.merge(match.drop(columns="_merge"), on="city", validate="one_to_one")
    city_table["oecd_11_theme_equal_mean_derived"] = city_table[TOPICS].mean(axis=1, skipna=False)
    if len(city_table) != args.expected_cities or not city_table.image_count.eq(args.expected_per_city).all():
        raise ValueError(f"City summary did not yield exactly {args.expected_cities} cities with {args.expected_per_city} images each")

    region_keys = ["country", "oecd_region", "oecd_region_code"]
    region_table = city_table.groupby(region_keys, as_index=False).agg(
        vp_region_mean_spatial_equal_city=("vp_city_mean_spatial_equal_grid", "mean"),
        vp_region_mean_arithmetic_equal_city=("vp_city_mean_arithmetic", "mean"),
        city_count=("city", "nunique"),
        **{topic: (topic, "first") for topic in TOPICS},
    )
    if len(region_table) != expected_regions or region_table.oecd_region_code.nunique() != expected_regions:
        raise ValueError(f"Region aggregation did not yield exactly {expected_regions} unique OECD regions")
    region_table["oecd_11_theme_equal_mean_derived"] = region_table[TOPICS].mean(axis=1, skipna=False)

    themes = [*TOPICS, "oecd_11_theme_equal_mean_derived"]
    score_methods = {
        "primary_equal_grid_then_equal_city": "vp_region_mean_spatial_equal_city",
        "sensitivity_arithmetic_image_mean_then_equal_city": "vp_region_mean_arithmetic_equal_city",
    }
    groups = region_table.country.astype(str).to_numpy()
    results = []
    for method_index, (method, score_column) in enumerate(score_methods.items()):
        x = region_table[score_column].to_numpy(dtype=float)
        for theme_index, theme in enumerate(themes):
            y = region_table[theme].to_numpy(dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            x_valid, y_valid, groups_valid = x[valid], y[valid], groups[valid]
            if len(x_valid) < 3:
                raise ValueError(f"Fewer than three complete region observations for OECD topic {theme}")
            ci = cluster_bootstrap(x_valid, y_valid, groups_valid, args.bootstrap,
                                   SEED + method_index * 100 + theme_index)
            p, ps = pearsonr(x_valid, y_valid), spearmanr(x_valid, y_valid)
            results.append({
                "vp_aggregation": method,
                "oecd_theme": theme,
                "is_official_oecd_theme": theme in TOPICS,
                "n_regions": len(x_valid), "n_countries": len(np.unique(groups_valid)),
                "vp_mean_method": "city scores are either equal-weight occupied spatial-grid means or arithmetic image means; multiple cities in one OECD region are then averaged equally; each image predicted by 3-seed V8 arithmetic ensemble",
                "pearson_r": float(p.statistic), "pearson_country_cluster_bootstrap_ci95_low": ci["pearson"][0],
                "pearson_country_cluster_bootstrap_ci95_high": ci["pearson"][1],
                "spearman_rho": float(ps.statistic), "spearman_country_cluster_bootstrap_ci95_low": ci["spearman"][0],
                "spearman_country_cluster_bootstrap_ci95_high": ci["spearman"][1],
                "pearson_p_unadjusted_independent_region_assumption": float(p.pvalue),
                "spearman_p_unadjusted_independent_region_assumption": float(ps.pvalue),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    city_table.drop(columns=[]).to_csv(args.output_dir / "city_vp_oecd_join.csv", index=False, encoding="utf-8-sig")
    cell_means.to_csv(args.output_dir / "city_grid_vp_means.csv", index=False, encoding="utf-8-sig")
    region_table.to_csv(args.output_dir / "region_vp_oecd_join.csv", index=False, encoding="utf-8-sig")
    correlation = pd.DataFrame(results)
    correlation.to_csv(args.output_dir / "correlations.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "image_count": len(joined), "city_count": len(city_table), "oecd_region_count": int(region_table.oecd_region_code.nunique()),
        "country_cluster_count": int(city_table.country.nunique()), "images_per_city": args.expected_per_city,
        "primary_city_score": "equal-weight mean of occupied 1 km spatial-grid means; no population weights are available in the supplied OECD workbook",
        "primary_region_score": "equal-weight mean of the city-level primary scores when multiple sampled cities map to one OECD region; correlations use one row per unique OECD region",
        "sensitivity_city_score": f"arithmetic mean of all {args.expected_per_city} image predictions; region sensitivity score equally averages city arithmetic means",
        "image_model": "V8 Qwen/TrueSkill-trained DINOv3 ViT-B/16 last-two-block fine-tune; three seeds averaged equally",
        "image_score_direction": "higher values indicate greater predicted visual pollution; scores are not clipped or re-normalized",
        "preprocessing": "exact V8 deployment edge and preprocess function; EXIF transpose, RGB, aspect-preserving resize with long edge from model.json, nearest multiple of 16, Pillow bicubic, ImageNet normalization",
        "correlation": "Pearson and Spearman for both city-aggregation methods; correlations use one row per OECD region; 95% percentile intervals from country-cluster bootstrap; ordinary p-values included only as unadjusted descriptive values under an independent-region assumption",
        "topic_tests": "11 OECD topics are exploratory and not adjusted for multiple comparisons",
        "missing_oecd_values": "N/A entries are treated as missing; each official topic uses all cities with that topic observed. The derived 11-theme equal mean uses complete cases only and is not an official OECD composite index.",
        "bootstrap_replicates": args.bootstrap, "bootstrap_seed": SEED,
    }
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(correlation[["vp_aggregation", "oecd_theme", "n_regions", "pearson_r", "spearman_rho"]].to_string(index=False))
    print(f"Wrote {args.expected_cities}-city, {expected_regions}-region join and correlations to {args.output_dir}")


if __name__ == "__main__":
    main()
