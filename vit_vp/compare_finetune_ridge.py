#!/usr/bin/env python3
"""Compare last-two-block ViT fine-tuning against frozen Ridge on paired city OOF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "output/vit_vp_1000_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261019)
    return parser.parse_args()


def metric_values(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "spearman": float(spearmanr(y, pred).statistic),
    }


def paired_cluster_bootstrap(
    frame: pd.DataFrame, *, replicates: int, rng: np.random.Generator,
) -> list[dict[str, object]]:
    cities = sorted(frame["city"].unique())
    city_blocks = {city: block for city, block in frame.groupby("city", sort=False)}
    y = frame["y_true"].to_numpy(dtype=float)
    pred_ft = frame["finetune_pred"].to_numpy(dtype=float)
    pred_ridge = frame["ridge_pred"].to_numpy(dtype=float)
    point_ft = metric_values(y, pred_ft)
    point_ridge = metric_values(y, pred_ridge)
    boot = {metric: np.empty(replicates, dtype=float)
            for metric in ("mae", "rmse", "spearman", "mae_reduction_pct_vs_baseline")}

    for index in range(replicates):
        sampled_cities = rng.choice(cities, size=len(cities), replace=True)
        sampled = pd.concat([city_blocks[city] for city in sampled_cities], ignore_index=True)
        y_b = sampled["y_true"].to_numpy(dtype=float)
        ft_b = sampled["finetune_pred"].to_numpy(dtype=float)
        ridge_b = sampled["ridge_pred"].to_numpy(dtype=float)
        ft_metrics = metric_values(y_b, ft_b)
        ridge_metrics = metric_values(y_b, ridge_b)
        for metric in boot:
            if metric == "mae_reduction_pct_vs_baseline":
                boot[metric][index] = 100.0 * (ridge_metrics["mae"] - ft_metrics["mae"]) / ridge_metrics["mae"]
            else:
                boot[metric][index] = ft_metrics[metric] - ridge_metrics[metric]

    rows = []
    for metric, values in boot.items():
        if metric == "mae_reduction_pct_vs_baseline":
            estimate = 100.0 * (point_ridge["mae"] - point_ft["mae"]) / point_ridge["mae"]
            unit = "percent"
        else:
            estimate = point_ft[metric] - point_ridge[metric]
            unit = "metric_difference"
        rows.append({
            "metric": metric,
            "finetune_point": point_ft.get(metric, float("nan")),
            "ridge_point": point_ridge.get(metric, float("nan")),
            "difference_point": estimate,
            "ci95_lower": float(np.quantile(values, 0.025)),
            "ci95_upper": float(np.quantile(values, 0.975)),
            "unit": unit,
            "bootstrap_replicates": replicates,
            "cluster": "city",
            "cluster_count": len(cities),
        })
    return rows


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    run_dir = args.audit_dir / "finetune_last2"
    fine = pd.read_csv(run_dir / "oof_predictions.csv")
    ridge_all = pd.read_csv(args.audit_dir / "frozen_ridge/oof_predictions.csv")
    splits = pd.read_csv(args.audit_dir / "splits/unseen_city_outer.csv")

    keys = ["repeat", "image_id"]
    for frame in (fine, ridge_all, splits):
        if "repeat" in frame:
            frame["repeat"] = frame["repeat"].astype(int)
    fine["outer_fold"] = fine["outer_fold"].astype(int)
    ridge = ridge_all.loc[ridge_all["model"] == "dinov3_ridge"].copy()
    ridge["outer_fold"] = ridge["outer_fold"].astype(int)
    if len(fine) != 3000 or len(ridge) != 3000:
        raise ValueError(f"Expected 3000 OOF rows per model; got fine={len(fine)}, ridge={len(ridge)}")
    if fine.duplicated(keys).any() or ridge.duplicated(keys).any():
        raise ValueError("Duplicate repeat/image OOF key found")

    fine = fine.rename(columns={"y_pred": "finetune_pred", "y_true": "fine_y_true"})
    ridge = ridge.rename(columns={"y_pred": "ridge_pred", "y_true": "ridge_y_true"})
    merged = fine[["repeat", "outer_fold", "image_id", "city", "fine_y_true", "finetune_pred"]].merge(
        ridge[["repeat", "outer_fold", "image_id", "city", "ridge_y_true", "ridge_pred"]],
        on=keys, how="outer", suffixes=("_fine", "_ridge"), indicator=True,
        validate="one_to_one",
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("Fine-tuned and Ridge OOF keys do not match exactly")
    for field in ("outer_fold", "city"):
        if not (merged[f"{field}_fine"] == merged[f"{field}_ridge"]).all():
            raise ValueError(f"Models have mismatched OOF {field} assignments")
    if not np.allclose(merged["fine_y_true"], merged["ridge_y_true"], rtol=0, atol=1e-6):
        raise ValueError("Models do not use identical target values")
    merged["y_true"] = merged["fine_y_true"].astype(float)
    merged["city"] = merged["city_fine"]
    merged["outer_fold"] = merged["outer_fold_fine"].astype(int)
    merged = merged.drop(columns=["_merge", "outer_fold_fine", "outer_fold_ridge", "city_fine", "city_ridge",
                                  "fine_y_true", "ridge_y_true"])

    semantic = ridge_all.loc[ridge_all["model"] == "semantic_elastic_net"].copy()
    semantic["outer_fold"] = semantic["outer_fold"].astype(int)
    if len(semantic) != 3000 or semantic.duplicated(keys).any():
        raise ValueError(f"Expected 3000 unique semantic Elastic Net OOF rows; got {len(semantic)}")
    fine_identity = fine[["repeat", "outer_fold", "image_id", "city", "fine_y_true"]]
    semantic_pairs = fine_identity.merge(
        semantic[["repeat", "outer_fold", "image_id", "city", "y_true", "y_pred"]],
        on=keys, how="outer", suffixes=("_fine", "_semantic"), indicator=True,
        validate="one_to_one",
    )
    if not (semantic_pairs["_merge"] == "both").all():
        raise ValueError("Fine-tuned and semantic Elastic Net OOF keys do not match exactly")
    for field in ("outer_fold", "city"):
        if not (semantic_pairs[f"{field}_fine"] == semantic_pairs[f"{field}_semantic"]).all():
            raise ValueError(f"Fine-tuned and semantic Elastic Net OOF {field} assignments differ")
    if not np.allclose(semantic_pairs["fine_y_true"], semantic_pairs["y_true"], rtol=0, atol=1e-6):
        raise ValueError("Fine-tuned and semantic Elastic Net targets differ")
    semantic_predictions = semantic_pairs[["repeat", "image_id", "y_pred"]].rename(
        columns={"y_pred": "semantic_elastic_net_pred"}
    )
    merged = merged.merge(semantic_predictions, on=keys, how="left", validate="one_to_one")
    if merged["semantic_elastic_net_pred"].isna().any():
        raise ValueError("Some fine-tuned OOF rows have no semantic Elastic Net prediction")

    test_cities = splits.loc[splits["partition"] == "test", ["repeat", "outer_fold", "city"]].drop_duplicates()
    if test_cities.duplicated(["repeat", "outer_fold", "city"]).any():
        raise ValueError("Duplicate held-out city in split manifest")
    merged = merged.merge(test_cities.assign(is_test_city=True), on=["repeat", "outer_fold", "city"],
                          how="left", validate="many_to_one")
    if merged["is_test_city"].isna().any():
        raise ValueError("OOF row city/fold is not marked as held out in split manifest")
    for (repeat, fold), group in merged.groupby(["repeat", "outer_fold"]):
        city_counts = group.groupby("city").size()
        if len(group) != 200 or len(city_counts) != 4 or not (city_counts == 50).all():
            raise ValueError(f"Unexpected test coverage for repeat={repeat}, outer_fold={fold}")
    for repeat, group in merged.groupby("repeat"):
        if len(group) != 1000 or group["image_id"].nunique() != 1000 or group["city"].nunique() != 20:
            raise ValueError(f"Incomplete OOF coverage for repeat={repeat}")
    if set(merged["repeat"].unique()) != {1, 2, 3}:
        raise ValueError("Expected exactly three OOF repeats")

    merged = merged.sort_values(["repeat", "city", "image_id"]).reset_index(drop=True)
    metric_rows = []
    averaged = merged.groupby(["image_id", "city"], as_index=False).agg(
        y_true=("y_true", "mean"), finetune_pred=("finetune_pred", "mean"),
        ridge_pred=("ridge_pred", "mean"),
        semantic_elastic_net_pred=("semantic_elastic_net_pred", "mean"),
    )
    rng = np.random.default_rng(args.seed)
    bootstrap_rows = []
    comparisons = [("dinov3_ridge", "ridge_pred"),
                   ("semantic_elastic_net", "semantic_elastic_net_pred")]
    evaluations = [(f"repeat_{int(repeat)}", group) for repeat, group in merged.groupby("repeat", sort=True)]
    evaluations.append(("repeat_averaged_predictions", averaged))
    for baseline, prediction_column in comparisons:
        for evaluation, group in evaluations:
            ft_m = metric_values(group["y_true"].to_numpy(), group["finetune_pred"].to_numpy())
            base_m = metric_values(group["y_true"].to_numpy(), group[prediction_column].to_numpy())
            for metric in ft_m:
                metric_rows.append({"baseline": baseline, "evaluation": evaluation, "metric": metric,
                                    "finetune": ft_m[metric], "baseline_score": base_m[metric],
                                    "difference_finetune_minus_baseline": ft_m[metric] - base_m[metric],
                                    "n_images": len(group)})
            boot_frame = group[["city", "y_true", "finetune_pred", prediction_column]].rename(
                columns={prediction_column: "ridge_pred"}
            )
            for row in paired_cluster_bootstrap(
                boot_frame, replicates=args.bootstrap_replicates, rng=rng,
            ):
                row["baseline"] = baseline
                row["evaluation"] = evaluation
                bootstrap_rows.append(row)

    out = run_dir
    pd.DataFrame(metric_rows).to_csv(out / "model_comparison_metrics.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(out / "model_comparison_city_bootstrap.csv", index=False)
    merged.to_csv(out / "paired_oof_predictions.csv", index=False)
    metadata = {
        "comparisons": ["last-two-block DINOv3 fine-tuning minus frozen DINOv3 Ridge",
                        "last-two-block DINOv3 fine-tuning minus semantic pooled Elastic Net"],
        "paired_keys": ["repeat", "image_id"],
        "validation": "3 repeats x 5 outer city folds; 1000 images and 20 cities per repeat; 50 images/city",
        "bootstrap_unit": "city, sampled with replacement; all images in each sampled city kept together",
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
        "primary_summary": "OOF predictions averaged per image across repeats, with repeat-specific results also reported",
        "interpretation": "Approximate uncertainty over 20 city clusters; repeated splits are not independent observations",
        "label_scope": "Fixed global Qwen/TrueSkill VP targets; not independent human ratings",
        "paired_rows": len(merged),
        "repeat_averaged_images": len(averaged),
        "cities": int(merged["city"].nunique()),
    }
    (out / "model_comparison_bootstrap_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("Paired OOF validation passed: 3000 matched repeat/image rows; 15 complete held-out folds.")
    print(pd.DataFrame(metric_rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(pd.DataFrame(bootstrap_rows).query("evaluation == 'repeat_averaged_predictions'")
          [["baseline", "metric", "difference_point", "ci95_lower", "ci95_upper"]]
          .to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Wrote paired predictions, model comparisons, bootstrap, and metadata to {out}")


if __name__ == "__main__":
    main()
