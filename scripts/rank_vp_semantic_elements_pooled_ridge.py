#!/usr/bin/env python3
"""Rank non-rare semantic elements by continuous pooled Ridge weights.

This is a descriptive companion to the pooled Elastic Net analysis.  The 250
images are one study population; the ``city`` column is deliberately unused.
Ridge is used because correlated compositional elements should retain
continuous, jointly-shrunk coefficients rather than being set to zero.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold

ALPHAS = tuple(np.logspace(-4, 4, 41))
MIN_IMAGES = 25  # 10% of the fixed 250-image study population


def parse_args() -> argparse.Namespace:
    root = Path("/home/colin/GUVP_pairwise_ranking")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path,
        default=root / "output/segmentation_mapillary_v2_finetune_3e-5_regression_250_baseline",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "output/vp_pollution_pooled_ridge_136k_nonrare_elements",
    )
    parser.add_argument("--target", default="visual_pollution_score_0_10")
    parser.add_argument("--outer-repeats", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261004)
    return parser.parse_args()


def make_strata(y: np.ndarray) -> np.ndarray:
    return pd.qcut(pd.Series(y).rank(method="first"), q=5, labels=False).astype(str).to_numpy()


@dataclass
class Fit:
    mean: np.ndarray
    scale: np.ndarray
    model: Ridge

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict((x - self.mean) / self.scale)

    @property
    def original_beta(self) -> np.ndarray:
        return self.model.coef_ / self.scale


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> Fit:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    model = Ridge(alpha=alpha).fit((x - mean) / scale, y)
    return Fit(mean, scale, model)


def select_alpha(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, pd.DataFrame]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    strata = make_strata(y)
    rows = []
    for alpha in ALPHAS:
        losses = []
        for train, test in splitter.split(x, strata):
            fitted = fit_ridge(x[train], y[train], alpha)
            losses.append(mean_squared_error(y[test], fitted.predict(x[test])))
        rows.append({
            "alpha": alpha,
            "mean_mse": float(np.mean(losses)),
            "se_mse": float(np.std(losses, ddof=1) / np.sqrt(len(losses))),
        })
    table = pd.DataFrame(rows).sort_values("mean_mse")
    return float(table.iloc[0]["alpha"]), table


def nested_cv(x: np.ndarray, y: np.ndarray, repeats: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    strata = make_strata(y)
    folds, predictions = [], []
    for repeat in range(repeats):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + repeat)
        for fold, (train, test) in enumerate(splitter.split(x, strata), 1):
            alpha, _ = select_alpha(x[train], y[train], seed + 10_000 + repeat * 10 + fold)
            fitted = fit_ridge(x[train], y[train], alpha)
            pred = fitted.predict(x[test])
            base = np.full(len(test), y[train].mean())
            folds.append({
                "repeat": repeat + 1, "fold": fold, "alpha": alpha,
                "mae": mean_absolute_error(y[test], pred),
                "rmse": float(np.sqrt(mean_squared_error(y[test], pred))),
                "r2": r2_score(y[test], pred),
                "spearman": pd.Series(y[test]).corr(pd.Series(pred), method="spearman"),
                "mean_baseline_mae": mean_absolute_error(y[test], base),
                "mean_baseline_rmse": float(np.sqrt(mean_squared_error(y[test], base))),
                "mean_baseline_r2": r2_score(y[test], base),
            })
            predictions.extend({
                "repeat": repeat + 1, "fold": fold, "row_index": int(i),
                "observed_vp_score": y[i], "predicted_vp_score": pred[pos],
                "mean_baseline_prediction": base[pos],
            } for pos, i in enumerate(test))
    return pd.DataFrame(folds), pd.DataFrame(predictions)


def main() -> None:
    args = parse_args()
    run_dir, output_dir = args.run_dir.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(run_dir / "elastic_net_ready.csv")
    schema = pd.read_csv(run_dir / "elastic_net_feature_schema.csv")
    if len(data) != 250 or args.target not in data:
        raise RuntimeError("Expected the 250-image VP table and requested target")
    all_features = schema["feature_column"].tolist()
    occurrence = (data[all_features].to_numpy(float) > 0).sum(axis=0)
    keep_mask = occurrence >= MIN_IMAGES
    kept = schema.loc[keep_mask].copy().reset_index(drop=True)
    feature_columns = kept["feature_column"].tolist()
    x, y = data[feature_columns].to_numpy(float), data[args.target].to_numpy(float)
    if len(feature_columns) < 2:
        raise RuntimeError("Too few non-rare elements after filtering")

    folds, predictions = nested_cv(x, y, args.outer_repeats, args.seed)
    alpha, tuning = select_alpha(x, y, args.seed + 90_000)
    final = fit_ridge(x, y, alpha)

    rng = np.random.default_rng(args.seed + 100_000)
    boot = []
    for number in range(args.bootstrap):
        indices = rng.choice(len(y), size=len(y), replace=True)
        boot.append(fit_ridge(x[indices], y[indices], alpha).original_beta)
    boot_array = np.vstack(boot)
    beta = final.original_beta
    standardised = final.model.coef_
    result = kept.copy()
    result["n_images_present"] = occurrence[keep_mask]
    result["image_prevalence"] = occurrence[keep_mask] / len(data)
    result["standardised_beta"] = standardised
    result["beta_per_full_share"] = beta
    result["beta_per_10_percentage_points"] = beta * 0.10
    result["bootstrap_beta_low_2_5"] = np.percentile(boot_array, 2.5, axis=0)
    result["bootstrap_beta_high_97_5"] = np.percentile(boot_array, 97.5, axis=0)
    result["bootstrap_sign_consistency"] = np.maximum((boot_array >= 0).mean(axis=0), (boot_array <= 0).mean(axis=0))
    result["association_direction"] = np.where(beta > 0, "higher VP", "lower VP")
    result = result.sort_values("standardised_beta", key=np.abs, ascending=False)

    summary = pd.DataFrame([
        {"metric": field, "mean": folds[field].mean(), "sd_across_outer_folds": folds[field].std(ddof=1)}
        for field in ["mae", "rmse", "r2", "spearman", "mean_baseline_mae", "mean_baseline_rmse", "mean_baseline_r2"]
    ])
    result.to_csv(output_dir / "continuous_semantic_element_weight_ranking.csv", index=False)
    kept.assign(n_images_present=occurrence[keep_mask], image_prevalence=occurrence[keep_mask] / len(data)).to_csv(
        output_dir / "included_nonrare_elements.csv", index=False
    )
    schema.loc[~keep_mask].assign(n_images_present=occurrence[~keep_mask], image_prevalence=occurrence[~keep_mask] / len(data)).to_csv(
        output_dir / "excluded_rare_elements.csv", index=False
    )
    tuning.to_csv(output_dir / "ridge_full_data_cv_grid.csv", index=False)
    folds.to_csv(output_dir / "ridge_pooled_nested_cv_folds.csv", index=False)
    predictions.to_csv(output_dir / "ridge_pooled_nested_cv_predictions.csv", index=False)
    summary.to_csv(output_dir / "ridge_pooled_nested_cv_summary.csv", index=False)
    metadata = {
        "analysis_scope": "pooled single population; city labels unused",
        "estimator": "Ridge regression (continuous shrinkage)",
        "target": args.target, "n_images": len(data), "n_kept_elements": len(feature_columns),
        "excluded_if_present_in_fewer_than_images": MIN_IMAGES,
        "outer_repeats": args.outer_repeats, "bootstrap_resamples": args.bootstrap,
        "final_alpha": alpha,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    metric = summary.set_index("metric")["mean"].to_dict()
    report = [
        "# Continuous semantic-element VP weight ranking", "",
        "- Scope: all 250 images treated as one pooled population; city labels are unused.",
        f"- Removed as too rare: elements occurring in fewer than {MIN_IMAGES}/250 images (<10%).",
        f"- Retained: {len(feature_columns)} individual semantic elements; no rare groups are used.",
        f"- Estimator: standardised Ridge regression, alpha={alpha:.6g}, selected by global 5-fold CV.",
        f"- Nested-CV: MAE {metric['mae']:.3f}; RMSE {metric['rmse']:.3f}; R² {metric['r2']:.3f}; Spearman {metric['spearman']:.3f}.",
        f"- Mean-score baseline: MAE {metric['mean_baseline_mae']:.3f}; RMSE {metric['mean_baseline_rmse']:.3f}; R² {metric['mean_baseline_r2']:.3f}.",
        "- Positive coefficients associate a larger share with higher VP score; negative coefficients with lower VP score.",
        "- A 10-point increase is necessarily offset by decreases in other shares; weights are descriptive associations, not causal effects.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n")
    print(f"Completed continuous Ridge ranking: {output_dir}")


if __name__ == "__main__":
    main()
