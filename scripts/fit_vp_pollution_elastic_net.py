#!/usr/bin/env python3
"""Nested-CV Elastic Net association analysis for VP pollution scores.

All preprocessing that can depend on the sample distribution is learned inside
the corresponding training fold.  See VP_POLLUTION_ELASTIC_NET_LOGBOOK.md.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)


L1_RATIOS = (0.10, 0.40, 0.70, 0.90, 1.00)
LAMBDAS = tuple(np.logspace(-3.0, 0.5, 25))
PREVALENCE_THRESHOLD = 0.05


def parse_args() -> argparse.Namespace:
    root = Path("/home/colin/GUVP_pairwise_ranking")
    default_run = root / "output/segmentation_mapillary_v2_baseline_resize_b1_772k_regression_250"
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=default_run)
    parser.add_argument("--output-dir", type=Path,
                        default=root / "output/vp_pollution_elastic_net_772k")
    parser.add_argument("--target", default="visual_pollution_score_0_10")
    parser.add_argument("--outer-repeats", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261004)
    return parser.parse_args()


def make_strata(cities: np.ndarray, y: np.ndarray, bins: int = 5) -> np.ndarray:
    """Five equally sized score strata within each city."""
    labels = np.empty(len(y), dtype=object)
    for city in np.unique(cities):
        idx = np.flatnonzero(cities == city)
        ranks = pd.Series(y[idx]).rank(method="first")
        quantiles = pd.qcut(ranks, q=bins, labels=False)
        labels[idx] = [f"{city}_{int(value)}" for value in quantiles]
    return labels.astype(str)


def clr(values: np.ndarray, zero: float = 1e-6) -> np.ndarray:
    """Multiplicative zero replacement followed by centred log ratios."""
    x = np.maximum(values, zero)
    x = x / x.sum(axis=1, keepdims=True)
    logged = np.log(x)
    return logged - logged.mean(axis=1, keepdims=True)


@dataclass
class PreparedData:
    keep: np.ndarray
    x_city_means: dict[str, np.ndarray]
    y_city_means: dict[str, float]
    x_global_mean: np.ndarray
    y_global_mean: float
    centre_mean: np.ndarray
    centre_scale: np.ndarray
    kind: str

    def transform_x(self, x: np.ndarray, cities: np.ndarray) -> np.ndarray:
        raw = x[:, self.keep]
        transformed = clr(raw) if self.kind == "clr" else raw
        means = np.vstack([
            self.x_city_means.get(city, self.x_global_mean) for city in cities
        ])
        centred = transformed - means
        return (centred - self.centre_mean) / self.centre_scale

    def city_y(self, cities: np.ndarray) -> np.ndarray:
        return np.array([
            self.y_city_means.get(city, self.y_global_mean) for city in cities
        ])


@dataclass
class FittedModel:
    prepared: PreparedData
    estimator: ElasticNet

    def predict(self, x: np.ndarray, cities: np.ndarray) -> np.ndarray:
        return self.estimator.predict(self.prepared.transform_x(x, cities)) + self.prepared.city_y(cities)

    def coefficients(self, p: int) -> tuple[np.ndarray, np.ndarray]:
        """Return aligned standardised and raw-share coefficients."""
        standardised = np.zeros(p)
        standardised[self.prepared.keep] = self.estimator.coef_
        original = np.zeros(p)
        if self.prepared.kind == "raw":
            original[self.prepared.keep] = self.estimator.coef_ / self.prepared.centre_scale
        return standardised, original


def prepare_and_fit(
    x: np.ndarray,
    y: np.ndarray,
    cities: np.ndarray,
    l1_ratio: float,
    regularisation: float,
    kind: str,
) -> FittedModel:
    prevalence = (x > 0).mean(axis=0)
    variance = x.var(axis=0)
    keep = np.flatnonzero((prevalence >= PREVALENCE_THRESHOLD) & (variance > 1e-12))
    if len(keep) < 2:
        raise RuntimeError("Fewer than two non-degenerate semantic features in a training fold")

    source = x[:, keep]
    source = clr(source) if kind == "clr" else source
    global_x = source.mean(axis=0)
    global_y = float(y.mean())
    city_x: dict[str, np.ndarray] = {}
    city_y: dict[str, float] = {}
    for city in np.unique(cities):
        idx = cities == city
        city_x[str(city)] = source[idx].mean(axis=0)
        city_y[str(city)] = float(y[idx].mean())
    x_means = np.vstack([city_x[str(city)] for city in cities])
    y_means = np.array([city_y[str(city)] for city in cities])
    centred_x = source - x_means
    centred_y = y - y_means
    scale = centred_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    mean = centred_x.mean(axis=0)
    z = (centred_x - mean) / scale
    estimator = ElasticNet(
        alpha=regularisation,
        l1_ratio=l1_ratio,
        max_iter=30_000,
        tol=1e-5,
        selection="cyclic",
        random_state=0,
    )
    estimator.fit(z, centred_y)
    prepared = PreparedData(
        keep=keep,
        x_city_means=city_x,
        y_city_means=city_y,
        x_global_mean=global_x,
        y_global_mean=global_y,
        centre_mean=mean,
        centre_scale=scale,
        kind=kind,
    )
    return FittedModel(prepared=prepared, estimator=estimator)


def select_hyperparameters(
    x: np.ndarray,
    y: np.ndarray,
    cities: np.ndarray,
    seed: int,
    kind: str,
) -> tuple[float, float, pd.DataFrame]:
    strata = make_strata(cities, y)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    losses: dict[tuple[float, float], list[float]] = {
        (l1, lam): [] for l1 in L1_RATIOS for lam in LAMBDAS
    }
    for train_idx, test_idx in splitter.split(x, strata):
        for l1 in L1_RATIOS:
            for lam in LAMBDAS:
                fitted = prepare_and_fit(x[train_idx], y[train_idx], cities[train_idx], l1, lam, kind)
                predicted = fitted.predict(x[test_idx], cities[test_idx])
                losses[(l1, lam)].append(mean_squared_error(y[test_idx], predicted))

    table = pd.DataFrame([
        {
            "l1_ratio": l1,
            "lambda": lam,
            "mean_mse": float(np.mean(values)),
            "se_mse": float(np.std(values, ddof=1) / np.sqrt(len(values))),
        }
        for (l1, lam), values in losses.items()
    ])
    minimum = table.loc[table["mean_mse"].idxmin()]
    allowed = table[table["mean_mse"] <= minimum["mean_mse"] + minimum["se_mse"]]
    # Higher lambda means more shrinkage; higher l1_ratio breaks ties toward sparsity.
    selected = allowed.sort_values(["lambda", "l1_ratio"], ascending=[False, False]).iloc[0]
    return float(selected["l1_ratio"]), float(selected["lambda"]), table.sort_values("mean_mse")


def evaluate_nested_cv(
    x: np.ndarray,
    y: np.ndarray,
    cities: np.ndarray,
    feature_columns: list[str],
    repeats: int,
    seed: int,
    kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    strata = make_strata(cities, y)
    folds: list[dict] = []
    predictions: list[dict] = []
    all_standardised: list[np.ndarray] = []
    all_original: list[np.ndarray] = []
    for repeat in range(repeats):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + repeat)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x, strata), start=1):
            l1, lam, _ = select_hyperparameters(
                x[train_idx], y[train_idx], cities[train_idx], seed + 10_000 + repeat * 10 + fold, kind
            )
            fitted = prepare_and_fit(x[train_idx], y[train_idx], cities[train_idx], l1, lam, kind)
            prediction = fitted.predict(x[test_idx], cities[test_idx])
            baseline_means = pd.Series(y[train_idx]).groupby(cities[train_idx]).mean().to_dict()
            fallback = float(y[train_idx].mean())
            baseline = np.array([baseline_means.get(city, fallback) for city in cities[test_idx]])
            std_coef, raw_coef = fitted.coefficients(len(feature_columns))
            all_standardised.append(std_coef)
            all_original.append(raw_coef)
            folds.append({
                "model": kind,
                "repeat": repeat + 1,
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_features_after_filter": len(fitted.prepared.keep),
                "l1_ratio": l1,
                "lambda": lam,
                "mae": mean_absolute_error(y[test_idx], prediction),
                "rmse": float(np.sqrt(mean_squared_error(y[test_idx], prediction))),
                "r2": r2_score(y[test_idx], prediction),
                "spearman": pd.Series(y[test_idx]).corr(pd.Series(prediction), method="spearman"),
                "city_baseline_mae": mean_absolute_error(y[test_idx], baseline),
                "city_baseline_rmse": float(np.sqrt(mean_squared_error(y[test_idx], baseline))),
                "city_baseline_r2": r2_score(y[test_idx], baseline),
            })
            for pos, original_idx in enumerate(test_idx):
                predictions.append({
                    "model": kind,
                    "repeat": repeat + 1,
                    "fold": fold,
                    "row_index": int(original_idx),
                    "city": cities[original_idx],
                    "observed_vp_score": y[original_idx],
                    "predicted_vp_score": prediction[pos],
                    "city_baseline_prediction": baseline[pos],
                })
            print(f"[{kind}] repeat {repeat + 1}/{repeats}, fold {fold}/5: lambda={lam:.4g}, l1={l1:.2f}", flush=True)
    return (
        pd.DataFrame(folds),
        pd.DataFrame(predictions),
        np.vstack(all_standardised),
        np.vstack(all_original),
    )


def bootstrap_coefficients(
    x: np.ndarray,
    y: np.ndarray,
    cities: np.ndarray,
    feature_count: int,
    l1: float,
    lam: float,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    standardised: list[np.ndarray] = []
    original: list[np.ndarray] = []
    city_indices = {city: np.flatnonzero(cities == city) for city in np.unique(cities)}
    for number in range(count):
        indices = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in city_indices.values()
        ])
        fitted = prepare_and_fit(x[indices], y[indices], cities[indices], l1, lam, "raw")
        std_coef, raw_coef = fitted.coefficients(feature_count)
        standardised.append(std_coef)
        original.append(raw_coef)
        if (number + 1) % 25 == 0 or number + 1 == count:
            print(f"[bootstrap] {number + 1}/{count}", flush=True)
    return np.vstack(standardised), np.vstack(original)


def metric_summary(folds: pd.DataFrame) -> pd.DataFrame:
    fields = ["mae", "rmse", "r2", "spearman", "city_baseline_mae", "city_baseline_rmse", "city_baseline_r2"]
    rows = []
    for field in fields:
        rows.append({"metric": field, "mean": folds[field].mean(), "sd_across_outer_folds": folds[field].std(ddof=1)})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(run_dir / "elastic_net_ready.csv")
    schema = pd.read_csv(run_dir / "elastic_net_feature_schema.csv")
    feature_columns = schema["feature_column"].tolist()
    missing = [column for column in feature_columns if column not in table.columns]
    if missing or args.target not in table.columns:
        raise RuntimeError(f"Missing required columns: {missing[:3]} target={args.target}")
    if len(table) < 25:
        raise RuntimeError(f"Expected at least 25 images for the repeated 5-fold protocol, found {len(table)}")
    x = table[feature_columns].to_numpy(dtype=float)
    y = table[args.target].to_numpy(dtype=float)
    cities = table["city"].astype(str).to_numpy()
    if not np.allclose(x.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Semantic shares do not close to one")

    raw_folds, raw_predictions, outer_std, outer_original = evaluate_nested_cv(
        x, y, cities, feature_columns, args.outer_repeats, args.seed, "raw"
    )
    clr_folds, clr_predictions, _, _ = evaluate_nested_cv(
        x, y, cities, feature_columns, args.outer_repeats, args.seed + 50_000, "clr"
    )
    final_l1, final_lambda, tuning = select_hyperparameters(x, y, cities, args.seed + 90_000, "raw")
    final_model = prepare_and_fit(x, y, cities, final_l1, final_lambda, "raw")
    final_std, final_original = final_model.coefficients(len(feature_columns))
    boot_std, boot_original = bootstrap_coefficients(
        x, y, cities, len(feature_columns), final_l1, final_lambda, args.bootstrap, args.seed + 100_000
    )

    outer_selected = np.abs(outer_std) > 1e-10
    boot_selected = np.abs(boot_std) > 1e-10
    selection = schema.copy()
    selection["standardised_beta"] = final_std
    selection["beta_per_full_share"] = final_original
    selection["beta_per_10_percentage_points"] = final_original * 0.10
    selection["outer_selection_frequency"] = outer_selected.mean(axis=0)
    outer_nonzero = outer_selected.sum(axis=0)
    selection["outer_sign_consistency"] = np.divide(
        np.abs(np.sign(outer_std).sum(axis=0)), outer_nonzero,
        out=np.zeros_like(outer_nonzero, dtype=float), where=outer_nonzero > 0,
    )
    selection["bootstrap_selection_frequency"] = boot_selected.mean(axis=0)
    bootstrap_nonzero = boot_selected.sum(axis=0)
    selection["bootstrap_sign_consistency"] = np.divide(
        np.abs(np.sign(boot_std).sum(axis=0)), bootstrap_nonzero,
        out=np.zeros_like(bootstrap_nonzero, dtype=float), where=bootstrap_nonzero > 0,
    )
    selection["bootstrap_beta_low_2_5"] = np.percentile(boot_original, 2.5, axis=0)
    selection["bootstrap_beta_high_97_5"] = np.percentile(boot_original, 97.5, axis=0)
    selection["stable"] = (
        (selection["outer_selection_frequency"] >= 0.70)
        & (selection["bootstrap_selection_frequency"] >= 0.70)
        & (selection["outer_sign_consistency"] >= 0.90)
        & (selection["bootstrap_sign_consistency"] >= 0.90)
    )
    selection = selection.sort_values("standardised_beta", key=np.abs, ascending=False)

    raw_folds.to_csv(output_dir / "raw_nested_cv_folds.csv", index=False)
    raw_predictions.to_csv(output_dir / "raw_nested_cv_predictions.csv", index=False)
    clr_folds.to_csv(output_dir / "clr_nested_cv_folds.csv", index=False)
    clr_predictions.to_csv(output_dir / "clr_nested_cv_predictions.csv", index=False)
    metric_summary(raw_folds).to_csv(output_dir / "raw_nested_cv_summary.csv", index=False)
    metric_summary(clr_folds).to_csv(output_dir / "clr_nested_cv_summary.csv", index=False)
    tuning.to_csv(output_dir / "raw_full_data_inner_cv_grid.csv", index=False)
    selection.to_csv(output_dir / "semantic_element_weights.csv", index=False)
    stable = selection[selection["stable"]]
    stable.to_csv(output_dir / "stable_semantic_element_weights.csv", index=False)

    metadata = {
        "run_dir": str(run_dir),
        "target": args.target,
        "n_images": int(len(table)),
        "cities": {city: int((cities == city).sum()) for city in sorted(set(cities))},
        "feature_count_before_fold_filtering": len(feature_columns),
        "outer_repeats": args.outer_repeats,
        "outer_folds": 5,
        "inner_folds": 5,
        "l1_ratio_grid": list(L1_RATIOS),
        "lambda_grid": list(LAMBDAS),
        "prevalence_threshold": PREVALENCE_THRESHOLD,
        "bootstrap_resamples": args.bootstrap,
        "final_raw_l1_ratio": final_l1,
        "final_raw_lambda": final_lambda,
        "stable_feature_count": int(len(stable)),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    raw = metric_summary(raw_folds).set_index("metric")["mean"].to_dict()
    clr_metrics = metric_summary(clr_folds).set_index("metric")["mean"].to_dict()
    lines = [
        "# VP pollution Elastic Net results",
        "",
        f"- Target: `{args.target}`; sample: {len(table)} images across {len(set(cities))} cities "
        f"({min((cities == city).sum() for city in set(cities))}–{max((cities == city).sum() for city in set(cities))} images per city).",
        f"- Primary raw-share model: repeated nested CV ({args.outer_repeats} × 5 outer folds); city-adjusted within folds.",
        f"- Raw model outer-CV: MAE {raw['mae']:.3f}, RMSE {raw['rmse']:.3f}, R² {raw['r2']:.3f}, Spearman {raw['spearman']:.3f}.",
        f"- City-only baseline: MAE {raw['city_baseline_mae']:.3f}, RMSE {raw['city_baseline_rmse']:.3f}, R² {raw['city_baseline_r2']:.3f}.",
        f"- CLR sensitivity outer-CV: MAE {clr_metrics['mae']:.3f}, RMSE {clr_metrics['rmse']:.3f}, R² {clr_metrics['r2']:.3f}, Spearman {clr_metrics['spearman']:.3f}.",
        f"- Final raw model inner-CV choice: l1_ratio={final_l1:.2f}, lambda={final_lambda:.6g}.",
        f"- Stable elements under the pre-registered rule: {len(stable)}.",
        "",
        "Coefficients are conditional within-city associations, not causal effects. See the logbook for the interpretation rule.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"Completed Elastic Net analysis: {output_dir}")


if __name__ == "__main__":
    main()
