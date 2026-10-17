#!/usr/bin/env python3
"""Exploratory pooled Elastic Net for semantic shares and VP pollution.

This intentionally treats all images as one study population.  It does not
remove, control for, or test generalisation across the recorded city labels.
The output is therefore an exploratory *global association* model, not a
within-city or causal analysis.  See VP_POLLUTION_ELASTIC_NET_LOGBOOK.md.
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

# A ridge-leaning grid deliberately keeps correlated visual elements together
# rather than asking lasso to choose one arbitrary representative.
L1_RATIOS = (0.10, 0.30, 0.50, 0.70)
LAMBDAS = tuple(np.logspace(-4.0, 0.2, 30))
PREVALENCE_THRESHOLD = 0.10


def parse_args() -> argparse.Namespace:
    root = Path("/home/colin/GUVP_pairwise_ranking")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path,
        default=root / "output/semantic_features_finetune136k_core10_grouped",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "output/vp_pollution_elastic_net_pooled_136k_core10_grouped",
    )
    parser.add_argument("--target", default="visual_pollution_score_0_10")
    parser.add_argument("--outer-repeats", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261004)
    return parser.parse_args()


def make_strata(y: np.ndarray, bins: int = 5) -> np.ndarray:
    """Global VP-score quintiles; no city variable participates here."""
    ranks = pd.Series(y).rank(method="first")
    return pd.qcut(ranks, q=bins, labels=False).astype(str).to_numpy()


def clr(values: np.ndarray, zero: float = 1e-6) -> np.ndarray:
    """Multiplicative zero replacement, then centred log-ratio transform."""
    x = np.maximum(values, zero)
    x = x / x.sum(axis=1, keepdims=True)
    logged = np.log(x)
    return logged - logged.mean(axis=1, keepdims=True)


@dataclass
class PreparedData:
    keep: np.ndarray
    centre_mean: np.ndarray
    centre_scale: np.ndarray
    kind: str

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        raw = x[:, self.keep]
        source = clr(raw) if self.kind == "clr" else raw
        return (source - self.centre_mean) / self.centre_scale


@dataclass
class FittedModel:
    prepared: PreparedData
    estimator: ElasticNet

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.estimator.predict(self.prepared.transform_x(x))

    def coefficients(self, p: int) -> tuple[np.ndarray, np.ndarray]:
        standardised = np.zeros(p)
        standardised[self.prepared.keep] = self.estimator.coef_
        original = np.zeros(p)
        if self.prepared.kind == "raw":
            original[self.prepared.keep] = self.estimator.coef_ / self.prepared.centre_scale
        return standardised, original


def prepare_and_fit(
    x: np.ndarray, y: np.ndarray, l1_ratio: float, regularisation: float, kind: str,
) -> FittedModel:
    prevalence = (x > 0).mean(axis=0)
    variance = x.var(axis=0)
    keep = np.flatnonzero((prevalence >= PREVALENCE_THRESHOLD) & (variance > 1e-12))
    if len(keep) < 2:
        raise RuntimeError("Fewer than two non-degenerate semantic features in a training fold")
    source = x[:, keep]
    source = clr(source) if kind == "clr" else source
    mean = source.mean(axis=0)
    scale = source.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (source - mean) / scale
    estimator = ElasticNet(
        alpha=regularisation, l1_ratio=l1_ratio, max_iter=30_000, tol=1e-5,
        selection="cyclic", random_state=0,
    )
    estimator.fit(z, y)
    return FittedModel(PreparedData(keep, mean, scale, kind), estimator)


def select_hyperparameters(
    x: np.ndarray, y: np.ndarray, seed: int, kind: str,
) -> tuple[float, float, pd.DataFrame]:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    strata = make_strata(y)
    losses: dict[tuple[float, float], list[float]] = {
        (l1, lam): [] for l1 in L1_RATIOS for lam in LAMBDAS
    }
    for train_idx, test_idx in splitter.split(x, strata):
        for l1 in L1_RATIOS:
            for lam in LAMBDAS:
                fitted = prepare_and_fit(x[train_idx], y[train_idx], l1, lam, kind)
                losses[(l1, lam)].append(
                    mean_squared_error(y[test_idx], fitted.predict(x[test_idx]))
                )
    table = pd.DataFrame([
        {
            "l1_ratio": l1, "lambda": lam, "mean_mse": float(np.mean(values)),
            "se_mse": float(np.std(values, ddof=1) / np.sqrt(len(values))),
        }
        for (l1, lam), values in losses.items()
    ]).sort_values("mean_mse")
    # Exploratory lambda-min rule: no one-SE inflation toward a null model.
    chosen = table.iloc[0]
    return float(chosen["l1_ratio"]), float(chosen["lambda"]), table


def evaluate_nested_cv(
    x: np.ndarray, y: np.ndarray, feature_columns: list[str], repeats: int, seed: int, kind: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    folds: list[dict] = []
    predictions: list[dict] = []
    all_standardised: list[np.ndarray] = []
    all_original: list[np.ndarray] = []
    strata = make_strata(y)
    for repeat in range(repeats):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + repeat)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x, strata), start=1):
            l1, lam, _ = select_hyperparameters(
                x[train_idx], y[train_idx], seed + 10_000 + repeat * 10 + fold, kind,
            )
            fitted = prepare_and_fit(x[train_idx], y[train_idx], l1, lam, kind)
            prediction = fitted.predict(x[test_idx])
            baseline = np.full(len(test_idx), y[train_idx].mean())
            std_coef, raw_coef = fitted.coefficients(len(feature_columns))
            all_standardised.append(std_coef)
            all_original.append(raw_coef)
            folds.append({
                "model": kind, "repeat": repeat + 1, "fold": fold,
                "n_train": len(train_idx), "n_test": len(test_idx),
                "n_features_after_filter": len(fitted.prepared.keep),
                "l1_ratio": l1, "lambda": lam,
                "mae": mean_absolute_error(y[test_idx], prediction),
                "rmse": float(np.sqrt(mean_squared_error(y[test_idx], prediction))),
                "r2": r2_score(y[test_idx], prediction),
                "spearman": pd.Series(y[test_idx]).corr(pd.Series(prediction), method="spearman"),
                "mean_baseline_mae": mean_absolute_error(y[test_idx], baseline),
                "mean_baseline_rmse": float(np.sqrt(mean_squared_error(y[test_idx], baseline))),
                "mean_baseline_r2": r2_score(y[test_idx], baseline),
            })
            for pos, original_idx in enumerate(test_idx):
                predictions.append({
                    "model": kind, "repeat": repeat + 1, "fold": fold,
                    "row_index": int(original_idx), "observed_vp_score": y[original_idx],
                    "predicted_vp_score": prediction[pos], "mean_baseline_prediction": baseline[pos],
                })
            print(f"[{kind}] repeat {repeat + 1}/{repeats}, fold {fold}/5: lambda={lam:.4g}, l1={l1:.2f}", flush=True)
    return pd.DataFrame(folds), pd.DataFrame(predictions), np.vstack(all_standardised), np.vstack(all_original)


def bootstrap_coefficients(
    x: np.ndarray, y: np.ndarray, feature_count: int, l1: float, lam: float, count: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    standardised: list[np.ndarray] = []
    original: list[np.ndarray] = []
    for number in range(count):
        indices = rng.choice(len(y), size=len(y), replace=True)
        fitted = prepare_and_fit(x[indices], y[indices], l1, lam, "raw")
        std_coef, raw_coef = fitted.coefficients(feature_count)
        standardised.append(std_coef)
        original.append(raw_coef)
        if (number + 1) % 25 == 0 or number + 1 == count:
            print(f"[bootstrap] {number + 1}/{count}", flush=True)
    return np.vstack(standardised), np.vstack(original)


def metric_summary(folds: pd.DataFrame) -> pd.DataFrame:
    fields = ["mae", "rmse", "r2", "spearman", "mean_baseline_mae", "mean_baseline_rmse", "mean_baseline_r2"]
    return pd.DataFrame([
        {"metric": field, "mean": folds[field].mean(), "sd_across_outer_folds": folds[field].std(ddof=1)}
        for field in fields
    ])


def main() -> None:
    args = parse_args()
    run_dir, output_dir = args.run_dir.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(run_dir / "elastic_net_ready.csv")
    schema = pd.read_csv(run_dir / "elastic_net_feature_schema.csv")
    feature_columns = schema["feature_column"].tolist()
    if len(table) != 250 or args.target not in table.columns:
        raise RuntimeError("Expected 250 rows and the requested VP target")
    missing = [column for column in feature_columns if column not in table.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns: {missing[:3]}")
    x, y = table[feature_columns].to_numpy(float), table[args.target].to_numpy(float)
    if not np.allclose(x.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Semantic shares do not close to one")

    raw_folds, raw_predictions, outer_std, _ = evaluate_nested_cv(
        x, y, feature_columns, args.outer_repeats, args.seed, "raw",
    )
    clr_folds, clr_predictions, _, _ = evaluate_nested_cv(
        x, y, feature_columns, args.outer_repeats, args.seed + 50_000, "clr",
    )
    final_l1, final_lambda, tuning = select_hyperparameters(x, y, args.seed + 90_000, "raw")
    final_model = prepare_and_fit(x, y, final_l1, final_lambda, "raw")
    final_std, final_original = final_model.coefficients(len(feature_columns))
    boot_std, boot_original = bootstrap_coefficients(
        x, y, len(feature_columns), final_l1, final_lambda, args.bootstrap, args.seed + 100_000,
    )

    outer_selected, boot_selected = np.abs(outer_std) > 1e-10, np.abs(boot_std) > 1e-10
    selection = schema.copy()
    selection["standardised_beta"] = final_std
    selection["beta_per_full_share"] = final_original
    selection["beta_per_10_percentage_points"] = final_original * 0.10
    selection["outer_selection_frequency"] = outer_selected.mean(axis=0)
    n_outer = outer_selected.sum(axis=0)
    selection["outer_sign_consistency"] = np.divide(
        np.abs(np.sign(outer_std).sum(axis=0)), n_outer, out=np.zeros_like(n_outer, dtype=float), where=n_outer > 0,
    )
    selection["bootstrap_selection_frequency"] = boot_selected.mean(axis=0)
    n_boot = boot_selected.sum(axis=0)
    selection["bootstrap_sign_consistency"] = np.divide(
        np.abs(np.sign(boot_std).sum(axis=0)), n_boot, out=np.zeros_like(n_boot, dtype=float), where=n_boot > 0,
    )
    selection["bootstrap_beta_low_2_5"] = np.percentile(boot_original, 2.5, axis=0)
    selection["bootstrap_beta_high_97_5"] = np.percentile(boot_original, 97.5, axis=0)
    # Relaxed exploratory screen, intentionally weaker than the confirmatory run.
    selection["exploratory_stable"] = (
        (selection["outer_selection_frequency"] >= 0.30)
        & (selection["bootstrap_selection_frequency"] >= 0.50)
        & (selection["outer_sign_consistency"] >= 0.70)
        & (selection["bootstrap_sign_consistency"] >= 0.70)
    )
    selection = selection.sort_values("standardised_beta", key=np.abs, ascending=False)
    screened = selection[selection["exploratory_stable"]]

    raw_folds.to_csv(output_dir / "raw_pooled_nested_cv_folds.csv", index=False)
    raw_predictions.to_csv(output_dir / "raw_pooled_nested_cv_predictions.csv", index=False)
    clr_folds.to_csv(output_dir / "clr_pooled_nested_cv_folds.csv", index=False)
    clr_predictions.to_csv(output_dir / "clr_pooled_nested_cv_predictions.csv", index=False)
    metric_summary(raw_folds).to_csv(output_dir / "raw_pooled_nested_cv_summary.csv", index=False)
    metric_summary(clr_folds).to_csv(output_dir / "clr_pooled_nested_cv_summary.csv", index=False)
    tuning.to_csv(output_dir / "raw_pooled_full_data_cv_grid.csv", index=False)
    selection.to_csv(output_dir / "semantic_element_weights.csv", index=False)
    screened.to_csv(output_dir / "exploratory_screened_semantic_element_weights.csv", index=False)

    raw = metric_summary(raw_folds).set_index("metric")["mean"].to_dict()
    clr_metrics = metric_summary(clr_folds).set_index("metric")["mean"].to_dict()
    metadata = {
        "analysis_scope": "pooled single-population exploratory association; city labels unused",
        "run_dir": str(run_dir), "target": args.target, "n_images": int(len(table)),
        "feature_count_before_fold_filtering": len(feature_columns),
        "outer_repeats": args.outer_repeats, "outer_folds": 5, "inner_folds": 5,
        "l1_ratio_grid": list(L1_RATIOS), "lambda_grid": list(LAMBDAS),
        "prevalence_threshold": PREVALENCE_THRESHOLD, "bootstrap_resamples": args.bootstrap,
        "final_raw_l1_ratio": final_l1, "final_raw_lambda": final_lambda,
        "exploratory_screened_feature_count": int(len(screened)),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    report = [
        "# VP pollution pooled Elastic Net results", "",
        f"- Target: `{args.target}`; sample: {len(table)} images treated as one pooled study population.",
        "- City labels were deliberately not used for adjustment, splitting, baselines, or inference.",
        "- Primary model: raw semantic shares; 10 × 5 nested CV with global VP-score quintile folds and exploratory lambda-min selection.",
        f"- Raw model outer-CV: MAE {raw['mae']:.3f}, RMSE {raw['rmse']:.3f}, R² {raw['r2']:.3f}, Spearman {raw['spearman']:.3f}.",
        f"- Global-mean baseline: MAE {raw['mean_baseline_mae']:.3f}, RMSE {raw['mean_baseline_rmse']:.3f}, R² {raw['mean_baseline_r2']:.3f}.",
        f"- CLR sensitivity: MAE {clr_metrics['mae']:.3f}, RMSE {clr_metrics['rmse']:.3f}, R² {clr_metrics['r2']:.3f}, Spearman {clr_metrics['spearman']:.3f}.",
        f"- Final raw model lambda-min choice: l1_ratio={final_l1:.2f}, lambda={final_lambda:.6g}.",
        f"- Elements passing the relaxed exploratory stability screen: {len(screened)}.", "",
        "Positive beta means that a larger element share is associated with a higher VP score in this pooled dataset; negative beta means the reverse.",
        "These are exploratory associations, not causal effects or evidence of generalisation to other cities/datasets.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n")
    print(f"Completed pooled Elastic Net analysis: {output_dir}")


if __name__ == "__main__":
    main()
