#!/usr/bin/env python3
"""Nested city-held-out comparison of frozen DINOv3 Ridge and pooled Elastic Net."""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "output/vit_vp_1000_v1"
SEEDS = (42, 137, 2026)
RIDGE_ALPHAS = np.logspace(-3, 6, 10)
EN_ALPHAS = np.logspace(-3, 0.5, 12)
EN_L1 = (0.1, 0.4, 0.7, 0.9, 1.0)


def read_fold_ids(path: Path, repeat: int, outer_fold: int, inner_fold: int | None = None) -> tuple[list[str], list[str]]:
    frame = pd.read_csv(path)
    mask = (frame.repeat == repeat) & (frame.outer_fold == outer_fold)
    if inner_fold is not None:
        mask &= frame.inner_fold == inner_fold
    selected = frame.loc[mask]
    train = selected.loc[selected.partition == "train", "city"].astype(str).tolist()
    valid_partition = "test" if inner_fold is None else "validation"
    valid = selected.loc[selected.partition == valid_partition, "city"].astype(str).tolist()
    return train, valid


def scaled_fit_predict(estimator, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    test_scaled = scaler.transform(x_test)
    estimator.fit(train_scaled, y_train)
    return estimator.predict(test_scaled)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rho = float(spearmanr(y, pred).statistic) if np.ptp(pred) > 1e-12 else float("nan")
    corr = float(pearsonr(y, pred).statistic) if np.ptp(pred) > 1e-12 else float("nan")
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "spearman": rho,
        "pearson": corr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT)
    parser.add_argument("--output-dir", type=Path, default=AUDIT / "frozen_ridge")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.repeats > len(SEEDS):
        raise ValueError(f"repeats must be between 1 and {len(SEEDS)}")

    images = pd.read_csv(args.audit_dir / "image_manifest.csv")
    y_map = images.set_index("image_id").target_visual_pollution_score_0_10.astype(float)
    city_map = images.set_index("image_id").city.astype(str)
    feature_files = {
        384: args.audit_dir / "features/dinov3_vitb16_lvd1689m_edge384.npz",
        512: args.audit_dir / "features/dinov3_vitb16_lvd1689m_edge512.npz",
    }
    features: dict[tuple[int, str], tuple[list[str], np.ndarray]] = {}
    for edge, path in feature_files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = np.load(path)
        ids = payload["image_ids"].astype(str).tolist()
        if set(ids) != set(y_map.index.astype(str)) or len(ids) != len(set(ids)):
            raise ValueError(f"Feature ID mismatch in {path}")
        order = {image_id: i for i, image_id in enumerate(ids)}
        for name in ("cls", "patch_mean", "cls_patch"):
            matrix = payload[name][[order[image_id] for image_id in y_map.index.astype(str)]]
            features[(edge, name)] = (y_map.index.astype(str).tolist(), matrix)
    seg = pd.read_csv(ROOT / "output/segmentation_expansion_1000/elastic_net_ready.csv")
    seg_ids = seg.image_id.astype(str).str.replace(r"\.[^.]+$", "", regex=True)
    if seg_ids.duplicated().any() or set(seg_ids) != set(y_map.index.astype(str)):
        raise ValueError("Segmentation feature IDs do not match frozen 1000-image target IDs")
    ratio_cols = [col for col in seg.columns if col.startswith("feature_ratio_")]
    seg_row_by_id = dict(zip(seg_ids, range(len(seg_ids))))
    seg_matrix_unordered = seg[ratio_cols].to_numpy(dtype=np.float64)
    seg_matrix = seg_matrix_unordered[[seg_row_by_id[image_id] for image_id in y_map.index.astype(str)]]

    outer_path = args.audit_dir / "splits/unseen_city_outer.csv"
    inner_path = args.audit_dir / "splits/unseen_city_inner.csv"
    oof_rows: list[dict[str, object]] = []
    search_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    y_series = y_map
    ids_all = y_series.index.astype(str).tolist()
    city_array = city_map.loc[ids_all].to_numpy()
    y_all = y_series.loc[ids_all].to_numpy(dtype=np.float64)

    for repeat in range(1, args.repeats + 1):
        for outer_fold in range(5):
            train_cities, test_cities = read_fold_ids(outer_path, repeat, outer_fold)
            train_idx = np.flatnonzero(np.isin(city_array, train_cities))
            test_idx = np.flatnonzero(np.isin(city_array, test_cities))
            if len(train_idx) != 800 or len(test_idx) != 200:
                raise ValueError(f"Unexpected fold size: {len(train_idx)} train / {len(test_idx)} test")

            inner_splits = []
            for inner_fold in range(4):
                tr_cities, val_cities = read_fold_ids(inner_path, repeat, outer_fold, inner_fold)
                tr_idx = np.flatnonzero(np.isin(city_array, tr_cities))
                val_idx = np.flatnonzero(np.isin(city_array, val_cities))
                if set(city_array[tr_idx]) & set(city_array[val_idx]):
                    raise AssertionError("Inner train/validation city leakage")
                inner_splits.append((tr_idx, val_idx))

            candidates: list[tuple[str, tuple[int, str] | None, float, float]] = []
            for feature_key in features:
                for alpha in RIDGE_ALPHAS:
                    candidates.append(("dinov3_ridge", feature_key, float(alpha), np.nan))
            for alpha in EN_ALPHAS:
                for l1_ratio in EN_L1:
                    candidates.append(("semantic_elastic_net", None, float(alpha), float(l1_ratio)))

            candidate_errors: dict[int, list[float]] = defaultdict(list)
            for tr_idx, val_idx in inner_splits:
                for candidate_idx, (family, feature_key, alpha, l1_ratio) in enumerate(candidates):
                    if family == "dinov3_ridge":
                        _, matrix = features[feature_key]
                        x = matrix
                        estimator = Ridge(alpha=alpha, solver="lsqr", tol=1e-4)
                    else:
                        x = seg_matrix
                        estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000, tol=1e-5, selection="cyclic")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        pred = scaled_fit_predict(estimator, x[tr_idx], y_all[tr_idx], x[val_idx])
                    candidate_errors[candidate_idx].append(mean_absolute_error(y_all[val_idx], pred))

            mean_errors = {idx: float(np.mean(values)) for idx, values in candidate_errors.items()}
            winners: dict[str, int] = {}
            for family in ("dinov3_ridge", "semantic_elastic_net"):
                idxs = [i for i, c in enumerate(candidates) if c[0] == family]
                winners[family] = min(idxs, key=lambda i: (mean_errors[i], i))
            for family in winners:
                i = winners[family]
                cand = candidates[i]
                search_rows.append({
                    "repeat": repeat, "seed": SEEDS[repeat - 1], "outer_fold": outer_fold,
                    "model": family, "inner_mae": mean_errors[i],
                    "edge": cand[1][0] if cand[1] else "", "feature_view": cand[1][1] if cand[1] else "",
                    "alpha": cand[2], "l1_ratio": cand[3], "candidate_count": sum(c[0] == family for c in candidates),
                })

            baseline = np.full(len(test_idx), np.mean(y_all[train_idx]), dtype=np.float64)
            for i, original_idx in enumerate(test_idx):
                oof_rows.append({"repeat": repeat, "outer_fold": outer_fold, "image_id": ids_all[original_idx], "city": city_array[original_idx], "y_true": y_all[original_idx], "model": "train_mean", "y_pred": baseline[i], "selected_config": "constant"})

            for family, candidate_idx in winners.items():
                _, feature_key, alpha, l1_ratio = candidates[candidate_idx]
                if family == "dinov3_ridge":
                    _, matrix = features[feature_key]
                    x = matrix
                    estimator = Ridge(alpha=alpha, solver="lsqr", tol=1e-4)
                    config = f"edge={feature_key[0]};view={feature_key[1]};alpha={alpha:.8g}"
                else:
                    x = seg_matrix
                    estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=20000, tol=1e-5, selection="cyclic")
                    config = f"alpha={alpha:.8g};l1_ratio={l1_ratio:.3f}"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    pred = scaled_fit_predict(estimator, x[train_idx], y_all[train_idx], x[test_idx])
                for i, original_idx in enumerate(test_idx):
                    oof_rows.append({"repeat": repeat, "outer_fold": outer_fold, "image_id": ids_all[original_idx], "city": city_array[original_idx], "y_true": y_all[original_idx], "model": family, "y_pred": float(pred[i]), "selected_config": config})

            for family in ("train_mean", "dinov3_ridge", "semantic_elastic_net"):
                subset = [row for row in oof_rows if row["repeat"] == repeat and row["outer_fold"] == outer_fold and row["model"] == family]
                fold_metrics = metrics(np.asarray([r["y_true"] for r in subset]), np.asarray([r["y_pred"] for r in subset]))
                fold_rows.append({"repeat": repeat, "outer_fold": outer_fold, "model": family, **fold_metrics})
            print(f"repeat {repeat}/{args.repeats} outer fold {outer_fold + 1}/5 complete", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof = pd.DataFrame(oof_rows)
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(search_rows).to_csv(args.output_dir / "inner_selected_configs.csv", index=False)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(args.output_dir / "outer_fold_metrics.csv", index=False)
    per_repeat = []
    for repeat, group in oof.groupby("repeat"):
        for family, frame in group.groupby("model"):
            per_repeat.append({"repeat": int(repeat), "model": family, **metrics(frame.y_true.to_numpy(), frame.y_pred.to_numpy())})
    repeat_df = pd.DataFrame(per_repeat)
    repeat_df.to_csv(args.output_dir / "per_repeat_metrics.csv", index=False)
    summary = repeat_df.groupby("model").agg({col: ["mean", "std"] for col in ("mae", "rmse", "r2", "spearman", "pearson")})
    summary.columns = ["_".join(col) for col in summary.columns]
    summary.reset_index().to_csv(args.output_dir / "summary.csv", index=False)

    city_metrics = []
    for (repeat, model, city), frame in oof.groupby(["repeat", "model", "city"]):
        city_metrics.append({"repeat": int(repeat), "model": model, "city": city, **metrics(frame.y_true.to_numpy(), frame.y_pred.to_numpy())})
    pd.DataFrame(city_metrics).to_csv(args.output_dir / "metrics_by_city.csv", index=False)
    image_oof = oof.groupby(["image_id", "city", "model"], as_index=False).agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))
    city_blocks = {
        model: {city: block for city, block in frame.groupby("city")}
        for model, frame in image_oof.groupby("model")
    }
    rng = np.random.default_rng(20261019)
    city_names = sorted(city_blocks["dinov3_ridge"])
    comparisons = [("dinov3_ridge", "semantic_elastic_net"), ("dinov3_ridge", "train_mean")]
    bootstrap_rows = []
    for model_a, model_b in comparisons:
        deltas = {metric: [] for metric in ("mae", "rmse", "spearman")}
        for _ in range(2000):
            sampled = rng.choice(city_names, size=len(city_names), replace=True)
            ys: list[np.ndarray] = []
            preds_a: list[np.ndarray] = []
            preds_b: list[np.ndarray] = []
            for city in sampled:
                block_a, block_b = city_blocks[model_a][city], city_blocks[model_b][city]
                ys.append(block_a.y_true.to_numpy())
                preds_a.append(block_a.y_pred.to_numpy())
                preds_b.append(block_b.y_pred.to_numpy())
            y_boot = np.concatenate(ys)
            a_boot = np.concatenate(preds_a)
            b_boot = np.concatenate(preds_b)
            deltas["mae"].append(float(mean_absolute_error(y_boot, a_boot) - mean_absolute_error(y_boot, b_boot)))
            deltas["rmse"].append(float(np.sqrt(mean_squared_error(y_boot, a_boot)) - np.sqrt(mean_squared_error(y_boot, b_boot))))
            deltas["spearman"].append(float(spearmanr(y_boot, a_boot).statistic - spearmanr(y_boot, b_boot).statistic))
        for metric, values in deltas.items():
            bootstrap_rows.append({
                "comparison": f"{model_a}_minus_{model_b}",
                "metric": metric,
                "point_estimate": float(np.mean(values)),
                "ci95_lower": float(np.quantile(values, 0.025)),
                "ci95_upper": float(np.quantile(values, 0.975)),
                "bootstrap_unit": "city",
                "bootstrap_replicates": 2000,
                "seed": 20261019,
            })
    pd.DataFrame(bootstrap_rows).to_csv(args.output_dir / "city_cluster_bootstrap.csv", index=False)
    metadata = {
        "target": "visual_pollution_score_0_10",
        "outer_protocol": "5-fold city-grouped CV, each held-out city predicted once per repeat",
        "inner_protocol": "4-fold city-grouped CV on outer-train cities; tune all candidates by mean inner MAE",
        "repeats": args.repeats,
        "seeds": list(SEEDS[: args.repeats]),
        "primary_metric": "MAE",
        "ridge_candidates_per_outer_fold": len(features) * len(RIDGE_ALPHAS),
        "elastic_net_candidates_per_outer_fold": len(EN_ALPHAS) * len(EN_L1),
        "feature_matrix_dimensions": {f"{edge}_{view}": int(matrix.shape[1]) for (edge, view), (_, matrix) in features.items()},
        "segmentation_feature_count": len(ratio_cols),
        "prediction_rows": len(oof),
        "paired_oof_predictions_averaged_over_repeats_for_cluster_bootstrap": len(image_oof),
        "cluster_bootstrap_replicates": 2000,
        "label_scope_warning": "Fixed global TrueSkill teacher labels are statistically dependent through the shared comparison graph; see the experiment plan.",
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(summary.reset_index().to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
