#!/usr/bin/env python3
"""Nested within-city spatial evaluation of frozen DINOv3 Ridge and semantic Elastic Net."""
from __future__ import annotations

import argparse
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


def split_ids(frame: pd.DataFrame, repeat: int, outer_fold: int, inner_fold: int | None = None):
    mask = (frame.repeat == repeat) & (frame.outer_fold == outer_fold)
    if inner_fold is not None:
        mask &= frame.inner_fold == inner_fold
    subset = frame.loc[mask]
    train = subset.loc[subset.partition == "train", "image_id"].astype(str).tolist()
    valid_role = "test" if inner_fold is None else "validation"
    valid = subset.loc[subset.partition == valid_role, "image_id"].astype(str).tolist()
    train_groups = set(subset.loc[subset.partition == "train", "spatial_group"].astype(str))
    valid_groups = set(subset.loc[subset.partition == valid_role, "spatial_group"].astype(str))
    if train_groups & valid_groups:
        raise ValueError(f"Spatial group leakage in repeat {repeat}, outer {outer_fold}, inner {inner_fold}")
    return train, valid


def fit_predict(estimator, x_train, y_train, x_eval):
    scaler = StandardScaler()
    xt = scaler.fit_transform(x_train)
    xe = scaler.transform(x_eval)
    estimator.fit(xt, y_train)
    return estimator.predict(xe)


def metrics(y, pred):
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "spearman": float(spearmanr(y, pred).statistic) if np.ptp(pred) > 1e-12 else float("nan"),
        "pearson": float(pearsonr(y, pred).statistic) if np.ptp(pred) > 1e-12 else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--protocol", choices=("250m", "buffer100"), default="250m")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    if not 1 <= args.repeats <= len(SEEDS):
        raise ValueError(f"repeats must be between 1 and {len(SEEDS)}")
    if args.output_dir is None:
        args.output_dir = args.audit_dir / ("within_city_spatial_250m" if args.protocol == "250m" else "within_city_spatial_250m_buffer100")

    manifest = pd.read_csv(args.audit_dir / "image_manifest.csv")
    ids = manifest.image_id.astype(str).tolist()
    index = {image_id: i for i, image_id in enumerate(ids)}
    if len(index) != len(ids):
        raise ValueError("Duplicate IDs in image manifest")
    y = manifest.target_visual_pollution_score_0_10.to_numpy(dtype=np.float64)
    cities = manifest.city.astype(str).to_numpy()

    suffix = "" if args.protocol == "250m" else "_buffer100"
    outer = pd.read_csv(args.audit_dir / f"splits/within_city_spatial_250m{suffix}_outer.csv")
    inner = pd.read_csv(args.audit_dir / f"splits/within_city_spatial_250m{suffix}_inner.csv")
    features = {}
    feature_keys = []
    for edge in (384, 512):
        path = args.audit_dir / f"features/dinov3_vitb16_lvd1689m_edge{edge}.npz"
        with np.load(path) as payload:
            feature_ids = payload["image_ids"].astype(str).tolist()
            if set(feature_ids) != set(ids) or len(feature_ids) != len(ids):
                raise ValueError(f"Feature IDs do not match manifest: {path}")
            order = {image_id: i for i, image_id in enumerate(feature_ids)}
            rows = [order[image_id] for image_id in ids]
            for view in ("cls", "patch_mean", "cls_patch"):
                key = (edge, view)
                features[key] = payload[view][rows]
                feature_keys.append(key)

    seg = pd.read_csv(ROOT / "output/segmentation_expansion_1000/elastic_net_ready.csv")
    seg_ids = seg.image_id.astype(str).str.replace(r"\.[^.]+$", "", regex=True)
    if seg_ids.duplicated().any() or set(seg_ids) != set(ids):
        raise ValueError("Segmentation feature IDs do not match VP dataset")
    ratio_cols = [c for c in seg.columns if c.startswith("feature_ratio_")]
    seg_order = dict(zip(seg_ids, range(len(seg_ids))))
    seg_x = seg[ratio_cols].to_numpy(dtype=np.float64)[[seg_order[i] for i in ids]]

    candidates = [("dinov3_ridge", key, float(alpha), np.nan) for key in feature_keys for alpha in RIDGE_ALPHAS]
    candidates += [("semantic_elastic_net", None, float(alpha), float(l1)) for alpha in EN_ALPHAS for l1 in EN_L1]
    oof_rows, search_rows, fold_rows = [], [], []

    for repeat in range(1, args.repeats + 1):
        for outer_fold in range(5):
            tr_ids, te_ids = split_ids(outer, repeat, outer_fold)
            tr_idx = np.array([index[i] for i in tr_ids], dtype=int)
            te_idx = np.array([index[i] for i in te_ids], dtype=int)
            if not tr_ids or not te_ids or set(tr_ids) & set(te_ids):
                raise ValueError("Invalid outer train/test image assignment")
            inner_splits = []
            for inner_fold in range(4):
                itr_ids, val_ids = split_ids(inner, repeat, outer_fold, inner_fold)
                itr_idx = np.array([index[i] for i in itr_ids], dtype=int)
                val_idx = np.array([index[i] for i in val_ids], dtype=int)
                if set(itr_ids) & set(val_ids):
                    raise ValueError("Invalid inner train/validation assignment")
                inner_splits.append((itr_idx, val_idx))

            errors = {ci: [] for ci in range(len(candidates))}
            for itr_idx, val_idx in inner_splits:
                for ci, (family, key, alpha, l1) in enumerate(candidates):
                    if family == "dinov3_ridge":
                        estimator = Ridge(alpha=alpha, solver="lsqr", tol=1e-4)
                        pred = fit_predict(estimator, features[key][itr_idx], y[itr_idx], features[key][val_idx])
                    else:
                        estimator = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=20000, tol=1e-5, selection="cyclic")
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", ConvergenceWarning)
                            pred = fit_predict(estimator, seg_x[itr_idx], y[itr_idx], seg_x[val_idx])
                    errors[ci].append(mean_absolute_error(y[val_idx], pred))

            mean_error = {ci: float(np.mean(values)) for ci, values in errors.items()}
            winners = {}
            for family in ("dinov3_ridge", "semantic_elastic_net"):
                candidates_family = [i for i, c in enumerate(candidates) if c[0] == family]
                winners[family] = min(candidates_family, key=lambda i: (mean_error[i], i))
            for family, ci in winners.items():
                _, key, alpha, l1 = candidates[ci]
                search_rows.append({"repeat": repeat, "seed": SEEDS[repeat - 1], "outer_fold": outer_fold,
                                    "protocol": args.protocol, "model": family, "inner_mae": mean_error[ci],
                                    "edge": key[0] if key else "", "feature_view": key[1] if key else "",
                                    "alpha": alpha, "l1_ratio": l1,
                                    "candidate_count": sum(c[0] == family for c in candidates),
                                    "inner_folds": 4, "n_outer_train": len(tr_idx), "n_outer_test": len(te_idx)})

            city_train_means = {city: float(np.mean(y[tr_idx[cities[tr_idx] == city]])) for city in set(cities[te_idx])}
            preds = {
                "train_mean": np.full(len(te_idx), float(np.mean(y[tr_idx]))),
                "city_mean": np.array([city_train_means[cities[i]] for i in te_idx]),
            }
            for family, ci in winners.items():
                _, key, alpha, l1 = candidates[ci]
                if family == "dinov3_ridge":
                    est = Ridge(alpha=alpha, solver="lsqr", tol=1e-4)
                    preds[family] = fit_predict(est, features[key][tr_idx], y[tr_idx], features[key][te_idx])
                else:
                    est = ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=20000, tol=1e-5, selection="cyclic")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        preds[family] = fit_predict(est, seg_x[tr_idx], y[tr_idx], seg_x[te_idx])
            for model, p in preds.items():
                for j, i in enumerate(te_idx):
                    oof_rows.append({"repeat": repeat, "outer_fold": outer_fold, "image_id": ids[i], "city": cities[i],
                                     "y_true": y[i], "model": model, "y_pred": float(p[j]), "protocol": args.protocol})
                fold_rows.append({"repeat": repeat, "outer_fold": outer_fold, "model": model,
                                  "n_train": len(tr_idx), "n_test": len(te_idx), **metrics(y[te_idx], p)})
            print(f"{args.protocol} repeat {repeat}/{args.repeats} outer fold {outer_fold + 1}/5 complete; train={len(tr_idx)}, test={len(te_idx)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof = pd.DataFrame(oof_rows)
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(search_rows).to_csv(args.output_dir / "inner_selected_configs.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "outer_fold_metrics.csv", index=False)
    repeat_rows = []
    for repeat, frame in oof.groupby("repeat"):
        for model, group in frame.groupby("model"):
            repeat_rows.append({"repeat": int(repeat), "model": model, **metrics(group.y_true.to_numpy(), group.y_pred.to_numpy())})
    repeat_df = pd.DataFrame(repeat_rows)
    repeat_df.to_csv(args.output_dir / "per_repeat_metrics.csv", index=False)
    summary = repeat_df.groupby("model").agg({m: ["mean", "std"] for m in ("mae", "rmse", "r2", "spearman", "pearson")})
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(args.output_dir / "summary.csv", index=False)
    city_rows = []
    for (repeat, model, city), group in oof.groupby(["repeat", "model", "city"]):
        city_rows.append({"repeat": int(repeat), "model": model, "city": city, **metrics(group.y_true.to_numpy(), group.y_pred.to_numpy())})
    pd.DataFrame(city_rows).to_csv(args.output_dir / "metrics_by_city.csv", index=False)

    image_oof = oof.groupby(["image_id", "city", "model"], as_index=False).agg(y_true=("y_true", "mean"), y_pred=("y_pred", "mean"))
    blocks = {model: {city: g for city, g in frame.groupby("city")} for model, frame in image_oof.groupby("model")}
    city_names = sorted(blocks["dinov3_ridge"])
    rng = np.random.default_rng(20261019)
    bootstrap_rows = []
    for other in ("semantic_elastic_net", "train_mean", "city_mean"):
        samples = {metric: [] for metric in ("mae", "rmse", "spearman")}
        for _ in range(args.bootstrap_replicates):
            selected = rng.choice(city_names, size=len(city_names), replace=True)
            yt, pa, pb = [], [], []
            for city in selected:
                a, b = blocks["dinov3_ridge"][city], blocks[other][city]
                yt.append(a.y_true.to_numpy()); pa.append(a.y_pred.to_numpy()); pb.append(b.y_pred.to_numpy())
            yt, pa, pb = np.concatenate(yt), np.concatenate(pa), np.concatenate(pb)
            samples["mae"].append(float(mean_absolute_error(yt, pa) - mean_absolute_error(yt, pb)))
            samples["rmse"].append(float(np.sqrt(mean_squared_error(yt, pa)) - np.sqrt(mean_squared_error(yt, pb))))
            samples["spearman"].append(float(spearmanr(yt, pa).statistic - spearmanr(yt, pb).statistic))
        for metric, values in samples.items():
            bootstrap_rows.append({"comparison": f"dinov3_ridge_minus_{other}", "metric": metric,
                                   "delta_mean_oof": float(np.mean(values)),
                                   "ci_2_5": float(np.quantile(values, .025)), "ci_97_5": float(np.quantile(values, .975)),
                                   "replicates": args.bootstrap_replicates, "cluster_count": len(city_names)})
    pd.DataFrame(bootstrap_rows).to_csv(args.output_dir / "city_cluster_bootstrap.csv", index=False)
    metadata = {"protocol": args.protocol, "grid_size_m": 250, "buffer_m": 100 if args.protocol == "buffer100" else 0,
                "repeats": args.repeats, "seeds": list(SEEDS[:args.repeats]), "outer_folds": 5, "inner_folds": 4,
                "ridge_candidates_per_outer_fold": len(feature_keys) * len(RIDGE_ALPHAS),
                "elastic_net_candidates_per_outer_fold": len(EN_ALPHAS) * len(EN_L1),
                "label_scope_warning": "Fixed global TrueSkill scores share edges across examples; this spatial evaluation does not isolate comparison edges."}
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
