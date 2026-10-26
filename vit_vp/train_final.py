#!/usr/bin/env python3
"""Select a frozen DINOv3 Ridge configuration and fit the final 1000-image head."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "output/vit_vp_1000_v1"
ALPHAS = np.logspace(-3, 6, 10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT)
    parser.add_argument("--output-dir", type=Path, default=AUDIT / "deployment")
    args = parser.parse_args()
    manifest = pd.read_csv(args.audit_dir / "image_manifest.csv")
    ids = manifest.image_id.astype(str).tolist()
    y = manifest.target_visual_pollution_score_0_10.to_numpy(dtype=np.float64)
    cities = manifest.city.astype(str).to_numpy()
    candidate_scores: list[dict[str, object]] = []
    feature_cache: dict[tuple[int, str], np.ndarray] = {}
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=2026).split(np.zeros(len(ids)), y, groups=cities))
    for edge in (384, 512):
        path = args.audit_dir / f"features/dinov3_vitb16_lvd1689m_edge{edge}.npz"
        payload = np.load(path)
        feature_ids = payload["image_ids"].astype(str).tolist()
        order = {image_id: i for i, image_id in enumerate(feature_ids)}
        if len(order) != len(ids) or set(order) != set(ids):
            raise ValueError(f"Feature IDs do not match image manifest: {path}")
        for view in ("cls", "patch_mean", "cls_patch"):
            feature_cache[(edge, view)] = payload[view][[order[image_id] for image_id in ids]]

    for key, x in feature_cache.items():
        for alpha in ALPHAS:
            errors = []
            for train_idx, val_idx in folds:
                scaler = StandardScaler().fit(x[train_idx])
                model = Ridge(alpha=float(alpha), solver="lsqr", tol=1e-4)
                model.fit(scaler.transform(x[train_idx]), y[train_idx])
                prediction = model.predict(scaler.transform(x[val_idx]))
                errors.append(mean_absolute_error(y[val_idx], prediction))
            candidate_scores.append({
                "edge": key[0], "feature_view": key[1], "alpha": float(alpha),
                "cv_mae_mean": float(np.mean(errors)), "cv_mae_sd": float(np.std(errors, ddof=1)),
                "fold_mae": ";".join(f"{v:.8f}" for v in errors),
            })
    candidates = pd.DataFrame(candidate_scores).sort_values(["cv_mae_mean", "edge", "feature_view", "alpha"])
    best = candidates.iloc[0]
    edge, view, alpha = int(best.edge), str(best.feature_view), float(best.alpha)
    x = feature_cache[(edge, view)]
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=alpha, solver="lsqr", tol=1e-4).fit(scaler.transform(x), y)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output_dir / "final_group_cv_candidates.csv", index=False)
    np.savez_compressed(
        args.output_dir / "ridge_head.npz",
        scaler_mean=scaler.mean_.astype(np.float64),
        scaler_scale=scaler.scale_.astype(np.float64),
        coefficient=model.coef_.astype(np.float64),
    )
    metadata = {
        "model": "DINOv3 ViT-B/16 LVD-1689M frozen encoder + Ridge regression head",
        "target": "visual_pollution_score_0_10",
        "target_direction": "larger means more visual pollution",
        "feature_max_edge": edge,
        "feature_view": view,
        "feature_dimension": int(x.shape[1]),
        "ridge_alpha": alpha,
        "intercept": float(model.intercept_),
        "final_fit_images": len(ids),
        "final_fit_cities": int(len(set(cities))),
        "selection": "5-fold GroupKFold by city on all images; choose the lowest mean validation MAE",
        "selection_cv_mae": float(best.cv_mae_mean),
        "selection_cv_mae_sd_across_folds": float(best.cv_mae_sd),
        "fit_note": "This final all-data model is for inference; its in-sample metrics are not a generalization estimate. Use nested out-of-fold results.",
        "feature_checkpoint": str((ROOT / "models/checkpoints/dinov3_vitb.safetensors").resolve()),
        "feature_checkpoint_sha256": "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b",
        "prediction_units": "frozen dataset VP 0-10 scale; raw regression output is not clipped",
        "source_manifest_sha256": json.loads((args.audit_dir / "dataset_audit.json").read_text())["input_sha256"]["manifest"],
        "source_scores_sha256": json.loads((args.audit_dir / "dataset_audit.json").read_text())["input_sha256"]["scores"],
    }
    (args.output_dir / "model.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(candidates.head(10).to_string(index=False))
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
