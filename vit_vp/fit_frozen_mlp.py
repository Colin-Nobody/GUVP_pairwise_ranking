#!/usr/bin/env python3
"""Nested city-held-out MLP comparison using cached frozen DINOv3 features."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from extract_features import require_cuda

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "output/vit_vp_1000_v1"
SEEDS = (42, 137, 2026)
LEARNING_RATES = (1e-4, 1e-3)
WEIGHT_DECAYS = (1e-3, 1e-2)
DROPOUTS = (0.2, 0.5)
MAX_EPOCHS = 200
PATIENCE = 20


def fold_cities(path: Path, repeat: int, outer_fold: int, inner_fold: int | None = None):
    frame = pd.read_csv(path)
    mask = (frame.repeat == repeat) & (frame.outer_fold == outer_fold)
    if inner_fold is not None:
        mask &= frame.inner_fold == inner_fold
    part = frame.loc[mask]
    valid_name = "test" if inner_fold is None else "validation"
    return (part.loc[part.partition == "train", "city"].astype(str).tolist(),
            part.loc[part.partition == valid_name, "city"].astype(str).tolist())


def metrics(y, pred):
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "spearman": float(spearmanr(y, pred).statistic),
        "pearson": float(pearsonr(y, pred).statistic),
    }


class Head(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def fit_head(x_train, y_train, x_eval, y_eval, *, dropout, lr, weight_decay, seed, device, max_epochs=MAX_EPOCHS, patience=PATIENCE):
    # Every learned scaling parameter is fitted on this training split only.
    x_scaler = StandardScaler()
    xt = x_scaler.fit_transform(x_train).astype(np.float32)
    xv = x_scaler.transform(x_eval).astype(np.float32)
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train)) or 1.0
    yt = ((y_train - y_mean) / y_std).astype(np.float32)
    yv = ((y_eval - y_mean) / y_std).astype(np.float32)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = Head(xt.shape[1], dropout).to(device)
    tx = torch.from_numpy(xt).to(device)
    ty = torch.from_numpy(yt).to(device)
    vx = torch.from_numpy(xv).to(device)
    vy = torch.from_numpy(yv).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    best_loss = float("inf")
    best_epoch = 1
    best_state = None
    wait = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_fn(model(tx), ty).backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(vx), vy).item())
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("MLP did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(vx).cpu().numpy().astype(np.float64) * y_std + y_mean
    return pred, best_epoch, best_loss


def fit_fixed_epochs(x_train, y_train, x_predict, *, dropout, lr, weight_decay, seed, device, epochs):
    """Refit on the full outer-training set without reading outer-test labels."""
    x_scaler = StandardScaler()
    xt = x_scaler.fit_transform(x_train).astype(np.float32)
    xp = x_scaler.transform(x_predict).astype(np.float32)
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train)) or 1.0
    yt = ((y_train - y_mean) / y_std).astype(np.float32)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = Head(xt.shape[1], dropout).to(device)
    tx = torch.from_numpy(xt).to(device)
    ty = torch.from_numpy(yt).to(device)
    px = torch.from_numpy(xp).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_fn(model(tx), ty).backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = model(px).cpu().numpy().astype(np.float64) * y_std + y_mean
    return pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT)
    parser.add_argument("--output-dir", type=Path, default=AUDIT / "frozen_mlp")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.repeats < 1 or args.repeats > len(SEEDS):
        raise ValueError(f"repeats must be between 1 and {len(SEEDS)}")
    device = require_cuda(torch.device(args.device))

    manifest = pd.read_csv(args.audit_dir / "image_manifest.csv")
    ids = manifest.image_id.astype(str).tolist()
    y = manifest.set_index("image_id").loc[ids].target_visual_pollution_score_0_10.to_numpy(dtype=np.float64)
    cities = manifest.set_index("image_id").loc[ids].city.astype(str).to_numpy()
    feature_specs = []
    x_by_key = {}
    for edge in (384, 512):
        path = args.audit_dir / f"features/dinov3_vitb16_lvd1689m_edge{edge}.npz"
        with np.load(path) as payload:
            feature_ids = payload["image_ids"].astype(str).tolist()
            if len(set(feature_ids)) != len(ids) or set(feature_ids) != set(ids):
                raise ValueError(f"Image IDs in {path} do not match manifest")
            index = {image_id: i for i, image_id in enumerate(feature_ids)}
            order = [index[image_id] for image_id in ids]
            for view in ("cls", "patch_mean", "cls_patch"):
                key = (edge, view)
                x_by_key[key] = payload[view][order].astype(np.float32)
                feature_specs.append(key)

    outer_path = args.audit_dir / "splits/unseen_city_outer.csv"
    inner_path = args.audit_dir / "splits/unseen_city_inner.csv"
    candidates = [(key, dropout, lr, wd) for key in feature_specs for dropout in DROPOUTS for lr in LEARNING_RATES for wd in WEIGHT_DECAYS]
    oof_rows, search_rows, fold_rows = [], [], []
    for repeat in range(1, args.repeats + 1):
        outer_seed = SEEDS[repeat - 1]
        for outer_fold in range(5):
            train_cities, test_cities = fold_cities(outer_path, repeat, outer_fold)
            train_idx = np.flatnonzero(np.isin(cities, train_cities))
            test_idx = np.flatnonzero(np.isin(cities, test_cities))
            if len(train_idx) != 800 or len(test_idx) != 200:
                raise ValueError(f"Unexpected outer sizes: {len(train_idx)} / {len(test_idx)}")
            inner_splits = []
            for inner_fold in range(4):
                tr_cities, val_cities = fold_cities(inner_path, repeat, outer_fold, inner_fold)
                tr_idx = np.flatnonzero(np.isin(cities, tr_cities))
                val_idx = np.flatnonzero(np.isin(cities, val_cities))
                if set(cities[tr_idx]) & set(cities[val_idx]):
                    raise ValueError("Inner city leakage")
                inner_splits.append((tr_idx, val_idx))

            val_mae = {i: [] for i in range(len(candidates))}
            best_epochs = {i: [] for i in range(len(candidates))}
            for inner_i, (tr_idx, val_idx) in enumerate(inner_splits):
                for ci, (key, dropout, lr, wd) in enumerate(candidates):
                    pred, epoch, _ = fit_head(x_by_key[key][tr_idx], y[tr_idx], x_by_key[key][val_idx], y[val_idx],
                                              dropout=dropout, lr=lr, weight_decay=wd, seed=42,
                                              device=device, max_epochs=args.max_epochs)
                    val_mae[ci].append(float(mean_absolute_error(y[val_idx], pred)))
                    best_epochs[ci].append(epoch)
                print(f"repeat {repeat}/{args.repeats} outer {outer_fold + 1}/5 inner {inner_i + 1}/4 complete", flush=True)
            mean_mae = {ci: float(np.mean(vals)) for ci, vals in val_mae.items()}
            selected_i = min(mean_mae, key=lambda ci: (mean_mae[ci], ci))
            key, dropout, lr, wd = candidates[selected_i]
            selected_epochs = max(1, int(np.median(best_epochs[selected_i])))
            search_rows.append({"repeat": repeat, "seed": outer_seed, "outer_fold": outer_fold,
                                "inner_mae": mean_mae[selected_i], "edge": key[0], "feature_view": key[1],
                                "dropout": dropout, "learning_rate": lr, "weight_decay": wd,
                                "selected_epoch_median": selected_epochs, "candidate_count": len(candidates),
                                "inner_fold_count": len(inner_splits)})
            seed_preds = []
            for seed in SEEDS:
                pred = fit_fixed_epochs(x_by_key[key][train_idx], y[train_idx], x_by_key[key][test_idx],
                                        dropout=dropout, lr=lr, weight_decay=wd, seed=seed,
                                        device=device, epochs=selected_epochs)
                seed_preds.append(pred)
            outer_pred = np.mean(seed_preds, axis=0)
            config = f"edge={key[0]};view={key[1]};dropout={dropout};lr={lr};wd={wd};epochs={selected_epochs};seeds={','.join(map(str, SEEDS))}"
            for j, idx in enumerate(test_idx):
                oof_rows.append({"repeat": repeat, "outer_fold": outer_fold, "image_id": ids[idx], "city": cities[idx],
                                 "y_true": y[idx], "model": "dinov3_frozen_mlp", "y_pred": float(outer_pred[j]),
                                 "selected_config": config})
            fold_rows.append({"repeat": repeat, "outer_fold": outer_fold, "model": "dinov3_frozen_mlp", **metrics(y[test_idx], outer_pred)})
            print(f"repeat {repeat}/{args.repeats} outer fold {outer_fold + 1}/5 selected {key[0]} {key[1]} lr={lr:g} dropout={dropout} epochs={selected_epochs}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof = pd.DataFrame(oof_rows)
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(search_rows).to_csv(args.output_dir / "inner_selected_configs.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "outer_fold_metrics.csv", index=False)
    per_repeat = [{"repeat": int(rep), **metrics(frame.y_true.to_numpy(), frame.y_pred.to_numpy())}
                  for rep, frame in oof.groupby("repeat")]
    pd.DataFrame(per_repeat).to_csv(args.output_dir / "per_repeat_metrics.csv", index=False)
    with open(args.output_dir / "run_metadata.json", "w") as f:
        json.dump({"device": str(device), "seeds": list(SEEDS), "repeats": args.repeats,
                   "candidate_count_per_outer_fold": len(candidates), "candidate_views": [f"{e}_{v}" for e, v in feature_specs],
                   "learning_rates": LEARNING_RATES, "weight_decays": WEIGHT_DECAYS, "dropouts": DROPOUTS,
                   "max_epochs": args.max_epochs, "patience": PATIENCE,
                   "inner_selection_metric": "mean validation MAE", "fit_batching": "full batch",
                   "outer_seed_ensemble": list(SEEDS)}, f, indent=2)
    print(pd.DataFrame(per_repeat).to_string(index=False))


if __name__ == "__main__":
    main()
