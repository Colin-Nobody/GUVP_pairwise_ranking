#!/usr/bin/env python3
"""Nested unseen-city fine-tuning of the final two DINOv3 ViT-B/16 blocks."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn

from extract_features import DEFAULT_CHECKPOINT, make_model, require_cuda

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "output/vit_vp_1000_v1"
SEEDS = (42, 137, 2026)
BACKBONE_LRS = (1e-6, 3e-6, 1e-5)
EDGES = (384, 512)
ACCUMULATION = 16
MICRO_BATCH = 4
MAX_EPOCHS = 60
PATIENCE = 20
WARMUP_EPOCHS = 5
DROPOUT = 0.5
WEIGHT_DECAY = 0.01


class RegressionHead(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(128, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)


def city_ids(frame: pd.DataFrame, repeat: int, outer_fold: int, inner_fold: int | None = None):
    mask = (frame.repeat == repeat) & (frame.outer_fold == outer_fold)
    if inner_fold is not None:
        mask &= frame.inner_fold == inner_fold
    part = frame.loc[mask]
    valid_role = "test" if inner_fold is None else "validation"
    return (set(part.loc[part.partition == "train", "city"].astype(str)),
            set(part.loc[part.partition == valid_role, "city"].astype(str)))


def score_metrics(y, pred):
    return {"mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred)),
            "spearman": float(spearmanr(y, pred).statistic),
            "pearson": float(pearsonr(y, pred).statistic)}


def atomic_csv(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(tmp_name, index=False)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def load_prefix_cache(root: Path, edges: tuple[int, ...], device: torch.device):
    cache = {}
    for edge in edges:
        edge_dir = root / f"edge{edge}"
        ids = pd.read_csv(edge_dir / "index.csv").image_id.astype(str).tolist()
        for image_id in ids:
            with np.load(edge_dir / f"{image_id}.npz") as item:
                if device.type == "cuda":
                    arrays = tuple(torch.from_numpy(item[name].copy()).to(device=device, dtype=torch.float16)
                                   for name in ("hidden", "rope_cos", "rope_sin"))
                else:
                    arrays = tuple(item[name].copy() for name in ("hidden", "rope_cos", "rope_sin"))
                cache[(edge, image_id)] = (*arrays, (int(item["image_width"]), int(item["image_height"])))
    return cache


def batched_embeddings(model, items, device):
    if not items or len({item[3] for item in items}) != 1:
        raise ValueError("Batched prefix activations must have identical image dimensions")
    if torch.is_tensor(items[0][0]):
        x = torch.stack([item[0] for item in items]).to(dtype=torch.float32)
        c = items[0][1].to(dtype=torch.float32)
        s = items[0][2].to(dtype=torch.float32)
    else:
        x = torch.stack([torch.from_numpy(item[0]) for item in items]).to(device=device, dtype=torch.float32)
        c = torch.from_numpy(items[0][1]).to(device=device, dtype=torch.float32)
        s = torch.from_numpy(items[0][2]).to(device=device, dtype=torch.float32)
    rope = (c, s)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for block in model.model.layer[10:12]:
            x = block(x, position_embeddings=rope)
        x = model.norm(x)
        pooled = x[:, 5:, :].mean(dim=1)
    return pooled.float()


def image_batches(image_ids, cache, edge, rng):
    buckets = {}
    for image_id in image_ids:
        buckets.setdefault(cache[(edge, image_id)][3], []).append(image_id)
    batches = []
    for bucket_ids in buckets.values():
        rng.shuffle(bucket_ids)
        batches.extend(bucket_ids[i:i + MICRO_BATCH] for i in range(0, len(bucket_ids), MICRO_BATCH))
    rng.shuffle(batches)
    return batches


def set_finetune_parameters(model):
    model.requires_grad_(False)
    trainable = []
    for name, parameter in model.named_parameters():
        if name.startswith("model.layer.10.") or name.startswith("model.layer.11."):
            parameter.requires_grad_(True)
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("Could not identify final two DINOv3 Transformer blocks")
    return trainable


def original_last_blocks(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            if key.startswith("model.layer.10.") or key.startswith("model.layer.11.")}


def reset_last_blocks(model, state):
    model.load_state_dict(state, strict=False)


def schedule(optimizer, steps_per_epoch: int, epochs: int):
    total = max(1, steps_per_epoch * epochs)
    warmup = min(total, WARMUP_EPOCHS * steps_per_epoch)
    def factor(step):
        if step < warmup:
            return float(step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def train_epoch(model, head, optimizer, scheduler, cache, image_ids, target_by_id, edge, y_mean, y_std, device, rng):
    model.train(); head.train()
    batches = image_batches(list(image_ids), cache, edge, rng)
    optimizer.zero_grad(set_to_none=True)
    accumulated = 0
    for image_ids_batch in batches:
        embedding = batched_embeddings(model, [cache[(edge, image_id)] for image_id in image_ids_batch], device)
        target = torch.tensor([(target_by_id[i] - y_mean) / y_std for i in image_ids_batch], dtype=torch.float32, device=device)
        loss_sum = torch.nn.functional.mse_loss(head(embedding), target, reduction="sum")
        loss_sum.backward()
        accumulated += len(image_ids_batch)
        if accumulated >= ACCUMULATION:
            for parameter in list(model.parameters()) + list(head.parameters()):
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); accumulated = 0
    if accumulated:
        for parameter in list(model.parameters()) + list(head.parameters()):
            if parameter.grad is not None:
                parameter.grad.div_(accumulated)
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)


@torch.no_grad()
def evaluate(model, head, cache, image_ids, target_by_id, edge, y_mean, y_std, device):
    model.eval(); head.eval()
    pred = {}
    losses = []
    rng = np.random.default_rng(42)
    for batch in image_batches(list(image_ids), cache, edge, rng):
        embedding = batched_embeddings(model, [cache[(edge, image_id)] for image_id in batch], device)
        values = head(embedding).cpu().numpy().astype(np.float64) * y_std + y_mean
        for image_id, value in zip(batch, values):
            pred[image_id] = float(value)
            if target_by_id is not None:
                losses.append((float(value) - target_by_id[image_id]) ** 2 / (y_std ** 2))
    mse = float(np.mean(losses)) if losses else float("nan")
    return pred, mse


def fit_tuned(model, base_blocks, cache, train_ids, y_map, valid_ids, y_valid, edge, backbone_lr, seed, device, max_epochs, patience):
    reset_last_blocks(model, base_blocks)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if device.type == "cuda" else None
    head = RegressionHead().to(device)
    model.train()
    backbone_params = [p for p in model.parameters() if p.requires_grad]
    head_params = list(head.parameters())
    optimizer = torch.optim.AdamW([{"params": backbone_params, "lr": backbone_lr},
                                  {"params": head_params, "lr": backbone_lr * 10}], weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(train_ids) / ACCUMULATION)
    lr_scheduler = schedule(optimizer, steps_per_epoch, max_epochs)
    y_mean = float(np.mean([y_map[i] for i in train_ids]))
    y_std = float(np.std([y_map[i] for i in train_ids])) or 1.0
    target_by_id = y_map
    best_mse, best_epoch, wait, best_state = float("inf"), 0, 0, None
    rng = np.random.default_rng(seed)
    for epoch in range(1, max_epochs + 1):
        train_epoch(model, head, optimizer, lr_scheduler, cache, train_ids, target_by_id, edge, y_mean, y_std, device, rng)
        _, val_mse = evaluate(model, head, cache, valid_ids, y_valid, edge, y_mean, y_std, device)
        if val_mse < best_mse - 1e-7:
            best_mse, best_epoch, wait = val_mse, epoch, 0
            best_state = ({k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                           if k.startswith("model.layer.10.") or k.startswith("model.layer.11.")},
                          {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is None:
        raise RuntimeError("Fine-tuning produced no valid inner checkpoint")
    reset_last_blocks(model, best_state[0]); head.load_state_dict(best_state[1])
    val_pred, _ = evaluate(model, head, cache, valid_ids, None, edge, y_mean, y_std, device)
    val_mae = mean_absolute_error([y_valid[i] for i in valid_ids], [val_pred[i] for i in valid_ids])
    return float(val_mae), int(best_epoch)


def fit_fixed(model, base_blocks, cache, train_ids, y_map, predict_ids, edge, backbone_lr, seed, epochs, device):
    reset_last_blocks(model, base_blocks)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if device.type == "cuda" else None
    head = RegressionHead().to(device)
    backbone_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": backbone_params, "lr": backbone_lr},
                                  {"params": list(head.parameters()), "lr": backbone_lr * 10}], weight_decay=WEIGHT_DECAY)
    lr_scheduler = schedule(optimizer, math.ceil(len(train_ids) / ACCUMULATION), epochs)
    y_mean = float(np.mean([y_map[i] for i in train_ids]))
    y_std = float(np.std([y_map[i] for i in train_ids])) or 1.0
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        train_epoch(model, head, optimizer, lr_scheduler, cache, train_ids, y_map, edge, y_mean, y_std, device, rng)
    prediction, _ = evaluate(model, head, cache, predict_ids, None, edge, y_mean, y_std, device)
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT)
    parser.add_argument("--prefix-dir", type=Path, default=AUDIT / "features" / "prefix_after_block10")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=AUDIT / "finetune_last2")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--profile", action="store_true", help="Run only one inner fold and one LR/resolution for timing and memory profiling")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true", help="Compatibility flag; CUDA is always required")
    parser.add_argument("--skip-ridge-comparison", action="store_true", help="Do not load a separate frozen-Ridge OOF file")
    args = parser.parse_args()
    if not 1 <= args.repeats <= len(SEEDS):
        raise ValueError(f"repeats must be between 1 and {len(SEEDS)}")
    device = torch.device(args.device)
    device = require_cuda(device)
    manifest = pd.read_csv(args.audit_dir / "image_manifest.csv")
    ids = manifest.image_id.astype(str).tolist()
    cities = manifest.city.astype(str).to_numpy()
    y = manifest.target_visual_pollution_score_0_10.to_numpy(dtype=np.float64)
    y_map = dict(zip(ids, y))
    index = {image_id: i for i, image_id in enumerate(ids)}
    outer = pd.read_csv(args.audit_dir / "splits/unseen_city_outer.csv")
    inner = pd.read_csv(args.audit_dir / "splits/unseen_city_inner.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.profile:
        config_path = args.output_dir / "resume_config.json"
        resume_config = {
            "repeats": args.repeats,
            "edges": list(EDGES),
            "backbone_lrs": list(BACKBONE_LRS),
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY,
            "micro_batch_size": MICRO_BATCH,
            "effective_batch_size": ACCUMULATION,
            "seeds": list(SEEDS),
        }
        if config_path.exists():
            saved_config = json.loads(config_path.read_text())
            if saved_config != resume_config:
                raise ValueError(f"Resume configuration differs from {config_path}; use the original settings or a new output directory")
        else:
            stale = [args.output_dir / name for name in ("inner_candidate_scores.csv", "outer_predictions_partial.csv")]
            if any(path.exists() for path in stale):
                raise ValueError("Partial results exist without resume_config.json; refusing an unsafe resume")
            config_path.write_text(json.dumps(resume_config, indent=2) + "\n")

    edges = (384,) if args.profile else EDGES
    cache = load_prefix_cache(args.prefix_dir, edges, device)
    model = make_model(args.checkpoint, device)
    trainable_backbone = set_finetune_parameters(model)
    base_blocks = original_last_blocks(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if args.profile:
        train_cities, valid_cities = city_ids(inner, 1, 0, 0)
        train_ids = [i for i, c in zip(ids, cities) if c in train_cities]
        valid_ids = [i for i, c in zip(ids, cities) if c in valid_cities]
        started = time.perf_counter()
        mae, best_epoch = fit_tuned(model, base_blocks, cache, train_ids, y_map, valid_ids, y_map,
                                    384, 3e-6, 42, device, args.max_epochs, args.patience)
        print(json.dumps({"profile": True, "train_images": len(train_ids), "validation_images": len(valid_ids),
                          "edge": 384, "backbone_lr": 3e-6, "mae": mae, "best_epoch": best_epoch,
                          "elapsed_seconds": round(time.perf_counter() - started, 2),
                          "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0}, indent=2))
        return

    feature_files = []
    for edge in edges:
        feature_files.append({"edge": edge, "cache_items": sum(1 for k in cache if k[0] == edge)})
    candidates = [(edge, lr) for edge in EDGES for lr in BACKBONE_LRS]
    search_path = args.output_dir / "inner_candidate_scores.csv"
    partial_oof_path = args.output_dir / "outer_predictions_partial.csv"
    search_rows = pd.read_csv(search_path).to_dict("records") if search_path.exists() else []
    oof_rows = pd.read_csv(partial_oof_path).to_dict("records") if partial_oof_path.exists() else []
    fold_rows = []
    completed_outer = {(int(r["repeat"]), int(r["outer_fold"])) for r in oof_rows}
    for repeat in range(1, args.repeats + 1):
        for outer_fold in range(5):
            if (repeat, outer_fold) in completed_outer:
                print(f"resume: repeat {repeat} outer {outer_fold + 1}/5 already complete", flush=True)
                continue
            train_cities, test_cities = city_ids(outer, repeat, outer_fold)
            train_ids = [i for i, c in zip(ids, cities) if c in train_cities]
            test_ids = [i for i, c in zip(ids, cities) if c in test_cities]
            candidate_mae, candidate_epochs = {}, {}
            for edge, lr in candidates:
                fold_maes, fold_epochs = [], []
                for inner_fold in range(4):
                    cached = next((r for r in search_rows if int(r["repeat"]) == repeat
                                   and int(r["outer_fold"]) == outer_fold and int(r["inner_fold"]) == inner_fold
                                   and int(r["edge"]) == edge and np.isclose(float(r["backbone_lr"]), lr)), None)
                    if cached is not None:
                        fold_maes.append(float(cached["validation_mae"]))
                        fold_epochs.append(int(cached["best_epoch"]))
                        continue
                    fit_cities, val_cities = city_ids(inner, repeat, outer_fold, inner_fold)
                    inner_train_ids = [i for i, c in zip(ids, cities) if c in fit_cities]
                    val_ids = [i for i, c in zip(ids, cities) if c in val_cities]
                    candidate_started = time.perf_counter()
                    mae, best_epoch = fit_tuned(model, base_blocks, cache, inner_train_ids, y_map, val_ids, y_map,
                                                edge, lr, 42, device, args.max_epochs, args.patience)
                    fold_maes.append(mae); fold_epochs.append(best_epoch)
                    search_rows.append({"repeat": repeat, "seed": SEEDS[repeat - 1], "outer_fold": outer_fold,
                                        "inner_fold": inner_fold, "edge": edge, "backbone_lr": lr,
                                        "validation_mae": mae, "best_epoch": best_epoch,
                                        "elapsed_seconds": round(time.perf_counter() - candidate_started, 3)})
                    atomic_csv(pd.DataFrame(search_rows), search_path)
                    print(f"repeat {repeat}/{args.repeats} outer {outer_fold + 1}/5 candidate edge={edge} lr={lr:g} inner {inner_fold + 1}/4 done ({best_epoch} epochs, {search_rows[-1]['elapsed_seconds']:.1f}s)", flush=True)
                candidate_mae[(edge, lr)] = float(np.mean(fold_maes))
                candidate_epochs[(edge, lr)] = fold_epochs
            selected = min(candidates, key=lambda item: (candidate_mae[item], candidates.index(item)))
            edge, lr = selected
            epochs = max(1, int(np.median(candidate_epochs[selected])))
            selected_config = {"repeat": repeat, "seed": SEEDS[repeat - 1], "outer_fold": outer_fold,
                                "selected_edge": edge, "backbone_lr": lr, "head_lr": lr * 10,
                                "inner_mae": candidate_mae[selected], "inner_best_epochs": ";".join(map(str, candidate_epochs[selected])),
                                "refit_epochs": epochs, "candidate_count": len(candidates), "inner_folds": 4}
            selected_path = args.output_dir / "inner_selected_configs_partial.csv"
            selected_rows = pd.read_csv(selected_path).to_dict("records") if selected_path.exists() else []
            if not any(int(r["repeat"]) == repeat and int(r["outer_fold"]) == outer_fold for r in selected_rows):
                selected_rows.append(selected_config)
                atomic_csv(pd.DataFrame(selected_rows), selected_path)
            seed_preds = []
            for seed in SEEDS:
                pred = fit_fixed(model, base_blocks, cache, train_ids, y_map, test_ids, edge, lr, seed, epochs, device)
                seed_preds.append(pred)
            combined = {image_id: float(np.mean([p[image_id] for p in seed_preds])) for image_id in test_ids}
            fold_metric = score_metrics(np.array([y_map[i] for i in test_ids]), np.array([combined[i] for i in test_ids]))
            fold_rows.append({"repeat": repeat, "outer_fold": outer_fold, "model": "dinov3_last2_finetune_mlp",
                              "edge": edge, "backbone_lr": lr, "refit_epochs": epochs,
                              "n_train": len(train_ids), "n_test": len(test_ids), **fold_metric})
            for image_id in test_ids:
                i = index[image_id]
                oof_rows.append({"repeat": repeat, "outer_fold": outer_fold, "image_id": image_id, "city": cities[i],
                                 "y_true": y_map[image_id], "y_pred": combined[image_id],
                                 "model": "dinov3_last2_finetune_mlp", "edge": edge, "backbone_lr": lr,
                                 "outer_train_seed_ensemble": ",".join(map(str, SEEDS))})
            atomic_csv(pd.DataFrame(oof_rows), partial_oof_path)
            print(f"repeat {repeat}/{args.repeats} outer fold {outer_fold + 1}/5 selected edge={edge} lr={lr:g} epochs={epochs}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof = pd.DataFrame(oof_rows)
    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    pd.read_csv(args.output_dir / "inner_selected_configs_partial.csv").to_csv(args.output_dir / "inner_selected_configs.csv", index=False)
    oof_frame = pd.DataFrame(oof_rows)
    fold_rows = [{"repeat": int(repeat), "outer_fold": int(outer_fold), "model": "dinov3_last2_finetune_mlp",
                  **score_metrics(group.y_true.to_numpy(), group.y_pred.to_numpy())}
                 for (repeat, outer_fold), group in oof_frame.groupby(["repeat", "outer_fold"])]
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "outer_fold_metrics.csv", index=False)
    repeats = []
    for repeat, group in oof.groupby("repeat"):
        repeats.append({"repeat": int(repeat), **score_metrics(group.y_true.to_numpy(), group.y_pred.to_numpy())})
    repeat_df = pd.DataFrame(repeats)
    repeat_df.to_csv(args.output_dir / "per_repeat_metrics.csv", index=False)
    city_metrics = []
    for (repeat, city), group in oof.groupby(["repeat", "city"]):
        city_metrics.append({"repeat": int(repeat), "city": str(city),
                             **score_metrics(group.y_true.to_numpy(), group.y_pred.to_numpy())})
    pd.DataFrame(city_metrics).to_csv(args.output_dir / "metrics_by_city.csv", index=False)
    deltas = []
    if not args.skip_ridge_comparison:
        compare = pd.read_csv(args.audit_dir / "frozen_ridge/oof_predictions.csv")
        compare = compare.loc[compare.model == "dinov3_ridge", ["repeat", "image_id", "y_pred"]].rename(columns={"y_pred": "ridge_pred"})
        merged = oof.merge(compare, on=["repeat", "image_id"], validate="one_to_one")
        for repeat, group in merged.groupby("repeat"):
            deltas.append({"repeat": int(repeat), "finetune_mae": mean_absolute_error(group.y_true, group.y_pred),
                           "ridge_mae": mean_absolute_error(group.y_true, group.ridge_pred),
                           "mae_delta_finetune_minus_ridge": mean_absolute_error(group.y_true, group.y_pred)-mean_absolute_error(group.y_true, group.ridge_pred),
                           "finetune_rmse": np.sqrt(mean_squared_error(group.y_true, group.y_pred)),
                           "ridge_rmse": np.sqrt(mean_squared_error(group.y_true, group.ridge_pred)),
                           "spearman_delta_finetune_minus_ridge": spearmanr(group.y_true, group.y_pred).statistic-spearmanr(group.y_true, group.ridge_pred).statistic})
        pd.DataFrame(deltas).to_csv(args.output_dir / "ridge_comparison_by_repeat.csv", index=False)
    metadata = {"model": "DINOv3 ViT-B/16 with last 2 blocks fine-tuned + patch-mean MLP",
                "repeats": args.repeats, "seeds": list(SEEDS), "outer_folds": 5, "inner_folds": 4,
                "candidate_edges": list(EDGES), "backbone_learning_rates": list(BACKBONE_LRS),
                "head_learning_rate_multiplier": 10, "weight_decay": 0.01, "max_epochs": args.max_epochs,
                "patience": args.patience, "warmup_epochs": WARMUP_EPOCHS, "effective_batch_size": ACCUMULATION,
                "micro_batch_size_by_identical_image_geometry": MICRO_BATCH,
                "feature_precision": "bfloat16 autocast; cached frozen prefix stored in float16",
                "outer_refit_epoch_rule": "median inner best epoch for selected config; outer test labels are not used",
                "device": str(device), "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
                "label_scope": "Fixed global Qwen/TrueSkill labels; city-grouped nested evaluation",
                "source_checkpoint": str(args.checkpoint.resolve()),
                "source_checkpoint_sha256": __import__("hashlib").sha256(args.checkpoint.read_bytes()).hexdigest(),
                "label_provenance": json.loads((args.audit_dir / "label_provenance.json").read_text())
                    if (args.audit_dir / "label_provenance.json").is_file() else None,
                "dataset_audit": json.loads((args.audit_dir / "dataset_audit.json").read_text())
                    if (args.audit_dir / "dataset_audit.json").is_file() else None}
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(repeat_df.to_string(index=False))
    if deltas:
        print(pd.DataFrame(deltas).to_string(index=False))


if __name__ == "__main__":
    main()
