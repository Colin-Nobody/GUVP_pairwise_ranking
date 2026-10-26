#!/usr/bin/env python3
"""Train the fixed DINOv3 last-two-block model on the V9 development 1,000."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from extract_features import DEFAULT_CHECKPOINT, checkpoint_sha256, make_model, require_cuda
from train_finetune import (
    ACCUMULATION,
    SEEDS,
    fit_fixed,
    load_prefix_cache,
    original_last_blocks,
    score_metrics,
    set_finetune_parameters,
)
from train_final_finetune import fit_final_seed


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temp_name, index=False)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def save_safetensors(tensors: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        save_file({name: value.detach().cpu().contiguous() for name, value in tensors.items()}, temp_name)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parents[1] / "output/vit_vp_v9_holdout_1000_200")
    parser.add_argument("--prefix-dir", type=Path, default=Path(__file__).resolve().parents[1] / "output/vit_vp_1000_v1/features/prefix_after_block10")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=16)
    args = parser.parse_args()

    root = args.experiment_dir
    training_manifest = pd.read_csv(root / "training/image_manifest.csv", dtype={"image_id": str})
    if len(training_manifest) != 1000 or training_manifest.image_id.duplicated().any():
        raise ValueError("Training input must contain 1,000 unique development image IDs")
    ids = training_manifest.image_id.astype(str).tolist()
    if training_manifest.cohort.nunique() != 1 or training_manifest.cohort.iloc[0] != "original_1000":
        raise ValueError("Only the original_1000 cohort can be passed to this training script")
    if not set(training_manifest.image_path.map(lambda p: Path(p).is_file())) == {True}:
        raise FileNotFoundError("One or more training image paths are missing")
    if not set(training_manifest.target_visual_pollution_score_0_10.astype(float).between(0, 10)) == {True}:
        raise ValueError("Training labels must be finite V9 VP scores in [0, 10]")

    y_map = dict(zip(ids, training_manifest.target_visual_pollution_score_0_10.astype(float)))
    cities = dict(zip(ids, training_manifest.city.astype(str)))
    outer = pd.read_csv(root / "splits/unseen_city_outer.csv", dtype={"city": str})
    if outer.empty or set(outer.city) != set(cities.values()):
        raise ValueError("Training-only city split file does not match the 1,000 development images")

    device = require_cuda(torch.device(args.device))
    cache = load_prefix_cache(args.prefix_dir, (512,), device)
    cache_ids = {image_id for edge, image_id in cache if edge == 512}
    if cache_ids != set(ids):
        raise ValueError(f"Training-only prefix cache mismatch: missing={len(set(ids)-cache_ids)}, extra={len(cache_ids-set(ids))}")

    config = {
        "model": "DINOv3 ViT-B/16 LVD-1689M; final two blocks fine-tuned; patch-mean MLP head",
        "source_checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "label_version": "V9 optimized prompt / joint 1,200-image Qwen-TrueSkill labels",
        "training_images": 1000,
        "excluded_holdout_images": 200,
        "internal_evaluation": "3 repeats x 5 city-held-out folds over the 1,000 development images; fixed config, no V9 holdout tuning",
        "fixed_edge": 512,
        "backbone_learning_rate": 1e-5,
        "head_learning_rate": 1e-4,
        "weight_decay": 0.01,
        "dropout": 0.5,
        "epochs": args.epochs,
        "effective_batch_size": ACCUMULATION,
        "seeds": list(SEEDS),
        "split_source": "output/vit_vp_1000_v1/splits/unseen_city_outer.csv; city groups restricted to original_1000",
        "holdout_features_or_targets_loaded_during_training": False,
    }
    config_path = root / "training_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Existing training configuration differs; use the original configuration or a new experiment directory")
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    model = make_model(args.checkpoint, device)
    set_finetune_parameters(model)
    base_blocks = original_last_blocks(model)
    partial_path = root / "internal_oof_predictions_partial.csv"
    oof_rows = pd.read_csv(partial_path).to_dict("records") if partial_path.exists() else []
    completed = {(int(row["repeat"]), int(row["outer_fold"])) for row in oof_rows}
    for repeat in (1, 2, 3):
        for fold in range(5):
            key = (repeat, fold)
            if key in completed:
                print(f"resume: repeat={repeat} fold={fold+1}/5 already complete", flush=True)
                continue
            local = outer.loc[(outer.repeat.astype(int) == repeat) & (outer.outer_fold.astype(int) == fold)]
            if local.empty:
                raise ValueError(f"Missing city split rows for repeat={repeat}, fold={fold}")
            train_cities = set(local.loc[local.partition == "train", "city"].astype(str))
            test_cities = set(local.loc[local.partition == "test", "city"].astype(str))
            train_ids = [image_id for image_id in ids if cities[image_id] in train_cities]
            test_ids = [image_id for image_id in ids if cities[image_id] in test_cities]
            if len(train_ids) != 800 or len(test_ids) != 200 or set(train_ids) & set(test_ids):
                raise ValueError(f"Unexpected internal fold sizes: train={len(train_ids)}, test={len(test_ids)}")
            predictions = []
            for seed in SEEDS:
                predictions.append(fit_fixed(model, base_blocks, cache, train_ids, y_map, test_ids,
                                             512, 1e-5, seed, args.epochs, device))
            combined = {image_id: float(np.mean([part[image_id] for part in predictions])) for image_id in test_ids}
            y_true = np.asarray([y_map[image_id] for image_id in test_ids], dtype=np.float64)
            y_pred = np.asarray([combined[image_id] for image_id in test_ids], dtype=np.float64)
            metrics = score_metrics(y_true, y_pred)
            for image_id in test_ids:
                oof_rows.append({"repeat": repeat, "seed_ensemble": ",".join(map(str, SEEDS)), "outer_fold": fold,
                                 "image_id": image_id, "city": cities[image_id], "y_true": y_map[image_id],
                                 "y_pred": combined[image_id], "n_train": len(train_ids),
                                 "model": "dinov3_vitb16_last2_mlp_v9"})
            atomic_csv(pd.DataFrame(oof_rows), partial_path)
            print(f"internal OOF repeat={repeat}/3 fold={fold+1}/5 n_train={len(train_ids)} n_test={len(test_ids)} "
                  f"MAE={metrics['mae']:.4f} RMSE={metrics['rmse']:.4f} rho={metrics['spearman']:.4f}", flush=True)

    oof = pd.DataFrame(oof_rows)
    if len(oof) != 3000 or oof.duplicated(["repeat", "image_id"]).any():
        raise ValueError("Expected exactly one OOF prediction per training image and repeat")
    atomic_csv(oof, root / "internal_oof_predictions.csv")
    repeat_rows = []
    for repeat, group in oof.groupby("repeat"):
        repeat_rows.append({"repeat": int(repeat), **score_metrics(group.y_true.to_numpy(), group.y_pred.to_numpy())})
    atomic_csv(pd.DataFrame(repeat_rows), root / "internal_oof_metrics_by_repeat.csv")
    fold_rows = []
    for (repeat, fold), group in oof.groupby(["repeat", "outer_fold"]):
        fold_rows.append({"repeat": int(repeat), "outer_fold": int(fold), "n_test": len(group),
                          **score_metrics(group.y_true.to_numpy(), group.y_pred.to_numpy())})
    atomic_csv(pd.DataFrame(fold_rows), root / "internal_oof_metrics_by_fold.csv")

    deployment = root / "deployment/finetuned_last2_v9"
    deployment.mkdir(parents=True, exist_ok=True)
    artifact_rows = []
    if not (deployment / "model.json").exists():
        for seed in SEEDS:
            payload, y_mean, y_std = fit_final_seed(model, base_blocks, cache, ids, y_map, seed, args.epochs, device)
            weight_path = deployment / f"seed_{seed}.safetensors"
            save_safetensors(payload, weight_path)
            artifact_rows.append({"seed": seed, "weights": weight_path.name,
                                  "sha256": checkpoint_sha256(weight_path), "bytes": weight_path.stat().st_size,
                                  "target_mean": y_mean, "target_std": y_std})
            del payload
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"final all-1000 fit seed={seed} complete", flush=True)
        metadata = {
            **config,
            "label": "V9 optimized prompt Qwen/TrueSkill visual_pollution_score_0_10",
            "source_checkpoint": str(args.checkpoint.resolve()),
            "feature_cache_edge": 512,
            "pooling": "mean of final normalized image patch tokens excluding CLS and four register tokens",
            "preprocessing": "EXIF transpose, RGB, aspect-preserving resize to 512 long edge; dimensions rounded to nearest multiple of 16; Pillow bicubic; ImageNet mean/std",
            "prediction_rule": "arithmetic mean of seed 42, 137, 2026 predictions",
            "device": str(device),
            "torch_version": torch.__version__,
            "transformers_version": __import__("transformers").__version__,
            "artifacts": artifact_rows,
            "holdout_evaluation": "performed by evaluate_v9_200_holdout.py after this model package is finalized",
            "label_graph_caveat": "V9 train and holdout targets were jointly estimated; model fit excluded all holdout records, but target construction contains cross-partition comparison edges.",
        }
        (deployment / "model.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Training and internal OOF complete. Final ensemble: {deployment}", flush=True)


if __name__ == "__main__":
    main()
