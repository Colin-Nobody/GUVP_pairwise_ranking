#!/usr/bin/env python3
"""Run 3x5 city-held-out OOF on the 900 development images, then fit all 900."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

from extract_features import DEFAULT_CHECKPOINT, checkpoint_sha256, make_model, require_cuda
from train_finetune import (
    ACCUMULATION, SEEDS, fit_fixed, load_prefix_cache, original_last_blocks,
    score_metrics, set_finetune_parameters,
)
from train_final_finetune import fit_final_seed


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_safetensors(tensors: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def make_city_splits(cities: list[str]) -> pd.DataFrame:
    if len(cities) != 18:
        raise ValueError(f"Expected 18 training cities, got {len(cities)}")
    rows = []
    for repeat, seed in enumerate(SEEDS, start=1):
        rng = np.random.default_rng(seed)
        shuffled = np.asarray(cities, dtype=object)[rng.permutation(len(cities))]
        folds = np.array_split(shuffled, 5)
        for fold, test_cities in enumerate(folds):
            test_set = set(map(str, test_cities))
            for city in cities:
                rows.append({"repeat": repeat, "outer_fold": fold, "city": city,
                             "partition": "test" if city in test_set else "train"})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parents[1] / "output/vit_vp_v9_holdout_900_300_cv")
    ap.add_argument("--prefix-dir", type=Path, default=Path(__file__).resolve().parents[1] / "output/vit_vp_1000_v1/features/prefix_after_block10")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=16)
    args = ap.parse_args()

    root = args.experiment_dir
    train = pd.read_csv(root / "training/image_manifest.csv", dtype={"image_id": str})
    if len(train) != 900 or train.image_id.duplicated().any() or train.city.nunique() != 18:
        raise ValueError("Expected exactly 900 unique training images from 18 cities")
    if set(train.cohort) != {"original_1000"}:
        raise ValueError("Only the 900-image original cohort may be used for fitting")
    if not train.image_path.map(lambda p: Path(p).is_file()).all():
        raise FileNotFoundError("A training image is missing")
    if not train.target_visual_pollution_score_0_10.astype(float).between(0, 10).all():
        raise ValueError("Training targets must be in [0, 10]")

    ids = train.image_id.astype(str).tolist()
    y_map = dict(zip(ids, train.target_visual_pollution_score_0_10.astype(float)))
    cities = dict(zip(ids, train.city.astype(str)))
    city_set = sorted(set(cities.values()))
    split_path = root / "splits/training_city_outer.csv"
    outer = make_city_splits(city_set)
    if split_path.exists():
        prior = pd.read_csv(split_path)
        if not prior.equals(outer):
            raise ValueError("Persisted city splits do not match the deterministic split design")
    else:
        split_path.parent.mkdir(parents=True, exist_ok=True)
        outer.to_csv(split_path, index=False)

    device = require_cuda(torch.device(args.device))
    cache = load_prefix_cache(args.prefix_dir, (512,), device)
    cache_ids = {image_id for edge, image_id in cache if edge == 512}
    if not set(ids) <= cache_ids:
        raise ValueError(f"Training prefix cache is missing {len(set(ids)-cache_ids)} IDs")

    config = {
        "model": "DINOv3 ViT-B/16 LVD-1689M; final two blocks fine-tuned; patch-mean MLP head",
        "source_checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "initialization": "original pretrained checkpoint for every fold and seed; no prior V6/V9 fine-tuned weights",
        "label_version": "V9 optimized prompt / joint 1,200-image Qwen-TrueSkill labels",
        "training_images": 900,
        "outer_evaluation": "3 repeats x 5 city-held-out folds within the 900 development images; 3-seed ensemble per fold",
        "final_fit": "three seeds fitted on all 900 development images after OOF evaluation",
        "external_test_images": 300,
        "external_test_manifest_or_labels_loaded_during_fit": False,
        "fixed_edge": 512,
        "backbone_learning_rate": 1e-5,
        "head_learning_rate": 1e-4,
        "weight_decay": 0.01,
        "dropout": 0.5,
        "epochs": args.epochs,
        "effective_batch_size": ACCUMULATION,
        "seeds": list(SEEDS),
        "selection_and_early_stopping": "none; fixed configuration and fixed epochs; the 300-image test is untouched",
        "city_fold_seed_by_repeat": dict(zip((1, 2, 3), SEEDS)),
    }
    config_path = root / "training_config_cv.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Existing CV configuration differs")
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    model = make_model(args.checkpoint, device)
    set_finetune_parameters(model)
    base_blocks = original_last_blocks(model)
    partial = root / "internal_oof_predictions_partial.csv"
    oof_rows = pd.read_csv(partial).to_dict("records") if partial.exists() else []
    complete = {(int(r["repeat"]), int(r["outer_fold"])) for r in oof_rows}

    for repeat in (1, 2, 3):
        for fold in range(5):
            key = (repeat, fold)
            if key in complete:
                print(f"resume: repeat={repeat} fold={fold+1}/5 complete", flush=True)
                continue
            local = outer[(outer.repeat == repeat) & (outer.outer_fold == fold)]
            test_cities = set(local.loc[local.partition == "test", "city"].astype(str))
            train_ids = [i for i in ids if cities[i] not in test_cities]
            test_ids = [i for i in ids if cities[i] in test_cities]
            if len(train_ids) + len(test_ids) != 900 or set(train_ids) & set(test_ids):
                raise ValueError("Bad city fold partition")
            if {cities[i] for i in train_ids} & {cities[i] for i in test_ids}:
                raise ValueError("City leakage in internal OOF split")
            seed_predictions = [fit_fixed(model, base_blocks, cache, train_ids, y_map, test_ids,
                                          512, 1e-5, seed, args.epochs, device) for seed in SEEDS]
            combined = {i: float(np.mean([p[i] for p in seed_predictions])) for i in test_ids}
            fold_metrics = score_metrics(np.asarray([y_map[i] for i in test_ids]),
                                         np.asarray([combined[i] for i in test_ids]))
            for image_id in test_ids:
                oof_rows.append({"repeat": repeat, "seed_ensemble": ",".join(map(str, SEEDS)),
                                 "outer_fold": fold, "image_id": image_id, "city": cities[image_id],
                                 "y_true": y_map[image_id], "y_pred": combined[image_id],
                                 "n_train": len(train_ids), "n_test": len(test_ids),
                                 "model": "dinov3_vitb16_last2_mlp_v9"})
            atomic_csv(pd.DataFrame(oof_rows), partial)
            print(f"internal OOF repeat={repeat}/3 fold={fold+1}/5 "
                  f"n_train={len(train_ids)} n_test={len(test_ids)} "
                  f"MAE={fold_metrics['mae']:.4f} RMSE={fold_metrics['rmse']:.4f} "
                  f"rho={fold_metrics['spearman']:.4f}", flush=True)

    oof = pd.DataFrame(oof_rows)
    if len(oof) != 2700 or oof.duplicated(["repeat", "image_id"]).any():
        raise ValueError("Expected 2,700 OOF predictions, one per development image and repeat")
    atomic_csv(oof, root / "internal_oof_predictions.csv")
    per_repeat = [{"repeat": int(r), **score_metrics(g.y_true.to_numpy(), g.y_pred.to_numpy())}
                  for r, g in oof.groupby("repeat")]
    atomic_csv(pd.DataFrame(per_repeat), root / "internal_oof_metrics_by_repeat.csv")
    per_fold = []
    for (repeat, fold), group in oof.groupby(["repeat", "outer_fold"]):
        per_fold.append({"repeat": int(repeat), "outer_fold": int(fold), "n_test": len(group),
                         **score_metrics(group.y_true.to_numpy(), group.y_pred.to_numpy())})
    atomic_csv(pd.DataFrame(per_fold), root / "internal_oof_metrics_by_fold.csv")

    deployment = root / "deployment/finetuned_last2_v9"
    deployment.mkdir(parents=True, exist_ok=True)
    artifact_file = deployment / "seed_artifacts_partial.json"
    artifacts = json.loads(artifact_file.read_text(encoding="utf-8")) if artifact_file.exists() else []
    by_seed = {int(r["seed"]): r for r in artifacts}
    for seed in SEEDS:
        if seed in by_seed:
            weight_path = deployment / by_seed[seed]["weights"]
            if not weight_path.is_file() or checkpoint_sha256(weight_path) != by_seed[seed]["sha256"]:
                raise ValueError(f"Partial final model missing/hash mismatch: {weight_path}")
            print(f"resume: final seed={seed} complete", flush=True)
            continue
        payload, y_mean, y_std = fit_final_seed(model, base_blocks, cache, ids, y_map, seed, args.epochs, device)
        weight_path = deployment / f"seed_{seed}.safetensors"
        atomic_safetensors(payload, weight_path)
        row = {"seed": seed, "weights": weight_path.name, "sha256": checkpoint_sha256(weight_path),
               "bytes": weight_path.stat().st_size, "target_mean": y_mean, "target_std": y_std}
        artifacts.append(row)
        by_seed[seed] = row
        artifact_file.write_text(json.dumps(artifacts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        del payload
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"final full-900 fit seed={seed} complete", flush=True)

    artifacts = [by_seed[s] for s in SEEDS]
    metadata = {
        **config,
        "label": "V9 optimized prompt Qwen/TrueSkill visual_pollution_score_0_10",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "feature_cache_edge": 512,
        "pooling": "mean of final normalized image patch tokens excluding CLS and four register tokens",
        "preprocessing": "EXIF transpose, RGB, aspect-preserving resize to 512 long edge; dimensions rounded to nearest multiple of 16; Pillow bicubic; ImageNet mean/std",
        "prediction_rule": "arithmetic mean of seed 42, 137, 2026 predictions",
        "device": str(device), "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "artifacts": artifacts,
        "label_graph_caveat": "V9 labels were jointly estimated on the 1,200-image graph, although all 300 final-test images and six cities were excluded from ViT fitting and selection.",
    }
    (deployment / "model.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"3x5 city OOF and final 900-image fit complete: {deployment}", flush=True)


if __name__ == "__main__":
    main()
