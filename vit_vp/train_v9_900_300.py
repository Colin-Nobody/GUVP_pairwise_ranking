#!/usr/bin/env python3
"""Fine-tune DINOv3 on the fixed 900-image V9 development partition."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

from extract_features import DEFAULT_CHECKPOINT, checkpoint_sha256, make_model, require_cuda
from train_finetune import ACCUMULATION, SEEDS, load_prefix_cache, original_last_blocks, set_finetune_parameters
from train_final_finetune import fit_final_seed


def atomic_save(tensors: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, tmp_name)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment-dir", type=Path, default=Path(__file__).resolve().parents[1] / "output/vit_vp_v9_holdout_900_300")
    ap.add_argument("--prefix-dir", type=Path, default=Path(__file__).resolve().parents[1] / "output/vit_vp_1000_v1/features/prefix_after_block10")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=16)
    args = ap.parse_args()

    root = args.experiment_dir
    train = pd.read_csv(root / "training/image_manifest.csv", dtype={"image_id": str})
    if len(train) != 900 or train.image_id.duplicated().any():
        raise ValueError("Training manifest must contain 900 unique images")
    if set(train.cohort) != {"original_1000"} or train.city.nunique() != 18:
        raise ValueError("Training must contain 18 original-cohort cities only")
    if not train.image_path.map(lambda p: Path(p).is_file()).all():
        raise FileNotFoundError("A training image path is missing")
    targets = train.target_visual_pollution_score_0_10.astype(float)
    if not targets.between(0, 10).all() or not np.isfinite(targets).all():
        raise ValueError("Training labels must be finite V9 VP scores in [0, 10]")

    ids = train.image_id.astype(str).tolist()
    y_map = dict(zip(ids, targets))
    device = require_cuda(torch.device(args.device))
    cache = load_prefix_cache(args.prefix_dir, (512,), device)
    cache_ids = {image_id for edge, image_id in cache if edge == 512}
    if not set(ids) <= cache_ids:
        raise ValueError(f"Training cache missing {len(set(ids)-cache_ids)} images")

    config = {
        "model": "DINOv3 ViT-B/16 LVD-1689M; final two blocks fine-tuned; patch-mean MLP head",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "initialization": "original pretrained checkpoint; no previous V6 or V9 fine-tuned weights",
        "label_version": "V9 optimized prompt / joint 1,200-image Qwen-TrueSkill labels",
        "training_images": len(ids),
        "training_cities": sorted(train.city.astype(str).unique().tolist()),
        "test_images_excluded": 300,
        "test_city_groups_excluded": 6,
        "test_manifest_or_labels_loaded": False,
        "fixed_edge": 512,
        "backbone_learning_rate": 1e-5,
        "head_learning_rate": 1e-4,
        "weight_decay": 0.01,
        "dropout": 0.5,
        "epochs": args.epochs,
        "effective_batch_size": ACCUMULATION,
        "seeds": list(SEEDS),
        "selection_and_early_stopping": "none; fixed config and fixed epoch count, test partition untouched",
    }
    config_path = root / "training_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Training configuration mismatch; use the original configuration or a new experiment directory")
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    deployment = root / "deployment/finetuned_last2_v9"
    deployment.mkdir(parents=True, exist_ok=True)
    if (deployment / "model.json").exists():
        print(f"Final ensemble already complete: {deployment}", flush=True)
        return
    partial_path = deployment / "seed_artifacts_partial.json"
    artifacts = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.exists() else []
    complete = {int(row["seed"]): row for row in artifacts}

    model = make_model(args.checkpoint, device)
    set_finetune_parameters(model)
    base_blocks = original_last_blocks(model)
    for seed in SEEDS:
        if seed in complete:
            weight_path = deployment / complete[seed]["weights"]
            if weight_path.is_file() and checkpoint_sha256(weight_path) == complete[seed]["sha256"]:
                print(f"resume: seed={seed} already complete", flush=True)
                continue
            raise ValueError(f"Partial seed artifact is missing or has a hash mismatch: {weight_path}")
        payload, y_mean, y_std = fit_final_seed(model, base_blocks, cache, ids, y_map, seed, args.epochs, device)
        weight_path = deployment / f"seed_{seed}.safetensors"
        atomic_save(payload, weight_path)
        row = {"seed": seed, "weights": weight_path.name, "sha256": checkpoint_sha256(weight_path),
               "bytes": weight_path.stat().st_size, "target_mean": y_mean, "target_std": y_std}
        artifacts.append(row)
        complete[seed] = row
        partial_path.write_text(json.dumps(artifacts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        del payload
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"saved seed={seed} from original pretrained checkpoint", flush=True)

    if set(complete) != set(SEEDS):
        raise ValueError("Expected all three seed models before finalizing metadata")
    artifacts = [complete[seed] for seed in SEEDS]
    metadata = {
        **config,
        "label": "V9 optimized prompt Qwen/TrueSkill visual_pollution_score_0_10",
        "feature_cache_edge": 512,
        "pooling": "mean of final normalized image patch tokens excluding CLS and four register tokens",
        "preprocessing": "EXIF transpose, RGB, aspect-preserving resize to 512 long edge; dimensions rounded to nearest multiple of 16; Pillow bicubic; ImageNet mean/std",
        "prediction_rule": "arithmetic mean of seed 42, 137, 2026 predictions",
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "artifacts": artifacts,
        "label_graph_caveat": "All V9 targets were jointly estimated on the 1,200-image comparison graph; model fitting excluded all 300 test records and six test cities.",
    }
    (deployment / "model.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Training complete: {deployment}", flush=True)


if __name__ == "__main__":
    main()
