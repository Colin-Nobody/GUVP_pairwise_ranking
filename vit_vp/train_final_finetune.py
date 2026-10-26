#!/usr/bin/env python3
"""Fit the selected last-two-block DINOv3 regressor on all development images."""
from __future__ import annotations

import argparse
import json
import statistics
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

from extract_features import DEFAULT_CHECKPOINT, checkpoint_sha256, make_model, require_cuda
from train_finetune import (
    ACCUMULATION,
    AUDIT,
    SEEDS,
    RegressionHead,
    load_prefix_cache,
    original_last_blocks,
    reset_last_blocks,
    schedule,
    set_finetune_parameters,
    train_epoch,
)


def atomic_save_safetensors(tensors: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
    try:
        save_file({name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}, str(tmp))
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def fit_final_seed(model, base_blocks, cache, ids, y_map, seed: int, epochs: int, device):
    reset_last_blocks(model, base_blocks)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    head = RegressionHead().to(device)
    backbone_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 1e-5},
            {"params": list(head.parameters()), "lr": 1e-4},
        ],
        weight_decay=0.01,
    )
    steps_per_epoch = int(np.ceil(len(ids) / ACCUMULATION))
    lr_scheduler = schedule(optimizer, steps_per_epoch, epochs)
    y_mean = float(np.mean([y_map[i] for i in ids]))
    y_std = float(np.std([y_map[i] for i in ids])) or 1.0
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        train_epoch(model, head, optimizer, lr_scheduler, cache, ids, y_map, 512, y_mean, y_std, device, rng)
        print(f"seed={seed} epoch={epoch}/{epochs}", flush=True)
    payload = {
        **{f"encoder.{name}": value.detach().cpu() for name, value in model.state_dict().items()
           if name.startswith("model.layer.10.") or name.startswith("model.layer.11.")},
        **{f"head.{name}": value.detach().cpu() for name, value in head.state_dict().items()},
    }
    return payload, y_mean, y_std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT)
    parser.add_argument("--prefix-dir", type=Path, default=AUDIT / "features" / "prefix_after_block10")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=AUDIT / "deployment" / "finetuned_last2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="Replace an existing fine-tuned deployment package")
    args = parser.parse_args()

    selected_path = args.audit_dir / "finetune_last2" / "inner_selected_configs.csv"
    selected = pd.read_csv(selected_path)
    required = {"selected_edge", "backbone_lr", "refit_epochs"}
    if not required.issubset(selected.columns) or len(selected) != 15:
        raise ValueError(f"Expected 15 complete outer-fold selections in {selected_path}")
    edge_counts = selected.selected_edge.astype(int).value_counts().to_dict()
    if edge_counts.get(512, 0) <= edge_counts.get(384, 0):
        raise ValueError(f"512 must be the unique modal edge based on inner-fold selections: {edge_counts}")
    if not np.allclose(selected.backbone_lr.astype(float), 1e-5):
        raise ValueError("Backbone LR is not consistently 1e-5 across outer-fold inner selections")
    epochs = int(statistics.median(selected.refit_epochs.astype(int).tolist()))
    if epochs != 16:
        raise ValueError(f"Unexpected median refit epoch count: {epochs}; review the selection protocol")

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output is not empty; choose a new directory or pass --force: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.audit_dir / "image_manifest.csv")
    ids = manifest.image_id.astype(str).tolist()
    if len(ids) != 1000 or len(set(ids)) != 1000:
        raise ValueError(f"Expected 1,000 unique development image IDs; got {len(ids)} rows")
    y_values = manifest.target_visual_pollution_score_0_10.astype(float).to_numpy()
    y_map = dict(zip(ids, y_values))
    device = require_cuda(torch.device(args.device))
    cache = load_prefix_cache(args.prefix_dir, (512,), device)
    cache_ids = {image_id for edge, image_id in cache if edge == 512}
    if cache_ids != set(ids):
        raise ValueError(f"512 prefix cache ID mismatch: missing={len(set(ids)-cache_ids)}, extra={len(cache_ids-set(ids))}")

    model = make_model(args.checkpoint, device)
    set_finetune_parameters(model)
    base_blocks = original_last_blocks(model)
    seed_artifacts = []
    for seed in SEEDS:
        payload, y_mean, y_std = fit_final_seed(model, base_blocks, cache, ids, y_map, seed, epochs, device)
        path = args.output_dir / f"seed_{seed}.safetensors"
        atomic_save_safetensors(payload, path)
        seed_artifacts.append({"seed": seed, "weights": path.name, "sha256": checkpoint_sha256(path),
                               "bytes": path.stat().st_size, "target_mean": y_mean, "target_std": y_std})
        del payload
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata = {
        "model": "DINOv3 ViT-B/16 LVD-1689M; final two transformer blocks fine-tuned; patch-mean MLP head",
        "label": "visual_pollution_score_0_10; fixed Qwen/TrueSkill-derived v6 labels",
        "training_images": len(ids),
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "feature_cache_edge": 512,
        "preprocessing": "EXIF transpose, RGB, aspect-preserving resize to 512 long edge; dimensions rounded to nearest multiple of 16; Pillow bicubic; ImageNet mean/std",
        "pooling": "mean of final normalized image patch tokens excluding CLS and four register tokens",
        "optimizer": "AdamW",
        "backbone_learning_rate": 1e-5,
        "head_learning_rate": 1e-4,
        "weight_decay": 0.01,
        "dropout": 0.5,
        "micro_batch_size": 4,
        "gradient_accumulation_target": ACCUMULATION,
        "epochs": epochs,
        "epoch_selection_rule": "median of the 15 outer-fold refit_epochs; hyperparameters/epochs aggregated from inner-CV selections; no full-data early stopping",
        "seed_ensemble": [seed["seed"] for seed in seed_artifacts],
        "prediction_rule": "arithmetic mean of the three seed-model predictions",
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "artifacts": seed_artifacts,
        "generalization_metrics": "Use nested out-of-fold evaluation in output/vit_vp_1000_v1/finetune_last2; this all-data fit is for deployment only.",
    }
    (args.output_dir / "model.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
