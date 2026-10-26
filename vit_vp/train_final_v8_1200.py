#!/usr/bin/env python3
"""Fit the selected V8 last-two-block model on all 1,200 images for deployment."""
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
from torch import nn

from extract_features import DEFAULT_CHECKPOINT, checkpoint_sha256, make_model, require_cuda
from train_finetune import (
    ACCUMULATION, SEEDS, load_prefix_cache, original_last_blocks,
    reset_last_blocks, schedule, set_finetune_parameters, train_epoch,
)

ROOT = Path(__file__).resolve().parents[1]


class RegressionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(768, 128), nn.GELU(), nn.Dropout(0.5), nn.Linear(128, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)


def save_atomic(payload: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
    try:
        save_file({key: value.detach().cpu().contiguous() for key, value in payload.items()}, str(temp))
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def fit_seed(model, base_blocks, cache, ids, targets, edge, learning_rate, seed, epochs, device):
    reset_last_blocks(model, base_blocks)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    head = RegressionHead().to(device)
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad], "lr": learning_rate},
         {"params": list(head.parameters()), "lr": learning_rate * 10}], weight_decay=0.01)
    steps = int(np.ceil(len(ids) / ACCUMULATION))
    lr_scheduler = schedule(optimizer, steps, epochs)
    target_mean = float(np.mean([targets[image_id] for image_id in ids]))
    target_std = float(np.std([targets[image_id] for image_id in ids])) or 1.0
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        train_epoch(model, head, optimizer, lr_scheduler, cache, ids, targets, edge,
                    target_mean, target_std, device, rng)
        print(f"final fit seed={seed} epoch={epoch}/{epochs}", flush=True)
    payload = {
        **{f"encoder.{name}": value.detach().cpu() for name, value in model.state_dict().items()
           if name.startswith("model.layer.10.") or name.startswith("model.layer.11.")},
        **{f"head.{name}": value.detach().cpu() for name, value in head.state_dict().items()},
    }
    return payload, target_mean, target_std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=ROOT / "output/vit_vp_v8_1200_citycv")
    parser.add_argument("--training-dir", type=Path, default=ROOT / "output/vit_vp_v8_1200_citycv/finetune_last2")
    parser.add_argument("--prefix-dir", type=Path, default=ROOT / "output/vit_vp_v8_1200_citycv/prefix_after_block10")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/vit_vp_v8_1200_citycv/deployment/finetuned_last2_v8")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = require_cuda(torch.device(args.device))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V8 deployment fitting requires CUDA; CPU training is disabled")

    selected_path = args.training_dir / "inner_selected_configs.csv"
    selected = pd.read_csv(selected_path)
    if len(selected) != 15:
        raise ValueError(f"Expected 15 complete outer-fold selections in {selected_path}")
    edge_counts = selected.selected_edge.astype(int).value_counts()
    lr_counts = selected.backbone_lr.astype(float).value_counts()
    edge = int(sorted(edge_counts[edge_counts == edge_counts.max()].index)[0])
    learning_rate = float(sorted(lr_counts[lr_counts == lr_counts.max()].index)[0])
    epochs = max(1, int(statistics.median(selected.refit_epochs.astype(int).tolist())))

    manifest = pd.read_csv(args.audit_dir / "image_manifest.csv", dtype={"image_id": str})
    if len(manifest) != 1200 or manifest.image_id.nunique() != 1200:
        raise ValueError("Expected 1,200 unique V8 training images")
    ids = manifest.image_id.astype(str).tolist()
    targets = dict(zip(ids, manifest.target_visual_pollution_score_0_10.astype(float)))
    cache = load_prefix_cache(args.prefix_dir, (edge,), device)
    cached_ids = {image_id for current_edge, image_id in cache if current_edge == edge}
    if cached_ids != set(ids):
        raise ValueError("Prefix cache IDs do not match the V8 image manifest")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.json"
    config = {"label_version": "V8 Qwen/TrueSkill 1,200-image labels",
              "training_images": 1200, "training_cities": sorted(manifest.city.astype(str).unique()),
              "initialization": "original pretrained DINOv3 checkpoint for each seed; no V6/V9 fine-tuned weights",
              "source_checkpoint_sha256": checkpoint_sha256(args.checkpoint),
              "edge": edge, "backbone_learning_rate": learning_rate,
              "head_learning_rate": learning_rate * 10, "weight_decay": 0.01,
              "dropout": 0.5, "epochs": epochs, "seeds": list(SEEDS),
              "selection_rule": "modal edge and backbone LR from 15 nested outer-fold selections; median refit epoch"}
    config_path = args.output_dir / "fit_config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("Existing deployment fit configuration differs")
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if model_path.exists():
        print(f"Deployment ensemble already complete: {args.output_dir}", flush=True)
        return

    partial_path = args.output_dir / "seed_artifacts_partial.json"
    artifacts = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.exists() else []
    by_seed = {int(row["seed"]): row for row in artifacts}
    model = make_model(args.checkpoint, device)
    set_finetune_parameters(model)
    base_blocks = original_last_blocks(model)
    for seed in SEEDS:
        if seed in by_seed:
            row = by_seed[seed]
            weight_path = args.output_dir / row["weights"]
            if not weight_path.is_file() or checkpoint_sha256(weight_path) != row["sha256"]:
                raise ValueError(f"Partial deployment seed missing/hash mismatch: {weight_path}")
            print(f"resume: deployment seed={seed} complete", flush=True)
            continue
        payload, target_mean, target_std = fit_seed(model, base_blocks, cache, ids, targets,
                                                    edge, learning_rate, seed, epochs, device)
        weight_path = args.output_dir / f"seed_{seed}.safetensors"
        save_atomic(payload, weight_path)
        row = {"seed": seed, "weights": weight_path.name, "sha256": checkpoint_sha256(weight_path),
               "bytes": weight_path.stat().st_size, "target_mean": target_mean, "target_std": target_std}
        artifacts.append(row)
        by_seed[seed] = row
        partial_path.write_text(json.dumps(artifacts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        del payload
        torch.cuda.empty_cache()

    if set(by_seed) != set(SEEDS):
        raise ValueError("Expected one completed model for each of the three fixed seeds")
    artifacts = [by_seed[seed] for seed in SEEDS]
    provenance = json.loads((args.audit_dir / "label_provenance.json").read_text(encoding="utf-8"))
    metadata = {
        "model": "DINOv3 ViT-B/16 LVD-1689M; final two transformer blocks fine-tuned; patch-mean MLP head",
        "label": "V8 Qwen/TrueSkill-derived visual_pollution_score_0_10; model-derived labels, not human ground truth",
        "training_images": 1200, "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": checkpoint_sha256(args.checkpoint), "feature_cache_edge": edge,
        "preprocessing": "EXIF transpose, RGB, aspect-preserving resize to selected long edge; dimensions rounded to nearest multiple of 16; Pillow bicubic; ImageNet mean/std",
        "pooling": "mean of final normalized image patch tokens excluding CLS and four register tokens",
        "optimizer": "AdamW", "backbone_learning_rate": learning_rate,
        "head_learning_rate": learning_rate * 10, "weight_decay": 0.01, "dropout": 0.5,
        "epochs": epochs, "epoch_selection_rule": config["selection_rule"],
        "seed_ensemble": list(SEEDS), "prediction_rule": "arithmetic mean of the three seed-model predictions",
        "device": str(device), "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "label_provenance": provenance, "artifacts": artifacts,
        "generalization_metrics": "Use nested city-held-out OOF metrics in finetune_last2; this all-data fit is for deployment only.",
    }
    model_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Deployment ensemble complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

