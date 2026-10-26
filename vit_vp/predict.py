#!/usr/bin/env python3
"""Run the deployed frozen DINOv3 ViT-B/16 VP model on an image or directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from extract_features import DEFAULT_CHECKPOINT, checkpoint_sha256, make_model, preprocess, require_cuda

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "output/vit_vp_1000_v1/deployment"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@torch.inference_mode()
def predict_one(model: torch.nn.Module, path: Path, device: torch.device, meta: dict, head: dict[str, np.ndarray]) -> float:
    tensor, _, _ = preprocess(path, int(meta["feature_max_edge"]), device)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
    with context:
        output = model(pixel_values=tensor, return_dict=True)
    hidden = output.last_hidden_state[0].float()
    if meta["feature_view"] == "cls":
        feature = hidden[0].cpu().numpy()
    elif meta["feature_view"] == "patch_mean":
        feature = hidden[5:].mean(dim=0).cpu().numpy()
    elif meta["feature_view"] == "cls_patch":
        feature = torch.cat((hidden[0], hidden[5:].mean(dim=0))).cpu().numpy()
    else:
        raise ValueError(f"Unknown feature view: {meta['feature_view']}")
    standardized = (feature - head["scaler_mean"]) / head["scaler_scale"]
    return float(standardized @ head["coefficient"] + float(meta["intercept"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="An image file or a directory of images")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    meta = json.loads((args.model_dir / "model.json").read_text(encoding="utf-8"))
    head = dict(np.load(args.model_dir / "ridge_head.npz"))
    expected_hash = meta["feature_checkpoint_sha256"]
    actual_hash = checkpoint_sha256(args.checkpoint)
    if actual_hash != expected_hash:
        raise ValueError(f"ViT checkpoint hash mismatch: expected {expected_hash}, got {actual_hash}")
    if args.input.is_file():
        images = [args.input]
    elif args.input.is_dir():
        images = sorted(p for p in args.input.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        raise FileNotFoundError(args.input)
    if not images:
        raise ValueError(f"No images found under {args.input}")
    device = require_cuda(torch.device(args.device))
    encoder = make_model(args.checkpoint, device)
    rows = [
        {"image_id": path.stem, "image_path": str(path.resolve()), "predicted_vp_score": predict_one(encoder, path, device, meta, head), "model": meta["model"]}
        for path in images
    ]
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} predictions to {args.output}")
    else:
        print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
