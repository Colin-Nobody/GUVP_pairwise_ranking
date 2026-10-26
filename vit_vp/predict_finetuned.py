#!/usr/bin/env python3
"""Predict fixed-scale VP scores using the three-seed fine-tuned DINOv3 ensemble."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch import nn

from extract_features import DEFAULT_CHECKPOINT, checkpoint_sha256, make_model, preprocess, require_cuda

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "output/vit_vp_1000_v1/deployment/finetuned_last2"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class RegressionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(768, 128), nn.GELU(), nn.Dropout(0.5), nn.Linear(128, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)


def load_seed_model(checkpoint: Path, weights_path: Path, device: torch.device):
    payload = load_file(str(weights_path), device="cpu")
    encoder_state = {key.removeprefix("encoder."): value for key, value in payload.items() if key.startswith("encoder.")}
    head_state = {key.removeprefix("head."): value for key, value in payload.items() if key.startswith("head.")}
    encoder = make_model(checkpoint, device)
    result = encoder.load_state_dict(encoder_state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        expected_missing = {key for key in encoder.state_dict() if not (key.startswith("model.layer.10.") or key.startswith("model.layer.11."))}
        if set(result.missing_keys) != expected_missing or result.unexpected_keys:
            raise RuntimeError(f"Fine-tuned encoder state mismatch: {result}")
    head = RegressionHead().to(device)
    head.load_state_dict(head_state, strict=True)
    return encoder.eval(), head.eval()


@torch.inference_mode()
def predict_one(path: Path, ensemble, device: torch.device, meta: dict) -> float:
    tensor, _, _ = preprocess(path, int(meta["feature_cache_edge"]), device)
    predictions = []
    for encoder, head, seed_meta in ensemble:
        context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
        with context:
            output = encoder(pixel_values=tensor, return_dict=True)
        feature = output.last_hidden_state[0, 5:, :].float().mean(dim=0)
        standardized = head(feature).float().item()
        predictions.append(standardized * seed_meta["target_std"] + seed_meta["target_mean"])
    return float(np.mean(predictions))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Image file or directory")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    meta = json.loads((args.model_dir / "model.json").read_text(encoding="utf-8"))
    if checkpoint_sha256(args.checkpoint) != meta["source_checkpoint_sha256"]:
        raise ValueError("Pretrained DINOv3 checkpoint SHA-256 does not match the deployment metadata")
    if args.input.is_file():
        images = [args.input]
    elif args.input.is_dir():
        images = sorted(path for path in args.input.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    else:
        raise FileNotFoundError(args.input)
    if not images:
        raise ValueError(f"No supported images found at {args.input}")

    device = require_cuda(torch.device(args.device))
    ensemble = []
    for artifact in meta["artifacts"]:
        weights = args.model_dir / artifact["weights"]
        if checkpoint_sha256(weights) != artifact["sha256"]:
            raise ValueError(f"Fine-tuned checkpoint hash mismatch: {weights}")
        encoder, head = load_seed_model(args.checkpoint, weights, device)
        ensemble.append((encoder, head, artifact))

    rows = []
    for index, path in enumerate(images, start=1):
        rows.append({"image_id": path.stem, "image_path": str(path.resolve()),
                     "predicted_vp_score": predict_one(path, ensemble, device, meta),
                     "model": meta["model"]})
        if index % 25 == 0 or index == len(images):
            print(f"{index}/{len(images)} images scored", flush=True)
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
