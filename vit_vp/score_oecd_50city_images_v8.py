#!/usr/bin/env python3
"""Score the OECD-linked Mapillary image cohort with the final V8 ViT ensemble."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file
from torch import nn

from extract_features import DEFAULT_CHECKPOINT, checkpoint_sha256, make_model, preprocess, require_cuda


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "output/vit_vp_v8_1200_citycv/deployment/finetuned_last2_v8"


class RegressionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(768, 128), nn.GELU(), nn.Dropout(0.5), nn.Linear(128, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)


def load_model_seed(checkpoint: Path, weight_path: Path, device: torch.device):
    payload = load_file(str(weight_path), device="cpu")
    encoder_state = {k.removeprefix("encoder."): v for k, v in payload.items() if k.startswith("encoder.")}
    head_state = {k.removeprefix("head."): v for k, v in payload.items() if k.startswith("head.")}
    encoder = make_model(checkpoint, device)
    result = encoder.load_state_dict(encoder_state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        expected_missing = {k for k in encoder.state_dict() if not (k.startswith("model.layer.10.") or k.startswith("model.layer.11."))}
        if set(result.missing_keys) != expected_missing or result.unexpected_keys:
            raise RuntimeError(f"V8 encoder state mismatch: {result}")
    head = RegressionHead().to(device)
    head.load_state_dict(head_state, strict=True)
    return encoder.eval(), head.eval()


def preprocess_cpu(path: Path, edge: int) -> tuple[torch.Tensor, int, int]:
    """Same transform as the V8 training cache: EXIF, RGB, long-edge, patch aligned, bicubic, ImageNet norm."""
    tensor, width, height = preprocess(path, edge, torch.device("cpu"))
    return tensor, width, height


@torch.inference_mode()
def score_group(batch: torch.Tensor, models, device: torch.device) -> dict[int, list[float]]:
    x = batch.to(device, non_blocking=True)
    per_seed: dict[int, list[float]] = {}
    for artifact, encoder, head in models:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = encoder(pixel_values=x, output_hidden_states=False, return_dict=True)
        patch_mean = output.last_hidden_state[:, 5:, :].float().mean(dim=1)
        standardized = head(patch_mean).float()
        raw = standardized * float(artifact["target_std"]) + float(artifact["target_mean"])
        per_seed[int(artifact["seed"])] = raw.cpu().tolist()
        del output, patch_mean, standardized, raw
    return per_seed


def atomic_write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temp = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 16:
        raise ValueError("--batch-size must be 1 through 16")

    meta = json.loads((args.model_dir / "model.json").read_text(encoding="utf-8"))
    if meta.get("training_images") != 1200 or "V8" not in meta.get("label", ""):
        raise ValueError("The deployment metadata does not identify the full-data V8 model")
    edge = int(meta["feature_cache_edge"])
    if meta.get("prediction_rule") != "arithmetic mean of the three seed-model predictions":
        raise ValueError("Unexpected V8 ensemble rule")
    if checkpoint_sha256(args.checkpoint) != meta["source_checkpoint_sha256"]:
        raise ValueError("The original DINOv3 checkpoint hash differs from the V8 deployment metadata")

    manifest = pd.read_csv(args.manifest, dtype={"uuid": str, "sequence_id": str, "oecd_region_code": str})
    required = {"uuid", "city", "country", "oecd_region", "oecd_region_code", "orig_id", "lat", "lon"}
    if missing := required - set(manifest.columns):
        raise ValueError(f"Image manifest lacks columns: {sorted(missing)}")
    if args.limit:
        manifest = manifest.head(args.limit).copy()
    if manifest.uuid.duplicated().any() or manifest.orig_id.duplicated().any():
        raise ValueError("Image manifest has duplicate UUIDs or Mapillary IDs")

    device = require_cuda(torch.device(args.device))
    models = []
    for artifact in meta["artifacts"]:
        weights = args.model_dir / artifact["weights"]
        if checkpoint_sha256(weights) != artifact["sha256"]:
            raise ValueError(f"V8 deployment weight hash mismatch: {weights}")
        encoder, head = load_model_seed(args.checkpoint, weights, device)
        models.append((artifact, encoder, head))
    if len(models) != 3 or {int(x[0]["seed"]) for x in models} != {42, 137, 2026}:
        raise ValueError("Expected V8 deployment seed ensemble {42, 137, 2026}")

    done: dict[str, dict] = {}
    if args.output.is_file():
        prior = pd.read_csv(args.output, dtype={"image_id": str})
        done = {str(row.image_id): row._asdict() for row in prior.itertuples(index=False)}
    rows = []
    pending = [(i, row) for i, row in enumerate(manifest.to_dict("records")) if str(row["uuid"]) not in done]
    for offset in range(0, len(pending), 64):
        chunk = pending[offset : offset + 64]
        groups: dict[tuple[int, int], list[tuple[int, dict, torch.Tensor]]] = defaultdict(list)
        for index, row in chunk:
            image_path = args.image_dir / f"{row['uuid']}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing downloaded image: {image_path}")
            pixels, width, height = preprocess_cpu(image_path, edge)
            if width % 16 or height % 16:
                raise ValueError(f"Patch grid misalignment: {row['uuid']} {width}x{height}")
            groups[(width, height)].append((index, row, pixels))
        for _, group in groups.items():
            for start in range(0, len(group), args.batch_size):
                batch_rows = group[start : start + args.batch_size]
                tensors = torch.cat([r[2] for r in batch_rows], dim=0)
                predictions = score_group(tensors, models, device)
                for position, (manifest_index, row, _) in enumerate(batch_rows):
                    seed_scores = {f"seed_{seed}_score": predictions[seed][position] for seed in sorted(predictions)}
                    combined = float(np.mean([predictions[s][position] for s in sorted(predictions)]))
                    rows.append({
                        "manifest_index": manifest_index,
                        "image_id": str(row["uuid"]),
                        "mapillary_image_id": int(row["orig_id"]),
                        "image_path": str((args.image_dir / f"{row['uuid']}.jpg").resolve()),
                        "city": row["city"], "country": row["country"],
                        "oecd_region": row["oecd_region"], "oecd_region_code": row["oecd_region_code"],
                        "lat": float(row["lat"]), "lon": float(row["lon"]),
                        "predicted_vp_score": combined, **seed_scores,
                        "model_version": "DINOv3 ViT-B/16 V8 all-1200 fit; 3-seed ensemble",
                        "input_long_edge": edge,
                    })
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        current = list(done.values()) + rows
        current.sort(key=lambda r: int(r["manifest_index"]))
        atomic_write(current, args.output)
        count = len(current)
        print(f"{count}/{len(manifest)} images scored on {device}; edge={edge}; batch={args.batch_size}", flush=True)
    final = list(done.values()) + rows
    final.sort(key=lambda r: int(r["manifest_index"]))
    if len(final) != len(manifest) or len({r["image_id"] for r in final}) != len(manifest):
        raise RuntimeError("Prediction coverage does not match the requested manifest")
    atomic_write(final, args.output)
    print(f"Complete: {len(final)} image scores -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
