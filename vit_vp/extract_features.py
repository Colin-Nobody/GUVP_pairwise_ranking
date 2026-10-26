#!/usr/bin/env python3
"""Extract deterministic DINOv3 ViT-B/16 CLS and patch-mean features."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.torch import load_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = ROOT / "output/vit_vp_1000_v1"
DEFAULT_CHECKPOINT = ROOT / "models/checkpoints/dinov3_vitb.safetensors"


def require_cuda(device: torch.device) -> torch.device:
    """Reject CPU execution for DINOv3 model forwards and training."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"GPU-only model execution is required; selected device is {device}")
    return device


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_model(path: Path, device: torch.device) -> DINOv3ViTModel:
    device = require_cuda(device)
    config = DINOv3ViTConfig(
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        patch_size=16,
        num_register_tokens=4,
        image_size=512,
        rope_theta=100.0,
        pos_embed_rescale=2.0,
        query_bias=True,
        key_bias=False,
        value_bias=True,
        proj_bias=True,
        mlp_bias=True,
        layerscale_value=1.0,
        use_gated_mlp=False,
        apply_layernorm=True,
    )
    model = DINOv3ViTModel(config)
    state = load_file(str(path), device="cpu")
    remapped = {(f"model.{key}" if key.startswith("layer.") else key): value for key, value in state.items()}
    incompatible = model.load_state_dict(remapped, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint architecture mismatch: {incompatible}")
    del state, remapped
    return model.eval().to(device)


def preprocess(path: Path, max_edge: int, device: torch.device) -> tuple[torch.Tensor, int, int]:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        original_w, original_h = image.size
        scale = max_edge / max(original_w, original_h)
        width = max(16, int(round(original_w * scale / 16)) * 16)
        height = max(16, int(round(original_h * scale / 16)) * 16)
        image = image.resize((width, height), Image.Resampling.BICUBIC)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
    return ((tensor - mean) / std).unsqueeze(0).to(device), width, height


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-edge", type=int, choices=(384, 512), default=512)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N images for a smoke run")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.max_edge % 16:
        raise ValueError("max-edge must be divisible by the 16-pixel patch size")

    import csv

    with (args.audit_dir / "image_manifest.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[: args.limit]
    device = require_cuda(torch.device(args.device))
    model = make_model(args.checkpoint, device)
    cls_features: list[np.ndarray] = []
    patch_features: list[np.ndarray] = []
    result_ids: list[str] = []
    started = time.perf_counter()
    max_memory_bytes = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for index, row in enumerate(rows, start=1):
        image, width, height = preprocess(Path(row["image_path"]), args.max_edge, device)
        if width % 16 or height % 16:
            raise AssertionError(f"Patch grid misalignment for {row['image_id']}: {width}x{height}")
        context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
        with context:
            output = model(pixel_values=image, output_hidden_states=False, return_dict=True)
        hidden = output.last_hidden_state[0].float()
        cls_features.append(hidden[0].cpu().numpy())
        patch_features.append(hidden[5:].mean(dim=0).cpu().numpy())
        result_ids.append(row["image_id"])
        if device.type == "cuda":
            max_memory_bytes = max(max_memory_bytes, torch.cuda.max_memory_allocated(device))
        if index % 25 == 0 or index == len(rows):
            elapsed = time.perf_counter() - started
            print(f"{index}/{len(rows)} images; {elapsed:.1f}s elapsed", flush=True)

    cls = np.stack(cls_features).astype(np.float32)
    patch_mean = np.stack(patch_features).astype(np.float32)
    combined = np.concatenate((cls, patch_mean), axis=1).astype(np.float32)
    target = args.output or args.audit_dir / "features" / f"dinov3_vitb16_lvd1689m_edge{args.max_edge}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, image_ids=np.asarray(result_ids), cls=cls, patch_mean=patch_mean, cls_patch=combined)
    metadata = {
        "model": "DINOv3 ViT-B/16 LVD-1689M",
        "source_repo": "ZhengGuangze/dinov3",
        "source_revision": "ec54a6b69c8ff6066571dec86ee2b7dd26836570",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "architecture_load": "strict after adding the Transformers DINOv3ViTModel model. prefix to checkpoint layer.* keys",
        "device": str(device),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "max_edge": args.max_edge,
        "resize": "EXIF transpose, RGB, preserve aspect ratio, longest edge target, dimensions rounded to nearest multiple of 16",
        "interpolation": "Pillow bicubic",
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "autocast": "bfloat16 on CUDA, float32 on CPU",
        "pooling": "final normalized CLS token; mean of image patch tokens after CLS and four register tokens",
        "image_count": len(result_ids),
        "output_npz": str(target.resolve()),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_cuda_allocated_bytes": max_memory_bytes,
        "feature_shapes": {"cls": list(cls.shape), "patch_mean": list(patch_mean.shape), "cls_patch": list(combined.shape)},
    }
    metadata_path = target.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
