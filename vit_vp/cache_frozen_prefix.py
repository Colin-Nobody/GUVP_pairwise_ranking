#!/usr/bin/env python3
"""Cache DINOv3 embeddings after the first ten frozen blocks for efficient V3 tuning."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from extract_features import DEFAULT_CHECKPOINT, make_model, preprocess, require_cuda

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "output/vit_vp_1000_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=AUDIT / "features" / "prefix_after_block10")
    parser.add_argument("--edges", type=int, nargs="+", choices=(384, 512), default=(384, 512))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = require_cuda(torch.device(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    manifest = pd.read_csv(args.audit_dir / "image_manifest.csv")
    if manifest.image_id.duplicated().any():
        raise ValueError("Duplicate image ID in manifest")
    model = make_model(args.checkpoint, device)
    model.requires_grad_(False)
    model.eval()
    if len(model.model.layer) != 12:
        raise ValueError("Expected ViT-B/16 with 12 Transformer blocks")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"model": "DINOv3 ViT-B/16 LVD-1689M", "frozen_blocks": list(range(10)),
                "cached_state": "hidden states after block 10 plus RoPE cos/sin for final two trainable blocks",
                "precision": "stored float16; generated with bfloat16 autocast on CUDA",
                "image_count": len(manifest), "edges": {}}
    started = time.perf_counter()
    for edge in args.edges:
        edge_dir = args.output_dir / f"edge{edge}"
        edge_dir.mkdir(parents=True, exist_ok=True)
        index_rows = []
        for n, row in enumerate(manifest.itertuples(index=False), start=1):
            image_id = str(row.image_id)
            target = edge_dir / f"{image_id}.npz"
            if not target.exists():
                pixel, width, height = preprocess(Path(row.image_path), edge, device)
                if device.type == "cuda":
                    context = torch.autocast("cuda", dtype=torch.bfloat16)
                else:
                    context = torch.autocast("cpu", enabled=False)
                with torch.inference_mode(), context:
                    hidden = model.embeddings(pixel)
                    rope = model.rope_embeddings(pixel)
                    for block in model.model.layer[:10]:
                        hidden = block(hidden, position_embeddings=rope)
                prefix = hidden[0].to(torch.float16).cpu().numpy()
                cos, sin = (part.to(torch.float16).cpu().numpy() for part in rope)
                np.savez(target, hidden=prefix, rope_cos=cos, rope_sin=sin,
                         image_width=np.int32(width), image_height=np.int32(height))
            with np.load(target) as payload:
                index_rows.append({"image_id": image_id, "file": target.name,
                                   "tokens": int(payload["hidden"].shape[0]),
                                   "patches": int(payload["rope_cos"].shape[0]),
                                   "image_width": int(payload["image_width"]),
                                   "image_height": int(payload["image_height"])})
            if n % 50 == 0 or n == len(manifest):
                print(f"edge {edge}: {n}/{len(manifest)} prefixes cached", flush=True)
        pd.DataFrame(index_rows).to_csv(edge_dir / "index.csv", index=False)
        metadata["edges"][str(edge)] = {"path": str(edge_dir.resolve()), "images": len(index_rows),
                                       "bytes": sum(p.stat().st_size for p in edge_dir.glob("*.npz")),
                                       "token_range": [min(r["tokens"] for r in index_rows), max(r["tokens"] for r in index_rows)]}
    metadata["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
