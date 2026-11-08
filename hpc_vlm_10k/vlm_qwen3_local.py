#!/usr/bin/env python3
"""Reuse HPC pairwise code with a verified local pre-quantized Qwen3 model."""
import os
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import vlm_pairwise as vlm

def load_local(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.model != "Qwen/Qwen3-VL-8B-Instruct":
        raise ValueError("This wrapper is dedicated to Qwen3-VL-8B")
    model_dir = Path(os.environ.get("QWEN3_LOCAL_DIR", "/share/home/sgxcw3/qwen3-nf4/qwen3_vl_8b_nf4_export"))
    if not (model_dir / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(model_dir)
    device = torch.cuda.current_device()
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir), device_map=device, local_files_only=True,
        attn_implementation="sdpa",
    )
    device_map = getattr(model, "hf_device_map", {})
    if not device_map or any(not (isinstance(x, int) or isinstance(x, str) and x.startswith("cuda")) for x in device_map.values()):
        raise RuntimeError(f"GPU-only placement failed: {device_map}")
    processor = AutoProcessor.from_pretrained(str(model_dir), max_pixels=args.max_pixels, local_files_only=True)
    print(f"Loaded pre-quantized model locally from {model_dir}; GPU placement {device_map}", flush=True)
    return model.eval(), processor

vlm.load_model = load_local
if __name__ == "__main__":
    vlm.main()
