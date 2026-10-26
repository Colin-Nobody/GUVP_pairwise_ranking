#!/usr/bin/env python3
"""Run a mixed A/B position and visual-access audit with a Qwen VLM."""

from __future__ import annotations

import argparse
import csv
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Deprecated and rejected: project inference requires GPU-only execution")
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--label-layout", choices=("after", "before", "anchored"), default="after")
    parser.add_argument("--vp-prompt-file", type=Path, help="Override the prompt for natural and identical VP cases")
    return parser.parse_args()


def choice_from(response: str) -> str | None:
    match = re.fullmatch(r"\s*([ABT])\s*[.!。]?\s*", response.upper())
    return match.group(1) if match else None


def infer(model, processor, row: dict[str, str], max_pixels: int, label_layout: str) -> tuple[str | None, str]:
    if label_layout == "anchored":
        content = [
            {"type": "text", "text": "BEGIN IMAGE A. Bind the immediately following image to the neutral identifier A."},
            {"type": "image", "image": row["image_a"], "max_pixels": max_pixels},
            {"type": "text", "text": "END IMAGE A.\nBEGIN IMAGE B. Bind the immediately following image to the neutral identifier B."},
            {"type": "image", "image": row["image_b"], "max_pixels": max_pixels},
            {"type": "text", "text": f"END IMAGE B.\n\n{row['prompt']}"},
        ]
    elif label_layout == "before":
        content = [
            {"type": "text", "text": "Image A:"},
            {"type": "image", "image": row["image_a"], "max_pixels": max_pixels},
            {"type": "text", "text": "Image B:"},
            {"type": "image", "image": row["image_b"], "max_pixels": max_pixels},
            {"type": "text", "text": row["prompt"]},
        ]
    else:
        content = [
            {"type": "image", "image": row["image_a"], "max_pixels": max_pixels},
            {"type": "text", "text": "Image A"},
            {"type": "image", "image": row["image_b"], "max_pixels": max_pixels},
            {"type": "text", "text": f"Image B\n\n{row['prompt']}"},
        ]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    response = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    return choice_from(response), response


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only inference is required, but CUDA is unavailable")
    if args.cpu_offload:
        raise ValueError("--cpu-offload is disabled by the project GPU-only policy")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be combined")
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        manifest_fields = list(reader.fieldnames or [])
        rows = list(reader)
    required = {"test_id", "case_id", "case_type", "order", "image_a", "image_b", "expected_image", "prompt"}
    if not required.issubset(manifest_fields):
        raise ValueError(f"Manifest missing: {sorted(required - set(manifest_fields))}")
    if args.vp_prompt_file:
        vp_prompt = args.vp_prompt_file.read_text(encoding="utf-8").strip()
        for row in rows:
            if row["case_type"] in {"natural_vp", "identical_vp"}:
                row["prompt"] = vp_prompt

    completed: set[str] = set()
    existing_count = 0
    if args.resume:
        if not args.output.exists():
            raise ValueError("Cannot resume a missing output")
        with args.output.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                completed.add(row["test_id"])
                existing_count += 1
        rows = [row for row in rows if row["test_id"] not in completed]
    elif args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}")

    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=False,
    )
    device_index = torch.cuda.current_device()
    device_map = device_index
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        quantization_config=config,
        dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    placement = getattr(model, "hf_device_map", None)
    is_gpu = lambda device: isinstance(device, int) or (
        isinstance(device, str) and (device == "cuda" or device.startswith("cuda:"))
    )
    if not placement or any(not is_gpu(device) for device in placement.values()):
        raise RuntimeError(f"GPU-only model placement check failed: {placement}")
    print(f"GPU-only device placement verified across {len(placement)} model modules", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels, local_files_only=True)

    output_fields = manifest_fields + ["choice", "chosen_image", "model", "label_layout", "max_pixels", "timestamp_utc", "raw_response", "status"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with args.output.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        if not args.resume:
            writer.writeheader()
        total = existing_count + len(rows)
        for index, row in enumerate(rows, 1):
            choice, response = infer(model, processor, row, args.max_pixels, args.label_layout)
            chosen = row["image_a"] if choice == "A" else row["image_b"] if choice == "B" else ""
            writer.writerow({**row, "choice": choice or "", "chosen_image": chosen, "model": args.model,
                "label_layout": args.label_layout,
                "max_pixels": args.max_pixels, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "raw_response": response, "status": "ok" if choice else "invalid_model_response"})
            handle.flush()
            if index % 20 == 0:
                os.fsync(handle.fileno())
            if args.progress_every and (index % args.progress_every == 0 or index == len(rows)):
                print(f"Completed {existing_count + index}/{total}", flush=True)
    print(f"Wrote {len(rows)} tests to {args.output}")


if __name__ == "__main__":
    main()
