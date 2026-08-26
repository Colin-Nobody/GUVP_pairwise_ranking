"""Create auditable Qwen VLM pairwise votes for street-view images."""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
PROMPT_VERSION = "vp-overall-minimal-v2"
PROMPT = (
    "Compare the two street-view images fairly. Which image has LOWER overall visual pollution and looks cleaner? "
    "Return exactly: A or B."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, help="Optional CSV with image_a,image_b filenames")
    parser.add_argument("--output", type=Path, default=Path("data/vlm_pairwise_votes.csv"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--all-pairs", action="store_true", help="Compare every unique unordered image pair")
    parser.add_argument("--resume", action="store_true", help="Append only pairs not already present in --output")
    parser.add_argument("--overwrite", action="store_true", help="Replace --output if it already exists")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N completed pairs (0 disables)")
    parser.add_argument("--progress-file", type=Path, help="Optional local file updated whenever progress is printed")
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Keep the input embedding and output head on CPU (useful for Qwen3-VL 8B on an 8GB GPU)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_images(directory: Path) -> dict[str, Path]:
    files = {item.name: item for item in sorted(directory.iterdir()) if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES}
    if len(files) < 2:
        raise ValueError(f"Need at least two images in {directory}; found {len(files)}")
    return files


def get_pairs(args: argparse.Namespace, files: dict[str, Path]) -> list[tuple[str, str]]:
    if args.pairs:
        with args.pairs.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"image_a", "image_b"}.issubset(reader.fieldnames):
                raise ValueError("--pairs must contain image_a,image_b columns")
            pairs = [(row["image_a"].strip(), row["image_b"].strip()) for row in reader]
    else:
        all_pairs = list(itertools.combinations(files, 2))
        order = torch.randperm(len(all_pairs), generator=torch.Generator().manual_seed(args.seed)).tolist()
        pairs = [all_pairs[item] for item in order] if args.all_pairs else [all_pairs[item] for item in order[: args.max_pairs]]
    if not pairs:
        raise ValueError("No image pairs to compare")
    for image_a, image_b in pairs:
        if image_a == image_b or image_a not in files or image_b not in files:
            raise ValueError(f"Invalid image pair: {image_a!r}, {image_b!r}")
    return pairs


def choice_from(response: str) -> str | None:
    match = re.fullmatch(r"\s*([AB])\s*[.!。]?\s*", response.upper())
    return match.group(1) if match else None


def infer(model, processor, image_a: Path, image_b: Path, max_pixels: int) -> tuple[str | None, str]:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(image_a), "max_pixels": max_pixels},
        {"type": "text", "text": "Image A"},
        {"type": "image", "image": str(image_b), "max_pixels": max_pixels},
        {"type": "text", "text": f"Image B\n\n{PROMPT}"},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    response = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    return choice_from(response), response


def main() -> None:
    args = parse_args()
    files = get_images(args.images_dir)
    pairs = get_pairs(args, files)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")
    if args.resume and not args.output.exists():
        raise ValueError(f"Cannot resume: output does not exist: {args.output}")
    existing_pairs: set[frozenset[str]] = set()
    existing_count = 0
    if args.resume:
        with args.output.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"image_a", "image_b"}.issubset(reader.fieldnames):
                raise ValueError("Existing --output must contain image_a,image_b columns")
            for row in reader:
                existing_pairs.add(frozenset((row["image_a"], row["image_b"])))
                existing_count += 1
        pairs = [pair for pair in pairs if frozenset(pair) not in existing_pairs]
    elif args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}. Use a new path, --overwrite, or --resume.")
    if args.progress_file:
        args.progress_file.parent.mkdir(parents=True, exist_ok=True)
        args.progress_file.write_text(f"Starting {existing_count + len(pairs)} comparisons\n", encoding="utf-8")
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=args.cpu_offload,
    )
    device_map = "auto"
    if args.cpu_offload:
        device_map = {
            "model.visual": 0,
            "model.language_model.embed_tokens": "cpu",
            "model.language_model.layers": 0,
            "model.language_model.norm": 0,
            "model.language_model.rotary_emb": 0,
            "lm_head": "cpu",
        }
    # AutoModel keeps the same evaluation path usable with both Qwen2.5-VL and
    # Qwen3-VL instead of hard-coding a generation-specific model class.
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        quantization_config=config,
        dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["comparison_id", "image_a", "image_b", "winner", "model", "prompt_version", "max_pixels", "timestamp_utc", "raw_response", "status"]
    mode = "a" if args.resume else "w"
    with args.output.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not args.resume:
            writer.writeheader()
        total = existing_count + len(pairs)
        for index, (image_a, image_b) in enumerate(pairs, 1):
            choice, response = infer(model, processor, files[image_a], files[image_b], args.max_pixels)
            writer.writerow({"comparison_id": f"vlm-{existing_count + index:05d}", "image_a": image_a, "image_b": image_b,
                "winner": image_a if choice == "A" else image_b if choice == "B" else "", "model": args.model,
                "prompt_version": PROMPT_VERSION, "max_pixels": args.max_pixels,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), "raw_response": response,
                "status": "ok" if choice else "invalid_model_response"})
            handle.flush()
            if index % 100 == 0:
                os.fsync(handle.fileno())
            if args.progress_every and (index % args.progress_every == 0 or index == len(pairs)):
                progress = f"Completed {existing_count + index}/{total} comparisons"
                print(progress, flush=True)
                if args.progress_file:
                    with args.progress_file.open("a", encoding="utf-8") as progress_handle:
                        progress_handle.write(progress + "\n")
    print(f"Wrote {len(pairs)} comparisons to {args.output}")
    if torch.cuda.is_available():
        peak_gib = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak CUDA memory allocated: {peak_gib:.2f} GiB")


if __name__ == "__main__":
    main()
