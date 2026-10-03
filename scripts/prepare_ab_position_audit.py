#!/usr/bin/env python3
"""Prepare deterministic natural, identical-image, and visual-access A/B controls."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


VP_PROMPT = (
    "Compare the two street-view images fairly. Which image has LOWER overall visual pollution and looks cleaner? "
    "Return exactly: A or B."
)
ACCESS_PROMPT = (
    "Which image contains the large solid magenta rectangle labeled TARGET? Return exactly: A or B."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("data/vlm_pairwise_formal_v2.csv"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/street_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/ab_position_audit"))
    parser.add_argument("--natural-pairs", type=int, default=160)
    parser.add_argument("--identical-controls", type=int, default=40)
    parser.add_argument("--visual-controls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def marked_copy(source: Path, output: Path) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    x0, y0 = int(width * 0.08), int(height * 0.08)
    x1, y1 = int(width * 0.55), int(height * 0.42)
    border = max(8, min(width, height) // 80)
    draw.rectangle((x0, y0, x1, y1), fill=(255, 0, 255), outline=(0, 0, 0), width=border)
    font = ImageFont.load_default(size=max(18, min(width, height) // 18))
    draw.text((x0 + border * 2, y0 + border * 2), "TARGET", fill=(0, 0, 0), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=95)


def audit_row(
    test_id: str,
    case_id: str,
    case_type: str,
    order: str,
    image_a: Path,
    image_b: Path,
    prompt: str,
    expected_image: Path | None = None,
) -> dict[str, str]:
    return {
        "test_id": test_id,
        "case_id": case_id,
        "case_type": case_type,
        "order": order,
        "image_a": image_a.as_posix(),
        "image_b": image_b.as_posix(),
        "expected_image": expected_image.as_posix() if expected_image else "",
        "prompt": prompt,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    with args.pairs.open(newline="", encoding="utf-8") as handle:
        pairs = list(csv.DictReader(handle))
    images = sorted(path for path in args.images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if args.natural_pairs > len(pairs):
        raise ValueError("--natural-pairs exceeds available pairs")
    if max(args.identical_controls, args.visual_controls) > len(images):
        raise ValueError("Not enough images for requested controls")

    rows: list[dict[str, str]] = []
    for index, pair in enumerate(rng.sample(pairs, args.natural_pairs), 1):
        first = args.images_dir / pair["image_a"]
        second = args.images_dir / pair["image_b"]
        case_id = f"natural-{index:04d}"
        rows.append(audit_row(f"{case_id}-ab", case_id, "natural_vp", "AB", first, second, VP_PROMPT))
        rows.append(audit_row(f"{case_id}-ba", case_id, "natural_vp", "BA", second, first, VP_PROMPT))

    control_images = rng.sample(images, max(args.identical_controls, args.visual_controls))
    for index, source in enumerate(control_images[: args.identical_controls], 1):
        case_id = f"identical-{index:04d}"
        rows.append(audit_row(case_id, case_id, "identical_vp", "IDENTICAL", source, source, VP_PROMPT))

    controls_dir = args.output_dir / "controls"
    for index, source in enumerate(control_images[: args.visual_controls], 1):
        marked = controls_dir / f"marked-{index:04d}-{source.name}"
        marked_copy(source, marked)
        case_id = f"access-{index:04d}"
        rows.append(audit_row(f"{case_id}-ab", case_id, "visual_access", "AB", source, marked, ACCESS_PROMPT, marked))
        rows.append(audit_row(f"{case_id}-ba", case_id, "visual_access", "BA", marked, source, ACCESS_PROMPT, marked))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "audit_manifest.csv"
    fields = ["test_id", "case_id", "case_type", "order", "image_a", "image_b", "expected_image", "prompt"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} audit tests to {manifest}")


if __name__ == "__main__":
    main()
