#!/usr/bin/env python3
"""Create a deterministic pair manifest with near-equal participation per image."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, default=3000,
                        help="3,000 pairs give exactly 24 appearances for each of 250 images")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def round_robin_rounds(items: list[str]) -> list[list[tuple[str, str]]]:
    if len(items) % 2:
        raise ValueError("Round-robin balancing requires an even number of images")
    rotation = list(items)
    rounds: list[list[tuple[str, str]]] = []
    for _ in range(len(items) - 1):
        rounds.append([
            (rotation[index], rotation[-1 - index])
            for index in range(len(items) // 2)
        ])
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    return rounds


def select_balanced_pairs(items: list[str], pair_count: int, seed: int) -> list[tuple[str, str]]:
    maximum = len(items) * (len(items) - 1) // 2
    if pair_count < 1 or pair_count > maximum:
        raise ValueError(f"pair_count must be in [1, {maximum}]")
    rng = random.Random(seed)
    shuffled_items = list(items)
    rng.shuffle(shuffled_items)
    rounds = round_robin_rounds(shuffled_items)
    rng.shuffle(rounds)

    full_rounds, remainder = divmod(pair_count, len(items) // 2)
    selected = [pair for round_pairs in rounds[:full_rounds] for pair in round_pairs]
    if remainder:
        partial = list(rounds[full_rounds])
        rng.shuffle(partial)
        selected.extend(partial[:remainder])

    oriented: list[tuple[str, str]] = []
    for first, second in selected:
        oriented.append((second, first) if rng.random() < 0.5 else (first, second))
    rng.shuffle(oriented)
    return oriented


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}. Use --overwrite or a new path.")
    images = sorted(
        item.name for item in args.images_dir.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(images) < 2:
        raise ValueError(f"Need at least two images, found {len(images)}")
    pairs = select_balanced_pairs(images, args.pair_count, args.seed)
    degrees = Counter(image for pair in pairs for image in pair)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["comparison_id", "image_a", "image_b"])
        writer.writeheader()
        for index, (image_a, image_b) in enumerate(pairs, 1):
            writer.writerow({
                "comparison_id": f"pair-{index:05d}",
                "image_a": image_a,
                "image_b": image_b,
            })
    print(f"Wrote {len(pairs)} unique unordered pairs for {len(images)} images to {args.output}")
    print(f"Participation per image: min={min(degrees.values())}, max={max(degrees.values())}, "
          f"mean={sum(degrees.values()) / len(degrees):.3f}")


if __name__ == "__main__":
    main()
