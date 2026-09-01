#!/usr/bin/env python3
"""Validate one model's formal AB/BA inputs before loading model weights."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import vlm_pairwise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--model", required=True, choices=sorted(vlm_pairwise.SUPPORTED_MODELS))
    parser.add_argument("--expected-images", type=int, default=250)
    parser.add_argument("--expected-pairs", type=int, default=3000)
    parser.add_argument("--mec-samples", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = vlm_pairwise.get_images(args.images_dir)
    pair_args = argparse.Namespace(
        pairs=args.pairs, limit_pairs=None, all_pairs=False, max_pairs=0, seed=0,
    )
    pairs = vlm_pairwise.get_pairs(pair_args, files)
    if len(files) != args.expected_images:
        raise ValueError(f"Expected {args.expected_images} images, found {len(files)}")
    if len(pairs) != args.expected_pairs:
        raise ValueError(f"Expected {args.expected_pairs} pairs, found {len(pairs)}")

    participation = Counter(image for pair in pairs for image in pair)
    if set(participation) != set(files):
        missing = sorted(set(files) - set(participation))
        raise ValueError(f"Pair manifest does not cover every image; missing={missing[:10]}")
    if max(participation.values()) - min(participation.values()) > 1:
        raise ValueError("Pair participation differs by more than one comparison per image")

    print("Formal MEC+BPC preflight passed")
    print(f"Model: {args.model}")
    print(f"Images: {len(files)}")
    print(f"Unordered pairs: {len(pairs)}")
    print(f"MEC samples per position: {args.mec_samples}")
    print(f"Directional samples: {2 * args.mec_samples * len(pairs)}")
    print(f"Participation per image: {min(participation.values())}-{max(participation.values())}")
    print(f"Prompt version: {vlm_pairwise.PROMPT_VERSION}")
    print(f"Prompt SHA-256: {vlm_pairwise.PROMPT_SHA256}")


if __name__ == "__main__":
    main()
