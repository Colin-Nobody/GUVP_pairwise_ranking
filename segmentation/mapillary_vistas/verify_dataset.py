#!/usr/bin/env python3
"""Validate Mapillary v2.0 pairs and decode representative samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_ROOT = Path('/home/colin/datasets/mapillary_vistas')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--quick', action='store_true', help='Check names but decode only three masks per split.')
    return parser.parse_args()


def validate_split(root: Path, split: str, expected: int, quick: bool) -> None:
    image_dir = root / split / 'images'
    label_dir = root / split / 'v2.0' / 'labels'
    images = {path.stem: path for path in image_dir.glob('*.jpg')}
    labels = {path.stem: path for path in label_dir.glob('*.png')}
    if len(images) != expected or len(labels) != expected:
        raise RuntimeError(
            f'{split}: expected {expected} pairs, found {len(images)} images and {len(labels)} labels'
        )
    if images.keys() != labels.keys():
        missing_labels = sorted(images.keys() - labels.keys())[:10]
        missing_images = sorted(labels.keys() - images.keys())[:10]
        raise RuntimeError(
            f'{split}: unpaired files; missing labels={missing_labels}, missing images={missing_images}'
        )

    keys = sorted(images)
    inspect_keys = [keys[0], keys[len(keys) // 2], keys[-1]] if quick else keys
    observed: set[int] = set()
    for key in inspect_keys:
        with Image.open(images[key]) as image:
            image_size = image.size
        with Image.open(labels[key]) as label:
            mask = np.asarray(label)
        if mask.ndim != 2:
            raise RuntimeError(f'{labels[key]} is not a single-channel label image: {mask.shape}')
        if image_size != (mask.shape[1], mask.shape[0]):
            raise RuntimeError(f'{key}: image {image_size} != label {(mask.shape[1], mask.shape[0])}')
        if int(mask.max()) >= 124:
            raise RuntimeError(f'{key}: invalid class id {int(mask.max())}')
        observed.update(np.unique(mask).tolist())
    print(
        f'{split}: {len(images)} paired samples; decoded {len(inspect_keys)}; '
        f'observed label range {min(observed)}..{max(observed)}'
    )


def main() -> None:
    args = parse_args()
    config = json.loads((args.data_root / 'config_v2.0.json').read_text(encoding='utf-8'))
    if len(config['labels']) != 124:
        raise RuntimeError(f"Expected 124 v2.0 labels, got {len(config['labels'])}")
    validate_split(args.data_root, 'training', 18000, args.quick)
    validate_split(args.data_root, 'validation', 2000, args.quick)
    print('Mapillary Vistas v2.0 validation passed.')


if __name__ == '__main__':
    main()
