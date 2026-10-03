#!/usr/bin/env python3
"""Remove ADE20K classifier tensors before loading into a 124-class model."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    ROOT
    / 'segmentation/checkpoints/'
    'deeplabv3plus_r50-d8_512x512_80k_ade20k_20200614_185028-bf1400d8.pth'
)
DEFAULT_OUTPUT = (
    ROOT
    / 'segmentation/checkpoints/'
    'deeplabv3plus_r50-d8_ade20k_without_classifier.pth'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.source, map_location='cpu', weights_only=False)
    state = checkpoint['state_dict']
    removed = sorted(key for key in state if key.endswith('conv_seg.weight') or key.endswith('conv_seg.bias'))
    if len(removed) != 4:
        raise RuntimeError(f'Expected four classifier tensors, found: {removed}')
    for key in removed:
        del state[key]
    # This file is a weight initializer, not an ADE20K inference checkpoint.
    # Removing the old vocabulary prevents tools from mistaking its metadata
    # for the 124-class target vocabulary.
    checkpoint.setdefault('meta', {}).pop('CLASSES', None)
    checkpoint['meta'].pop('PALETTE', None)
    checkpoint['meta'].pop('dataset_meta', None)
    checkpoint.setdefault('meta', {})['transfer_note'] = (
        'ADE20K DeepLabV3+ R50-D8; decode and auxiliary conv_seg tensors removed '
        'for Mapillary Vistas v2.0 transfer.'
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f'Wrote {args.output}')
    print('Removed:')
    for key in removed:
        print(f'  {key}')


if __name__ == '__main__':
    main()
