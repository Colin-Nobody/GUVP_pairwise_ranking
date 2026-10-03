#!/usr/bin/env python3
"""Export an exact 512×512 sliding-window model input tile from an image."""

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--x", type=int, default=0)
    parser.add_argument("--y", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=512)
    args = parser.parse_args()

    with Image.open(args.image) as image:
        image = image.convert("RGB")
        if args.x < 0 or args.y < 0 or args.x + args.tile_size > image.width or args.y + args.tile_size > image.height:
            raise ValueError("Requested tile is outside the source image")
        tile = image.crop((args.x, args.y, args.x + args.tile_size, args.y + args.tile_size))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tile.save(args.output, quality=95)
    print(f"Wrote {tile.width}x{tile.height} tile to {args.output}")


if __name__ == "__main__":
    main()
