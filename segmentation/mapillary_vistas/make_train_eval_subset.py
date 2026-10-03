#!/usr/bin/env python3
"""Create a deterministic Mapillary training subset for checkpoint evaluation.

The file contains image stems, the format expected by MMsegmentation's
``ann_file`` option.  Ranking by a stable SHA-256 key keeps the exact subset
reproducible regardless of filesystem traversal order.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/home/colin/datasets/mapillary_vistas/training/images"),
    )
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/mapillary_vistas_train_eval_2000.txt"),
    )
    parser.add_argument("--seed", default="guvp-mapillary-train-eval-v1")
    args = parser.parse_args()

    stems = sorted(path.stem for path in args.images_dir.glob("*.jpg"))
    if len(stems) < args.count:
        raise SystemExit(f"Requested {args.count} images but found only {len(stems)}")
    ranked = sorted(
        stems,
        key=lambda stem: hashlib.sha256(f"{args.seed}:{stem}".encode()).hexdigest(),
    )
    selected = sorted(ranked[: args.count])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(selected) + "\n")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"Wrote {len(selected)} image stems to {args.output} (sha256={digest})")


if __name__ == "__main__":
    main()
