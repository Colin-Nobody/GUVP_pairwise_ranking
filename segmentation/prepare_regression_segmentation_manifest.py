#!/usr/bin/env python3
"""Create a complete 250-image inference manifest with the regression target."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores",
        type=Path,
        default=ROOT / "output/vlm_formal_v2/perception_scores_with_manifest.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.scores).copy()
    if len(frame) != 250 or frame["uuid"].nunique() != 250:
        raise ValueError("Expected exactly 250 unique scored images")
    if "visual_pollution_score_0_10" not in frame:
        raise ValueError("Scores input lacks visual_pollution_score_0_10")

    rows = []
    for order, (_, row) in enumerate(frame.sort_values("uuid").iterrows(), start=1):
        image_id = str(row["uuid"])
        path = ROOT / "data/street_images" / f"{image_id}.jpg"
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            width, height = image.size
        record = row.to_dict()
        record.update(
            audit_order=order,
            image_id=image_id,
            image_path=str(path.relative_to(ROOT)),
            width=width,
            height=height,
            # Compatibility metadata for the shared inference writer.
            vp_stratum="all",
            vp_score_v8=float(row["visual_pollution_score_0_10"]),
        )
        rows.append(record)

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
