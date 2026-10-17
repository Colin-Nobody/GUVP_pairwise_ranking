#!/usr/bin/env python3
"""Join expansion VLM targets to all images for frozen segmentation inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/expansion_750/images_manifest_1000.csv")
    parser.add_argument("--scores", type=Path, default=ROOT / "output/vlm_expansion_1000/perception_scores.csv")
    parser.add_argument("--image-dir", type=Path, default=ROOT / "data/expansion_750/street_images_1000")
    parser.add_argument("--output", type=Path, default=ROOT / "output/segmentation_expansion_1000/segmentation_manifest.csv")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    scores = pd.read_csv(args.scores)
    scores["uuid"] = scores["image_id"].str.replace(r"\.[^.]+$", "", regex=True)
    scores = scores[[
        "uuid", "rank_by_visual_pollution", "wins", "losses", "comparison_count",
        "q_score_0_10", "trueskill_mu", "trueskill_sigma", "visual_pollution_score_0_10",
    ]]
    frame = manifest.merge(scores, on="uuid", validate="one_to_one")
    if len(frame) != 1000 or frame["uuid"].nunique() != 1000:
        raise ValueError(f"Expected 1000 unique scored images; got {len(frame)}")

    frame = frame.sort_values(["city", "uuid"]).reset_index(drop=True)
    frame["image_id"] = frame["uuid"].astype(str)
    frame["vp_score_0_10"] = frame["visual_pollution_score_0_10"]
    frame["vp_stratum"] = ""
    for _, indices in frame.groupby("city", sort=True).groups.items():
        ranks = frame.loc[indices, "vp_score_0_10"].rank(method="first")
        frame.loc[indices, "vp_stratum"] = pd.qcut(
            ranks, q=5, labels=["q1", "q2", "q3", "q4", "q5"]
        ).astype(str).to_numpy()

    paths: list[str] = []
    widths: list[int] = []
    heights: list[int] = []
    for uuid in frame["uuid"]:
        path = args.image_dir / f"{uuid}.jpg"
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            width, height = image.size
        paths.append(str(path.resolve().relative_to(ROOT)))
        widths.append(width)
        heights.append(height)
    frame["image_path"] = paths
    frame["width"] = widths
    frame["height"] = heights
    frame["audit_order"] = range(1, len(frame) + 1)
    frame["dimension"] = "visual_pollution"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} images from {frame.city.nunique()} cities to {args.output}")
    print(frame.groupby("city").size().to_string())


if __name__ == "__main__":
    main()
