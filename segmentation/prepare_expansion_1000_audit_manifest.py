#!/usr/bin/env python3
"""Select a deterministic 30-image segmentation audit from the 1000-image set."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/expansion_750/images_manifest_1000.csv",
    )
    parser.add_argument(
        "--scores", type=Path,
        default=ROOT / "output/vlm_expansion_1000/perception_scores.csv",
    )
    parser.add_argument(
        "--image-dir", type=Path,
        default=ROOT / "data/expansion_750/street_images_1000",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "output/segmentation_expansion_1000_audit",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    scores = pd.read_csv(args.scores)
    scores["uuid"] = scores["image_id"].str.replace(r"\.[^.]+$", "", regex=True)
    scores = scores[["uuid", "visual_pollution_score_0_10", "rank_by_visual_pollution"]]
    frame = manifest.merge(scores, on="uuid", validate="one_to_one")
    if len(frame) != 1000 or frame["city"].nunique() != 20:
        raise ValueError(f"Expected 1000 images in 20 cities; got {len(frame)} rows")

    selected: list[pd.Series] = []
    # One median-score anchor per city ensures broad geographic coverage.
    for city, city_frame in frame.groupby("city", sort=True):
        city_frame = city_frame.copy()
        city_frame["median_distance"] = (
            city_frame["visual_pollution_score_0_10"]
            - city_frame["visual_pollution_score_0_10"].median()
        ).abs()
        row = city_frame.sort_values(["median_distance", "uuid"]).iloc[0].copy()
        row["vp_stratum"] = "mid"
        row["selection_role"] = "city_median_anchor"
        selected.append(row)

    # Add low/high stress cases in ten cities, alternating for geographic balance.
    for index, (city, city_frame) in enumerate(frame.groupby("city", sort=True)):
        if index >= 10:
            break
        city_frame = city_frame.sort_values(
            ["visual_pollution_score_0_10", "uuid"], ascending=[True, True]
        )
        choose_high = index % 2 == 1
        row = (city_frame.iloc[-1] if choose_high else city_frame.iloc[0]).copy()
        row["vp_stratum"] = "high" if choose_high else "low"
        row["selection_role"] = "city_score_extreme"
        selected.append(row)

    audit = pd.DataFrame(selected).reset_index(drop=True)
    audit.insert(0, "audit_order", range(1, len(audit) + 1))
    audit["image_id"] = audit["uuid"].astype(str)
    image_paths: list[str] = []
    widths: list[int] = []
    heights: list[int] = []
    for uuid in audit["uuid"]:
        path = args.image_dir / f"{uuid}.jpg"
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            width, height = image.size
        image_paths.append(str(path.resolve().relative_to(ROOT)))
        widths.append(width)
        heights.append(height)
    audit["image_path"] = image_paths
    audit["width"] = widths
    audit["height"] = heights
    audit["megapixels"] = audit["width"] * audit["height"] / 1_000_000
    audit["aspect_ratio"] = audit["width"] / audit["height"]
    audit["vp_score_0_10"] = audit["visual_pollution_score_0_10"]
    audit["selection_reason"] = audit.apply(
        lambda row: (
            "city median VLM score" if row["selection_role"] == "city_median_anchor"
            else f"city {row['vp_stratum']} score stress case"
        ), axis=1,
    )
    columns = [
        "audit_order", "image_id", "image_path", "city", "vp_stratum",
        "selection_role", "selection_reason", "vp_score_0_10",
        "rank_by_visual_pollution", "uuid", "source", "orig_id", "platform",
        "quality", "lighting_condition", "weather", "pano_status",
        "projection_type", "width", "height", "megapixels", "aspect_ratio",
        "lat", "lon",
    ]
    audit = audit[columns]
    if len(audit) != 30 or audit["image_id"].duplicated().any():
        raise RuntimeError("Audit selection must contain 30 unique images")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_dir / "audit_manifest.csv", index=False)
    audit.to_csv(args.output_dir / "segmentation_manifest.csv", index=False)
    audit[["audit_order", "image_id"]].rename(
        columns={"audit_order": "sample_number"}
    ).to_csv(args.output_dir / "selected_30.csv", index=False)
    print(f"Wrote {len(audit)} audit images to {args.output_dir}")
    print("Cities represented:", audit["city"].nunique())
    print("Score strata:", audit["vp_stratum"].value_counts().to_dict())


if __name__ == "__main__":
    main()
