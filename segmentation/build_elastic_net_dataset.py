#!/usr/bin/env python3
"""Convert per-class segmentation ratios into an Elastic Net ready dataset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def feature_name(class_id: int, class_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", class_name.lower()).strip("_")
    return f"feature_ratio_{class_id:03d}_{slug}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--semantic-features", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--inference-mode",
        default="sliding_512_o25",
        help="Human-readable inference strategy recorded in the summary.",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    features = pd.read_csv(args.semantic_features)
    if len(manifest) != 250 or len(features) != 250:
        raise ValueError("Manifest and semantic feature table must each contain 250 rows")
    merged = manifest.merge(features, on="image_id", validate="one_to_one", suffixes=("", "_seg"))
    if len(merged) != 250:
        raise RuntimeError("Feature/manifest merge did not retain 250 rows")

    config = json.loads(args.run_config.read_text(encoding="utf-8"))
    classes = config["classes"]
    source_columns = [f"ratio__{name}" for name in classes]
    missing = [column for column in source_columns if column not in merged]
    if missing:
        raise ValueError(f"Missing ratio columns: {missing[:5]}")
    ratio_sum = merged[source_columns].sum(axis=1)
    if not ((ratio_sum - 1.0).abs() < 1e-6).all():
        raise ValueError("At least one image does not have class ratios summing to one")

    schema_rows = []
    renamed = {}
    for class_id, (class_name, source) in enumerate(zip(classes, source_columns)):
        destination = feature_name(class_id, class_name)
        renamed[source] = destination
        schema_rows.append(
            {
                "feature_column": destination,
                "class_id": class_id,
                "class_name": class_name,
                "source_column": source,
            }
        )

    target_columns = [
        "image_id",
        "city",
        "visual_pollution_score_0_10",
        "rank_by_visual_pollution",
        "q_score_0_10",
    ]
    if any(column not in merged for column in target_columns):
        raise ValueError("Manifest lacks one or more regression target columns")
    matrix = merged[target_columns + source_columns].rename(columns=renamed)
    feature_columns = list(renamed.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.output_dir / "elastic_net_ready.csv", index=False)
    matrix[["image_id"] + feature_columns].to_csv(args.output_dir / "elastic_net_features.csv", index=False)
    matrix[target_columns].to_csv(args.output_dir / "elastic_net_targets.csv", index=False)
    pd.DataFrame(schema_rows).to_csv(args.output_dir / "elastic_net_feature_schema.csv", index=False)
    percentage_rows = matrix[["image_id", "city", "visual_pollution_score_0_10"] + feature_columns].melt(
        id_vars=["image_id", "city", "visual_pollution_score_0_10"],
        value_vars=feature_columns,
        var_name="feature_column",
        value_name="pixel_ratio",
    )
    schema = pd.DataFrame(schema_rows)
    percentage_rows = percentage_rows.merge(schema, on="feature_column", validate="many_to_one")
    percentage_rows["pixel_percentage"] = percentage_rows["pixel_ratio"] * 100.0
    percentage_rows.to_csv(args.output_dir / "component_percentages_long.csv", index=False)
    (args.output_dir / "elastic_net_summary.json").write_text(
        json.dumps(
            {
                "image_count": len(matrix),
                "feature_count": len(feature_columns),
                "target_column": "visual_pollution_score_0_10",
                "feature_type": "per-image predicted semantic-class pixel ratio",
                "component_percentages_long": "pixel_percentage is in the 0-100 range",
                "inference_mode": args.inference_mode,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote Elastic Net tables for {len(matrix)} images and {len(feature_columns)} features")


if __name__ == "__main__":
    main()
