#!/usr/bin/env python3
"""Construct a 10%-prevalence core + grouped-rare semantic feature table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def rare_group(class_name: str) -> str:
    if class_name in {"Bird", "Ground Animal"}:
        return "rare_animals"
    if class_name.startswith("Lane Marking"):
        return "rare_lane_markings"
    if class_name in {"Mountain", "Sand", "Snow", "Water"}:
        return "rare_natural_elements"
    if class_name in {"Person Group", "Motorcyclist", "Other Rider"}:
        return "rare_road_users"
    if class_name in {
        "Ambiguous Barrier", "Concrete Block", "Guard Rail", "Barrier", "Road Median",
        "Road Side", "Lane Separator", "Temporary Barrier", "Curb Cut", "Driveway",
        "Parking Aisle", "Rail Track", "Road Shoulder", "Service Lane", "Garage", "Tunnel",
    }:
        return "rare_built_and_road_infrastructure"
    if (
        class_name.startswith("Signage")
        or class_name.startswith("Traffic Sign")
        or class_name.startswith("Traffic Light")
        or class_name == "Traffic Cone"
    ):
        return "rare_signage_and_traffic_control"
    if class_name in {
        "Banner", "Bench", "Bike Rack", "Catch Basin", "CCTV Camera", "Fire Hydrant",
        "Junction Box", "Mailbox", "Parking Meter", "Phone Booth", "Pothole", "Water Valve",
    }:
        return "rare_street_furniture_and_utilities"
    if class_name in {
        "Boat", "Bus", "Caravan", "On Rails", "Other Vehicle", "Trailer", "Wheeled Slow",
    }:
        return "rare_vehicles"
    return "rare_other_context"


def main() -> None:
    root = Path("/home/colin/GUVP_pairwise_ranking")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-run-dir", type=Path,
        default=root / "output/segmentation_mapillary_v2_finetune_3e-5_regression_250_baseline",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "output/semantic_features_finetune136k_core10_grouped",
    )
    parser.add_argument("--core-prevalence", type=float, default=0.10)
    args = parser.parse_args()
    source = args.source_run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(source / "elastic_net_ready.csv")
    schema = pd.read_csv(source / "elastic_net_feature_schema.csv")
    feature_columns = schema["feature_column"].tolist()
    prevalence = (table[feature_columns] > 0).mean(axis=0)
    core_schema = schema[schema["feature_column"].map(prevalence) >= args.core_prevalence].copy()

    metadata = table[[
        "image_id", "city", "visual_pollution_score_0_10", "rank_by_visual_pollution", "q_score_0_10",
    ]].copy()
    feature_table = metadata.copy()
    output_schema_rows: list[dict] = []
    definitions: list[dict] = []
    for _, row in core_schema.iterrows():
        column = f"feature_core_{int(row.class_id):03d}"
        feature_table[column] = table[row.feature_column]
        output_schema_rows.append({
            "feature_column": column,
            "class_id": int(row.class_id),
            "class_name": row.class_name,
            "source_column": row.feature_column,
            "feature_type": "core_10pct",
        })
        definitions.append({"feature_column": column, "member_class_id": int(row.class_id), "member_class_name": row.class_name})

    rare_schema = schema[~schema.index.isin(core_schema.index)].copy()
    rare_schema["rare_group"] = rare_schema["class_name"].map(rare_group)
    for group, members in rare_schema.groupby("rare_group", sort=True):
        column = f"feature_group_{group}"
        feature_table[column] = table[members.feature_column.tolist()].sum(axis=1)
        output_schema_rows.append({
            "feature_column": column,
            "class_id": -1,
            "class_name": group.replace("_", " ").title(),
            "source_column": ";".join(members.feature_column.tolist()),
            "feature_type": "grouped_rare",
        })
        for _, member in members.iterrows():
            definitions.append({"feature_column": column, "member_class_id": int(member.class_id), "member_class_name": member.class_name})

    generated_columns = [row["feature_column"] for row in output_schema_rows]
    if not (feature_table[generated_columns].sum(axis=1).sub(1.0).abs() < 1e-6).all():
        raise RuntimeError("Core and grouped features do not close to one")
    feature_table.to_csv(output / "elastic_net_ready.csv", index=False)
    pd.DataFrame(output_schema_rows).to_csv(output / "elastic_net_feature_schema.csv", index=False)
    pd.DataFrame(definitions).to_csv(output / "core_group_definition.csv", index=False)
    summary = pd.DataFrame({
        "feature_column": generated_columns,
        "prevalence": [(feature_table[column] > 0).mean() for column in generated_columns],
    })
    summary.to_csv(output / "feature_prevalence.csv", index=False)
    print(f"Wrote {len(core_schema)} core classes and {len(output_schema_rows) - len(core_schema)} rare groups: {output}")


if __name__ == "__main__":
    main()
