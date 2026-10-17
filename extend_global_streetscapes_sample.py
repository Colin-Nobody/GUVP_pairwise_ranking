"""Sample 50 additional European street views from Global Streetscapes per city.

This expansion uses both KartaView and Mapillary candidates, like the archived
sample. It writes a new 750-image manifest and a separate combined 1,000-image
manifest without changing the archived 250-image experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


NEW_CITIES = (
    "Brussels", "Prague", "Madrid", "Munich", "Copenhagen",
    "Stockholm", "Helsinki", "Warsaw", "Budapest", "Zurich",
    "Edinburgh", "Oslo", "Krakow", "Tallinn", "Belgrade",
)
ALLOWED_PLATFORMS = ("driving surface", "walking surface", "cycling surface")
KEEP_COLUMNS = [
    "uuid", "source", "orig_id", "city", "city_ascii", "country", "continent",
    "lat", "lon", "datetime_local", "sequence_id", "sequence_index", "platform",
    "quality", "lighting_condition", "glare", "reflection", "weather", "pano_status",
    "projection_type", "heading", "highway", "type_highway", "name", "lanes",
    "maxspeed", "road_width", "green_view_index", "sky_view_index",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Global Streetscapes streetscapes.parquet")
    parser.add_argument("--base-manifest", type=Path, default=Path("data/images_manifest.csv"))
    parser.add_argument("--new-output", type=Path, default=Path("data/expansion_750/images_manifest_new_750.csv"))
    parser.add_argument("--combined-output", type=Path, default=Path("data/expansion_750/images_manifest_1000.csv"))
    parser.add_argument("--per-city", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--grid-size-degrees", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_city < 1 or args.grid_size_degrees <= 0:
        raise ValueError("--per-city and --grid-size-degrees must be positive")
    if args.new_output.exists() or args.combined_output.exists():
        raise FileExistsError("Expansion output exists; choose new paths to preserve prior results")
    missing = set(KEEP_COLUMNS) - set(pq.ParquetFile(args.input).schema_arrow.names)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    base = pd.read_csv(args.base_manifest)
    if len(base) != 250 or base["city"].nunique() != 5:
        raise ValueError(f"Expected the archived 250-image/five-city sample; got {len(base)} rows")

    rows = pq.read_table(args.input, columns=KEEP_COLUMNS).to_pandas()
    candidates = rows.loc[
        rows["city_ascii"].isin(NEW_CITIES)
        & rows["continent"].eq("Europe")
        & rows["source"].isin(("KartaView", "Mapillary"))
        & rows["platform"].isin(ALLOWED_PLATFORMS)
        & rows["quality"].eq("good")
        & rows["lighting_condition"].eq("day")
        & rows["glare"].eq(False)
        & rows["reflection"].eq(False)
        & rows["lat"].notna()
        & rows["lon"].notna()
        & rows["sequence_id"].notna()
    ].copy()
    candidates["grid_lat"] = (candidates["lat"] / args.grid_size_degrees).floordiv(1).astype("int64")
    candidates["grid_lon"] = (candidates["lon"] / args.grid_size_degrees).floordiv(1).astype("int64")
    candidates["grid_cell"] = candidates["grid_lat"].astype(str) + "_" + candidates["grid_lon"].astype(str)
    candidates = candidates.sample(frac=1, random_state=args.seed).drop_duplicates(["source", "sequence_id"])
    counts = candidates.groupby("city_ascii").size().reindex(NEW_CITIES, fill_value=0)
    print("Eligible unique-sequence candidates by city and source:")
    print(candidates.groupby(["city_ascii", "source"]).size().unstack(fill_value=0).reindex(NEW_CITIES, fill_value=0).to_string())
    if (counts < args.per_city).any():
        raise RuntimeError("Insufficient eligible candidates: " + ", ".join(
            f"{city}={count}" for city, count in counts[counts < args.per_city].items()
        ))

    selected: list[pd.DataFrame] = []
    for index, city in enumerate(NEW_CITIES):
        city_rows = candidates[candidates.city_ascii.eq(city)].sample(frac=1, random_state=args.seed + index)
        city_rows["grid_draw_order"] = city_rows.groupby("grid_cell", sort=False).cumcount()
        chosen = city_rows.sort_values(["grid_draw_order", "grid_cell"], kind="stable").head(args.per_city).copy()
        chosen["city"] = city
        chosen["selection_rank_within_city"] = range(1, len(chosen) + 1)
        selected.append(chosen)

    new = pd.concat(selected, ignore_index=True)
    new["filter_rules"] = (
        "source in KartaView/Mapillary; continent=Europe; platform in driving/walking/cycling surface; "
        "quality=good; lighting_condition=day; glare=false; reflection=false; "
        "one image per sequence_id; lat/lon present; grid-stratified sample"
    )
    new["random_seed"] = args.seed
    new["grid_size_degrees"] = args.grid_size_degrees
    new["source_set"] = "KartaView,Mapillary"
    new["replacement_of_uuid"] = ""
    new["replacement_reason"] = ""
    new = new.sort_values(["city", "selection_rank_within_city"], kind="stable")

    combined = pd.concat([base, new], ignore_index=True, sort=False)
    if len(combined) != 1000 or combined["uuid"].nunique() != 1000:
        raise ValueError("Combined sample must contain 1,000 unique images")
    if set(base["uuid"]) & set(new["uuid"]):
        raise ValueError("New sample overlaps the archived image IDs")

    args.new_output.parent.mkdir(parents=True, exist_ok=True)
    args.combined_output.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(args.new_output, index=False)
    combined.to_csv(args.combined_output, index=False)
    print(f"Wrote {len(new)} new rows to {args.new_output}")
    print(f"Wrote {len(combined)} combined rows to {args.combined_output}")


if __name__ == "__main__":
    main()
