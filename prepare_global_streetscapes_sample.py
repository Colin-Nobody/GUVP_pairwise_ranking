"""Create a reproducible, geographically stratified GUVP image manifest.

The input is the Global Streetscapes aggregate ``streetscapes.parquet`` file.
Start with KartaView only; pass ``--sources KartaView Mapillary`` only when a
city cannot meet its target. The manifest preserves the fields needed by the
dataset's official image downloader.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


CITIES = ("Amsterdam", "Paris", "Berlin", "Barcelona", "Rome")
ALLOWED_PLATFORMS = ("driving surface", "walking surface", "cycling surface")
SEED = 20260822
GRID_SIZE_DEGREES = 0.005
KEEP_COLUMNS = [
    "uuid", "source", "orig_id", "city", "lat", "lon", "datetime_local", "sequence_id",
    "sequence_index", "platform", "quality", "lighting_condition", "glare", "reflection",
    "weather", "pano_status", "projection_type", "heading", "highway", "type_highway",
    "name", "lanes", "maxspeed", "road_width", "green_view_index", "sky_view_index",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Global Streetscapes streetscapes.parquet")
    parser.add_argument("--output", type=Path, default=Path("data/images_manifest.csv"))
    parser.add_argument("--per-city", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--grid-size-degrees", type=float, default=GRID_SIZE_DEGREES)
    parser.add_argument("--sources", nargs="+", default=["KartaView"], choices=["KartaView", "Mapillary"])
    return parser.parse_args()


def filter_candidates(frame: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    return frame.loc[
        frame["city"].isin(CITIES)
        & frame["source"].isin(sources)
        & frame["platform"].isin(ALLOWED_PLATFORMS)
        & frame["quality"].eq("good")
        & frame["lighting_condition"].eq("day")
        & frame["glare"].eq(False)
        & frame["reflection"].eq(False)
        & frame["lat"].notna()
        & frame["lon"].notna()
        & frame["sequence_id"].notna()
    ].copy()


def make_grid(frame: pd.DataFrame, size: float) -> pd.DataFrame:
    frame["grid_lat"] = (frame["lat"] / size).floordiv(1).astype("int64")
    frame["grid_lon"] = (frame["lon"] / size).floordiv(1).astype("int64")
    frame["grid_cell"] = frame["grid_lat"].astype(str) + "_" + frame["grid_lon"].astype(str)
    return frame


def stratified_sample(city_frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    """Cycle through randomly ordered grid cells before taking a second image."""
    shuffled = city_frame.sample(frac=1, random_state=seed).copy()
    shuffled["grid_draw_order"] = shuffled.groupby("grid_cell", sort=False).cumcount()
    return (
        shuffled.sort_values(["grid_draw_order", "grid_cell"], kind="stable")
        .head(count)
        .copy()
    )


def main() -> None:
    args = parse_args()
    if args.per_city < 1 or args.grid_size_degrees <= 0:
        raise ValueError("--per-city and --grid-size-degrees must be positive")
    missing = set(KEEP_COLUMNS) - set(pq.ParquetFile(args.input).schema_arrow.names)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    raw = pq.read_table(args.input, columns=KEEP_COLUMNS).to_pandas()
    candidates = make_grid(filter_candidates(raw, args.sources), args.grid_size_degrees)
    # Keep one independently selected frame per capture sequence before sampling.
    candidates = candidates.sample(frac=1, random_state=args.seed).drop_duplicates(["source", "sequence_id"]).copy()

    counts = candidates.groupby("city", sort=False).size().reindex(CITIES, fill_value=0)
    source_counts = candidates.groupby(["city", "source"], sort=False).size().unstack(fill_value=0).reindex(CITIES, fill_value=0)
    print("Candidates after quality filters and per-source sequence de-duplication:")
    for city, available in counts.items():
        print(f"  {city}: {available}")
    print("By source:")
    print(source_counts.to_string())
    short = counts[counts < args.per_city]
    if not short.empty:
        raise RuntimeError(
            "Insufficient candidates for the requested source set. "
            + ", ".join(f"{city}={available}" for city, available in short.items())
        )

    chosen = pd.concat(
        [stratified_sample(candidates[candidates.city.eq(city)], args.per_city, args.seed + index)
         for index, city in enumerate(CITIES)],
        ignore_index=True,
    )
    chosen["selection_rank_within_city"] = chosen.groupby("city", sort=False).cumcount() + 1
    chosen["filter_rules"] = (
        f"source in {','.join(args.sources)}; platform in driving/walking/cycling surface; quality=good; "
        "lighting_condition=day; glare=false; reflection=false; one image per sequence_id; "
        "lat/lon present; grid-stratified sample"
    )
    chosen["random_seed"] = args.seed
    chosen["grid_size_degrees"] = args.grid_size_degrees
    chosen["source_set"] = ",".join(args.sources)
    chosen = chosen.sort_values(["city", "selection_rank_within_city"], kind="stable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(args.output, index=False)
    print(f"Wrote {len(chosen)} rows to {args.output}")
    print(chosen.groupby("city", sort=False).size().to_string())


if __name__ == "__main__":
    main()
